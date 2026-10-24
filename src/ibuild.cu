// GPU index builder for TB2Hop.  The paper builds the index with one temporal
// BFS per vertex (both time directions).  We batch B vertices and relax them all
// at once.  State per (vertex, source) is one u64:
//   bits 60..63 hop,  bits 30..59 ts' (range-(ts-tmin)),  bits 0..29 te'
// ts and te live in the same word, so a recorded interval always belongs to one
// real path; that is what keeps the result sound.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
using u32 = unsigned int;
using u64 = unsigned long long;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %d line %d", (int)e, __LINE__); exit(1); } } while (0)

static const u64 INF64 = 0xffffffffffffffffULL;
static const u64 TSMASK = 0x3fffffff00000000ULL;

__global__ void copyk(const u64* s, u64* d, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = s[i];
}

__global__ void fillk(u64* d, long long n, u64 v) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = v;
}

// hop-1 seeds from every adjacent edge of every source in the batch
__global__ void seedk(const u32* srcs, const u32* off, const u32* dst,
                      const u32* ta, const u32* tb, const u32* ordv,
                      u64* cur, int B, u32 tmin, u32 range) {
  int q = blockIdx.y * blockDim.y + threadIdx.y;
  if (q >= B) return;
  u32 u = srcs[q];
  u32 p0 = off[u] + blockIdx.x * blockDim.x + threadIdx.x;
  if (p0 >= off[u + 1]) return;
  u32 y = dst[p0];
  if (ordv[y] <= ordv[u]) return;
  u64 tsx = (u64)(range - (ta[p0] - tmin));
  u64 tex = (u64)(tb[p0] - tmin);
  u64 cand = (1ULL << 60) | (tsx << 30) | tex;
  u64* p = &cur[(long long)y * B + q];
  if (cand < *p) atomicMin(p, cand);
}

// one relaxation round: extend every state of hop h-1 to hop h
// dir > 0 : forward  (next edge starts no earlier than the state ends)
// dir < 0 : backward (next edge ends no later than the state starts)
__global__ void relaxk(const u32* xa, const u32* ya, const u32* ta, const u32* tb,
                       const u32* ordv, const u32* usrc,
                       const u64* cur, u64* nxt, int* changed,
                       long long M, int B, u32 h, int dir, u32 tmin, u32 range) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       tid < M * (long long)B; tid += stride) {
    int q = (int)(tid % B);
    long long c = tid / B;
    u64 p = cur[(long long)xa[c] * B + q];
    if (p == INF64) continue;
    if ((int)(p >> 60) != h - 1) continue;
    u32 y = ya[c];
    if (ordv[y] <= ordv[usrc[q]]) continue;
    u32 t1 = ta[c], t2 = tb[c];
    u64 tex = p & 0x3fffffffULL;
    u64 tsx = (p >> 30) & 0x3fffffffULL;
    u64 cand;
    if (dir > 0) {
      u32 curEnd = (u32)tex + tmin;
      if ((u64)t1 < (u64)curEnd) continue;
      if ((u64)t2 < (u64)t1) continue;
      cand = ((u64)h << 60) | (tsx << 30) | (u64)(t2 - tmin);
    } else {
      u32 curStart = (u32)(range - tsx) + tmin;
      if ((u64)t2 > (u64)curStart) continue;
      if ((u64)t2 < (u64)t1) continue;
      cand = ((u64)h << 60) | ((u64)(range - (t1 - tmin)) << 30) | tex;
    }
    u64* p2 = &nxt[(long long)y * B + q];
    u64 old = atomicMin(p2, cand);
    if (cand < old) atomicAdd(changed, 1);
  }
}

int main(int argc, char** argv) {
  const char* tgf = argv[1];
  int B = argc > 2 ? atoi(argv[2]) : 4096;
  int kmax = argc > 3 ? atoi(argv[3]) : 12;
  const char* out = argc > 4 ? argv[4] : 0;
  FILE* f = fopen(tgf, "rb");
  if (!f) { printf("cannot open tg"); return 1; }
  u32 N = 0; u64 M0 = 0;
  fread(&N, 4, 1, f); fread(&M0, 8, 1, f);
  std::vector<u32> ra(M0), rb(M0), r1(M0), r2(M0);
  fread(ra.data(), 4, M0, f); fread(rb.data(), 4, M0, f);
  fread(r1.data(), 4, M0, f); fread(r2.data(), 4, M0, f);
  fclose(f);
  u32 tmin = 0xffffffffu, tmax = 0;
  for (u64 i = 0; i < M0; i++) {
    if (r1[i] < tmin) tmin = r1[i];
    if (r2[i] > tmax) tmax = r2[i];
  }
  u32 range = tmax - tmin;
  printf("[gpu-build] N=%u M=%llu tmin=%u tmax=%u range=%u", N, M0, tmin, tmax, range);
  if (range >= (1u << 30)) { printf("time range too large for the 30-bit packing"); return 1; }

  u64 M = M0 * 2;
  std::vector<u32> ex(M), ey(M), e1(M), e2(M);
  std::vector<u32> off(N + 1, 0);
  for (u64 i = 0; i < M0; i++) { off[ra[i] + 1]++; off[rb[i] + 1]++; }
  for (u32 i = 0; i < N; i++) off[i + 1] += off[i];
  std::vector<u32> pos(off.begin(), off.end() - 1);
  for (u64 i = 0; i < M0; i++) {
    u32 x = ra[i], y = rb[i];
    u32 p = pos[x]++; ex[p] = x; ey[p] = y; e1[p] = r1[i]; e2[p] = r2[i];
    p = pos[y]++;     ex[p] = y; ey[p] = x; e1[p] = r1[i]; e2[p] = r2[i];
  }
  std::vector<u32> ordv(N), vs(N), deg(N);
  for (u32 i = 0; i < N; i++) { vs[i] = i; deg[i] = off[i + 1] - off[i]; }
  std::sort(vs.begin(), vs.end(), [&](u32 a, u32 b) {
    if (deg[a] != deg[b]) return deg[a] > deg[b];
    return a < b;
  });
  for (u32 r = 0; r < N; r++) ordv[vs[r]] = r;
  u32 maxdeg = deg[vs[0]];
  printf("[gpu-build] maxdeg=%u B=%d kmax=%d state=%.2f GB", maxdeg, B, kmax,
         (double)N * B * 8 * 2 / 1e9);

  u32 *dex, *dey, *de1, *de2, *dord, *dsrc, *doff, *ddst;
  CK(cudaMalloc(&dex, M * 4)); CK(cudaMalloc(&dey, M * 4));
  CK(cudaMalloc(&de1, M * 4)); CK(cudaMalloc(&de2, M * 4));
  CK(cudaMalloc(&dord, N * 4)); CK(cudaMalloc(&dsrc, B * 4));
  CK(cudaMalloc(&doff, (N + 1) * 4)); CK(cudaMalloc(&ddst, M * 4));
  CK(cudaMemcpy(dex, ex.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dey, ey.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(de1, e1.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(de2, e2.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dord, ordv.data(), N * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(doff, off.data(), (N + 1) * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(ddst, ey.data(), M * 4, cudaMemcpyHostToDevice));
  u64 *cur, *nxt; long long SZ = (long long)N * B;
  CK(cudaMalloc(&cur, SZ * 8)); CK(cudaMalloc(&nxt, SZ * 8));
  int* dchg; CK(cudaMalloc(&dchg, 4));
  std::vector<u64> hres(SZ);
  std::vector<u32> hsrc(B);

  struct E2 { u32 w, t1, t2, k; };
  std::vector<std::vector<E2>> gIn(N), gOut(N);
  u64 totalIn = 0, totalOut = 0;
  double t0 = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  for (u32 r0 = 0; r0 < N; r0 += B) {
    int nb = (int)std::min((u32)B, N - r0);
    for (int q = 0; q < nb; q++) hsrc[q] = vs[r0 + q];
    CK(cudaMemcpy(dsrc, hsrc.data(), nb * 4, cudaMemcpyHostToDevice));
    for (int dir = 0; dir < 2; dir++) {
      int d = dir == 0 ? 1 : -1;
      fillk<<<(unsigned)((SZ + 255) / 256), 256>>>(cur, SZ, INF64);
      dim3 sb((unsigned)((maxdeg + 255) / 256), (unsigned)nb);
      seedk<<<sb, 256>>>(dsrc, doff, ddst, de1, de2, dord, cur, nb, tmin, range);
      for (u32 h = 2; h <= (u32)kmax; h++) {
        copyk<<<(unsigned)((SZ + 255) / 256), 256>>>(cur, nxt, SZ);
        int zero = 0; CK(cudaMemcpy(dchg, &zero, 4, cudaMemcpyHostToDevice));
        relaxk<<<4096, 256>>>(dex, dey, de1, de2, dord, dsrc, cur, nxt, dchg,
                              (long long)M, nb, h, d, tmin, range);
        int chg = 0; CK(cudaMemcpy(&chg, dchg, 4, cudaMemcpyDeviceToHost));
        u64* t = cur; cur = nxt; nxt = t;
        if (chg == 0) break;
      }
      CK(cudaMemcpy(hres.data(), cur, SZ * 8, cudaMemcpyDeviceToHost));
      u64 cnt = 0;
      for (long long v = 0; v < N; v++)
        for (int q = 0; q < nb; q++) {
          u64 p = hres[v * B + q];
          if (p == INF64) continue;
          if (ordv[v] <= ordv[hsrc[q]]) continue;
          u32 ts = (u32)(range - (u32)((p >> 30) & 0x3fffffff)) + tmin;
          u32 te = (u32)(p & 0x3fffffff) + tmin;
          E2 e{hsrc[q], ts, te, (u32)(p >> 60)};
          if (d > 0) gIn[v].push_back(e); else gOut[v].push_back(e);
          cnt++;
        }
      if (d > 0) totalIn += cnt; else totalOut += cnt;
    }
    if ((r0 / B) % 5 == 0) printf("  batch %u / %u", r0, N);
  }
  double el = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;
  printf("[gpu-build] time=%.3f s  entries_in=%llu entries_out=%llu total=%llu",
         el, (unsigned long long)totalIn, (unsigned long long)totalOut,
         (unsigned long long)(totalIn + totalOut));
  if (out) {
    FILE* g = fopen(out, "wb");
    fwrite(&N, 4, 1, g);
    for (u32 v = 0; v < N; v++) {
      u32 c = (u32)gIn[v].size(); fwrite(&c, 4, 1, g);
      fwrite(gIn[v].data(), sizeof(E2), c, g);
      c = (u32)gOut[v].size(); fwrite(&c, 4, 1, g);
      fwrite(gOut[v].data(), sizeof(E2), c, g);
    }
    fclose(g);
    printf("[gpu-build] index written to %s", out);
  }
  return 0;
}
