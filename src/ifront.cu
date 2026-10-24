// GPU index builder, true Pareto frontier per (vertex, source).
// Canonical packing: (hop << 60) | (tsp << 30) | tep, with
//   tsp = range - (ts - tmin), so a SMALLER tsp means a LARGER (better) ts.
// Domination: A dominates B iff A.hop <= B.hop, A.tsp <= B.tsp, A.te <= B.te.
// A dominated entry can never answer a query the dominating one cannot, so the
// frontier is a sufficient summary of all paths.  PP slots per pair, kept with
// atomicCAS: no locks.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
using u32 = unsigned int;
using u64 = unsigned long long;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %d line %d", (int)e, __LINE__); exit(1); } } while (0)

#ifndef PP
#define PP 6
#endif
static const u64 EMPTY = 0xffffffffffffffffULL;
static const u32 M30 = 0x3fffffffu;

__device__ __forceinline__ bool dom(u64 a, u64 b) {
  return ((u32)(a >> 60) <= (u32)(b >> 60)) &&
         ((u32)((a >> 30) & M30) <= (u32)((b >> 30) & M30)) &&
         ((u32)(a & M30) <= (u32)(b & M30));
}

__device__ __forceinline__ bool padd(u64* p0, int st, u64 c) {
  for (int i = 0; i < PP; i++) {
    u64 old = p0[i * st];
    if (old == EMPTY) {
      u64 prev = atomicCAS((unsigned long long*)&p0[i * st], EMPTY, c);
      if (prev == EMPTY) return true;
      if (dom(prev, c)) return false;
      continue;
    }
    if (dom(old, c)) return false;
    if (dom(c, old)) {
      if (atomicCAS((unsigned long long*)&p0[i * st], old, c) == old) return true;
    }
  }
  return false;
}

__global__ void fillk(u64* d, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = EMPTY;
}

__global__ void seedk(const u32* srcs, const u32* off, const u32* dst,
                      const u32* ta, const u32* tb, const u32* ordv,
                      u64* S, int B, u32 tmin, u32 range) {
  int q = blockIdx.y * blockDim.y + threadIdx.y;
  if (q >= B) return;
  u32 u = srcs[q];
  u32 p0 = off[u] + blockIdx.x * blockDim.x + threadIdx.x;
  if (p0 >= off[u + 1]) return;
  u32 y = dst[p0];
  if (ordv[y] <= ordv[u]) return;
  u64 c = (1ULL << 60) | ((u64)(range - (ta[p0] - tmin)) << 30) | (u64)(tb[p0] - tmin);
  padd(&S[(long long)y * PP * B + q], B, c);
}

// one round: extend every frontier point of hop h-1 to hop h
__global__ void relaxk(const u32* xa, const u32* ya, const u32* ta, const u32* tb,
                       const u32* ordv, const u32* usrc, u64* S, int* changed,
                       long long M, int B, u32 h, int dir, u32 tmin, u32 range) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       tid < M * (long long)B; tid += stride) {
    int q = (int)(tid % B);
    long long c = tid / B;
    u32 y = ya[c];
    if (ordv[y] <= ordv[usrc[q]]) continue;
    u64* srcb = &S[(long long)xa[c] * PP * B + q];
    u32 t1 = ta[c], t2 = tb[c];
    if ((u64)t2 < (u64)t1) continue;
    for (int pr = 0; pr < PP; pr++) {
      u64 v = srcb[pr * B];
      if (v == EMPTY) continue;
      if ((u32)(v >> 60) != h - 1) continue;
      u32 tsp = (u32)((v >> 30) & M30), tep = (u32)(v & M30);
      u64 cand;
      if (dir > 0) {
        if ((u64)t1 < (u64)(tep + tmin)) continue;
        cand = ((u64)h << 60) | ((u64)tsp << 30) | (u64)(t2 - tmin);
      } else {
        if ((u64)t2 > (u64)((range - tsp) + tmin)) continue;
        cand = ((u64)h << 60) | ((u64)(range - (t1 - tmin)) << 30) | (u64)tep;
      }
      u64* dstb = &S[(long long)y * PP * B + q];
      if (padd(dstb, B, cand)) atomicAdd(changed, 1);
    }
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
  if (range >= (1u << 30)) { printf("range too large"); return 1; }
  u64 M = M0 * 2;
  std::vector<u32> ex(M), ey(M), e1(M), e2(M), off(N + 1, 0);
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
  printf("[gpu-frontier] PP=%d N=%u M=%llu maxdeg=%u B=%d state=%.2f GB",
         PP, N, M0, maxdeg, B, (double)N * B * PP * 8 / 1e9);

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
  long long SZ = (long long)N * B * PP;
  u64* S; CK(cudaMalloc(&S, SZ * 8));
  int* dchg; CK(cudaMalloc(&dchg, 4));
  std::vector<u64> hres(SZ);
  std::vector<u32> hsrc(B);
  struct E2 { u32 w, t1, t2, k; };
  std::vector<std::vector<E2>> gIn(N), gOut(N);
  u64 total = 0;
  double t0 = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  for (u32 r0 = 0; r0 < N; r0 += B) {
    int nb = (int)std::min((u32)B, N - r0);
    for (int q = 0; q < nb; q++) hsrc[q] = vs[r0 + q];
    CK(cudaMemcpy(dsrc, hsrc.data(), nb * 4, cudaMemcpyHostToDevice));
    for (int dir = 0; dir < 2; dir++) {
      int d = dir == 0 ? 1 : -1;
      fillk<<<(unsigned)((SZ + 255) / 256), 256>>>(S, SZ);
      dim3 sb((unsigned)((maxdeg + 255) / 256), (unsigned)nb);
      seedk<<<sb, 256>>>(dsrc, doff, ddst, de1, de2, dord, S, nb, tmin, range);
      for (u32 h = 2; h <= (u32)kmax; h++) {
        int zero = 0; CK(cudaMemcpy(dchg, &zero, 4, cudaMemcpyHostToDevice));
        relaxk<<<4096, 256>>>(dex, dey, de1, de2, dord, dsrc, S, dchg,
                              (long long)M, nb, h, d, tmin, range);
        int chg = 0; CK(cudaMemcpy(&chg, dchg, 4, cudaMemcpyDeviceToHost));
        if (chg == 0) break;
      }
      CK(cudaMemcpy(hres.data(), S, SZ * 8, cudaMemcpyDeviceToHost));
      for (long long v = 0; v < N; v++)
        for (int q = 0; q < nb; q++) {
          if (ordv[v] <= ordv[hsrc[q]]) continue;
          u64* base = &hres[(long long)v * PP * B + q];
          for (int p = 0; p < PP; p++) {
            u64 val = base[p * B];
            if (val == EMPTY) continue;
            u32 hh = (u32)(val >> 60);
            u32 ts = range - (u32)((val >> 30) & M30) + tmin;
            u32 te = (u32)(val & M30) + tmin;
            E2 e{hsrc[q], ts, te, hh};
            if (d > 0) gIn[v].push_back(e); else gOut[v].push_back(e);
            total++;
          }
        }
    }
  }
  double el = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;
  printf("[gpu-frontier] time=%.3f s entries=%llu", el, (unsigned long long)total);
  // dedupe once at the end: sort + unique per vertex (the per-insert linear
  // scan was quadratic and stalled on epinions)
  auto eqE = [](const E2& a, const E2& b) {
    return a.w == b.w && a.t1 == b.t1 && a.t2 == b.t2 && a.k == b.k;
  };
  auto ltE = [](const E2& a, const E2& b) {
    if (a.w != b.w) return a.w < b.w;
    if (a.t1 != b.t1) return a.t1 < b.t1;
    if (a.t2 != b.t2) return a.t2 < b.t2;
    return a.k < b.k;
  };
  for (u32 v = 0; v < N; v++) {
    for (int w = 0; w < 2; w++) {
      std::vector<E2>& L = w ? gOut[v] : gIn[v];
      std::sort(L.begin(), L.end(), ltE);
      size_t j = 0;
      for (size_t i = 0; i < L.size(); i++)
        if (i == 0 || !eqE(L[i], L[j - 1])) L[j++] = L[i];
      total -= (L.size() - j);
      L.resize(j);
    }
  }
  printf("[gpu-frontier] after dedupe entries=%llu", (unsigned long long)total);
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
    printf("[gpu-frontier] index written to %s", out);
  }
  return 0;
}
