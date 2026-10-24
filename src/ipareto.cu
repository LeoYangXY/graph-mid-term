// GPU index builder with a Pareto frontier per (vertex, source).
// A plain atomicMin keeps only one entry per pair, which caps recall near 50%.
// Here every pair keeps PP entries.  Each slot uses a DIFFERENT field order in
// the u64, so one atomicMin per slot maintains a different lexicographic
// minimum of the frontier: no locks, no races.
// All PP slots are also expanded, because different frontier points at a
// vertex satisfy different outgoing edges.
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
#define PP 4
#endif
static const u64 EMPTY = 0xffffffffffffffffULL;
static const u32 M30 = 0x3fffffffu;

__device__ __forceinline__ u64 pk(u32 h, u32 tsp, u32 tep, int p) {
  if (p == 0) return ((u64)h << 60) | ((u64)tsp << 30) | tep;
  if (p == 1) return ((u64)h << 60) | ((u64)tep << 30) | tsp;
  if (p == 2) return ((u64)tsp << 34) | ((u64)tep << 4) | h;
  if (p == 3) return ((u64)tep << 34) | ((u64)tsp << 4) | h;
  if (p == 4) return ((u64)tsp << 34) | ((u64)h << 30) | tep;
  return ((u64)tep << 34) | ((u64)h << 30) | tsp;
}
__device__ __forceinline__ void unpk(u64 v, int p, u32& h, u32& tsp, u32& tep) {
  if (p == 0) { h = (u32)(v >> 60); tsp = (u32)((v >> 30) & M30); tep = (u32)(v & M30); }
  else if (p == 1) { h = (u32)(v >> 60); tep = (u32)((v >> 30) & M30); tsp = (u32)(v & M30); }
  else if (p == 2) { tsp = (u32)(v >> 34); tep = (u32)((v >> 4) & M30); h = (u32)(v & 15); }
  else if (p == 3) { tep = (u32)(v >> 34); tsp = (u32)((v >> 4) & M30); h = (u32)(v & 15); }
  else if (p == 4) { tsp = (u32)(v >> 34); h = (u32)((v >> 30) & 15); tep = (u32)(v & M30); }
  else { tep = (u32)(v >> 34); h = (u32)((v >> 30) & 15); tsp = (u32)(v & M30); }
}

static inline u32 hop_of(u64 v, int p) {
  if (p == 0 || p == 1) return (u32)(v >> 60);
  if (p == 4 || p == 5) return (u32)((v >> 30) & 15);
  return (u32)(v & 15);
}
static inline u32 tsp_of(u64 v, int p) {
  if (p == 0) return (u32)((v >> 30) & M30);
  if (p == 1) return (u32)(v & M30);
  if (p == 2) return (u32)(v >> 34);
  if (p == 3) return (u32)((v >> 4) & M30);
  if (p == 4) return (u32)(v >> 34);
  return (u32)(v & M30);
}
static inline u32 tep_of(u64 v, int p) {
  if (p == 0) return (u32)(v & M30);
  if (p == 1) return (u32)((v >> 30) & M30);
  if (p == 2) return (u32)((v >> 4) & M30);
  if (p == 3) return (u32)(v >> 34);
  if (p == 4) return (u32)(v & M30);
  return (u32)(v >> 34);
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
  u32 tsp = range - (ta[p0] - tmin);
  u32 tep = tb[p0] - tmin;
  u64* base = &S[((long long)y * B + q) * PP];
  for (int p = 0; p < PP; p++) {
    u64 c = pk(1, tsp, tep, p);
    if (c < base[p]) atomicMin(&base[p], c);
  }
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
    u64* srcb = &S[((long long)xa[c] * B + q) * PP];
    u64* dstb = &S[((long long)y * B + q) * PP];
    u32 t1 = ta[c], t2 = tb[c];
    for (int pr = 0; pr < PP; pr++) {
      u64 v = srcb[pr];
      if (v == EMPTY) continue;
      u32 hh, tsp, tep;
      unpk(v, pr, hh, tsp, tep);
      if (hh != h - 1) continue;
      u32 nts, nte;
      if (dir > 0) {
        if ((u64)t1 < (u64)((u32)tep + tmin)) continue;
        nts = tsp; nte = t2 - tmin;
      } else {
        if ((u64)t2 > (u64)((u32)(range - tsp) + tmin)) continue;
        nts = range - (t1 - tmin); nte = tep;
      }
      if ((u64)t2 < (u64)t1) continue;
      for (int pw = 0; pw < PP; pw++) {
        u64 cand = pk(h, nts, nte, pw);
        if (cand < dstb[pw]) {
          u64 old = atomicMin(&dstb[pw], cand);
          if (cand < old) atomicAdd(changed, 1);
        }
      }
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
  printf("[gpu-pareto] PP=%d N=%u M=%llu maxdeg=%u B=%d state=%.2f GB",
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
          u64* base = &hres[(v * B + q) * PP];
          for (int p = 0; p < PP; p++) {
            u64 val = base[p];
            if (val == EMPTY) continue;
            u32 hh = hop_of(val, p);
            u32 ts = range - tsp_of(val, p) + tmin;
            u32 te = tep_of(val, p) + tmin;
            bool dup = false;
            for (size_t z = 0; z < (d > 0 ? gIn[v].size() : gOut[v].size()); z++) {
              const E2& o = d > 0 ? gIn[v][z] : gOut[v][z];
              if (o.w == hsrc[q] && o.t1 == ts && o.t2 == te && o.k == hh) { dup = true; break; }
            }
            if (dup) continue;
            E2 e{hsrc[q], ts, te, hh};
            if (d > 0) gIn[v].push_back(e); else gOut[v].push_back(e);
            total++;
          }
        }
    }
  }
  double el = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;
  printf("[gpu-pareto] time=%.3f s entries=%llu", el, (unsigned long long)total);
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
    printf("[gpu-pareto] index written to %s", out);
  }
  return 0;
}
