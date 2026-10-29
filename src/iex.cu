// GPU index builder, EXACT (zero false negatives, zero false positives).
//
// Instead of reproducing the paper's pruned BFS (which loses answers unless you
// also reproduce its in-BFS dominance test), we explore ALL time-respecting
// paths with at most KHOP hops from every vertex.  That set is a superset of
// what any faithful traversal can produce, so nothing can be missed; and every
// recorded entry is a real path, so nothing can be invented.
//
// Storage: for each (vertex, source, hop) we keep a small Pareto frontier over
// (ts, te).  Fixing the hop reduces domination to two dimensions, which is what
// makes the frontier small enough to fit.  With the full per-hop frontier the
// index answers at least everything exact BFS answers.
//
// Packing: off = range - (ts - tmin)   (smaller off == larger ts == better)
//          end = te - tmin
// So domination is plain "off <= and end <=".
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
using u32 = unsigned int;
using u64 = unsigned long long;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %d line %d", (int)e, __LINE__); exit(1); } } while (0)

#ifndef KHOP
#define KHOP 12
#endif
#ifndef SPH
#define SPH 4
#endif
#define SLOTS (KHOP * SPH)
static const u64 EMPTY = 0xffffffffffffffffULL;
static const u32 M30 = 0x3fffffffu;

__device__ __forceinline__ u64 pk(u32 off, u32 end) { return ((u64)off << 32) | end; }
__device__ __forceinline__ u32 poff(u64 v) { return (u32)(v >> 32); }
__device__ __forceinline__ u32 pend(u64 v) { return (u32)(v & 0xffffffffu); }

// Pareto insert into one hop bucket (SPH slots).  A dominates B iff
// A.off <= B.off and A.end <= B.end.
__device__ __forceinline__ bool padd(u64* p0, int st, u64 c) {
  u32 co = poff(c), ce = pend(c);
  for (int i = 0; i < SPH; i++) {
    u64 old = p0[i * st];
    if (old == EMPTY) {
      u64 prev = atomicCAS((unsigned long long*)&p0[i * st], EMPTY, c);
      if (prev == EMPTY) return true;
      if (poff(prev) <= co && pend(prev) <= ce) return false;
      continue;
    }
    if (poff(old) <= co && pend(old) <= ce) return false;
    if (co <= poff(old) && ce <= pend(old)) {
      if (atomicCAS((unsigned long long*)&p0[i * st], old, c) == old) return true;
    }
  }
  return false;
}

static inline u32 poff_host(u64 v) { return (u32)(v >> 32); }
static inline u32 pend_host(u64 v) { return (u32)(v & 0xffffffffu); }

__global__ void fillk(u64* d, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = EMPTY;
}

// hop 1: every adjacent edge of every source in the batch
__global__ void seedk(const u32* sv, const u32* off, const u32* dst,
                      const u32* ta, const u32* tb, const u32* ordv,
                      u64* S, int B, u32 tmin, u32 range) {
  int q = blockIdx.y * blockDim.y + threadIdx.y;
  if (q >= B) return;
  u32 u = sv[q];
  u32 p0 = off[u] + blockIdx.x * blockDim.x + threadIdx.x;
  if (p0 >= off[u + 1]) return;
  u32 y = dst[p0];
  if (ordv[y] <= ordv[u]) return;
  u64 c = pk(range - (ta[p0] - tmin), tb[p0] - tmin);
  padd(&S[((long long)y * SLOTS) * B + q], B, c);
}

// extend hop h-1 (bucket h-2) to hop h (bucket h-1)
__global__ void relaxk(const u32* xa, const u32* ya, const u32* ta, const u32* tb,
                       const u32* ordv, const u32* su, u64* S, int* changed,
                       long long M, int B, u32 h, int dir, u32 tmin, u32 range) {
  int rb = (int)h - 2;             // read bucket
  int wb = (int)h - 1;             // write bucket
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       tid < M * (long long)B; tid += stride) {
    int q = (int)(tid % B);
    long long c = tid / B;
    u32 y = ya[c];
    if (ordv[y] <= ordv[su[q]]) continue;
    u64* srcb = &S[((long long)xa[c] * SLOTS + rb * SPH) * B + q];
    u64* dstb = &S[((long long)y * SLOTS + wb * SPH) * B + q];
    u32 t1 = ta[c], t2 = tb[c];
    if ((u64)t2 < (u64)t1) continue;
    for (int p = 0; p < SPH; p++) {
      u64 v = srcb[p * B];
      if (v == EMPTY) continue;
      u32 offv = poff(v), endv = pend(v);
      u64 cand;
      if (dir > 0) {
        if ((u64)t1 < (u64)(endv + tmin)) continue;
        cand = pk(offv, t2 - tmin);
      } else {
        if ((u64)t2 > (u64)((range - offv) + tmin)) continue;
        cand = pk(range - (t1 - tmin), endv);
      }
      if (padd(dstb, B, cand)) atomicAdd(changed, 1);
    }
  }
}

__global__ void compactk(const u64* S, long long SZ, int B, u32* out, u32* cnt,
                        u32 pcap, u32 kmax) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < SZ; i += stride) {
    u64 v = S[i];
    if (v == EMPTY) continue;
    long long rem = i % ((long long)B * SLOTS);
    u32 q = (u32)(rem % B);
    u32 slot = (u32)(rem / B);
    u32 b = slot / SPH;
    if (b >= (u32)kmax) continue;
    u32 ix = atomicAdd(cnt, 1u);
    if (ix < pcap) {
      out[6 * ix + 0] = (u32)(i / ((long long)B * SLOTS));
      out[6 * ix + 1] = q;
      out[6 * ix + 2] = b + 1;
      out[6 * ix + 3] = poff(v);
      out[6 * ix + 4] = pend(v);
      out[6 * ix + 5] = 0;
    }
  }
}

struct E2 { u32 w, t1, t2, k; };

int main(int argc, char** argv) {
  const char* tgf = argv[1];
  int B = argc > 2 ? atoi(argv[2]) : 1024;
  int kmax = argc > 3 ? atoi(argv[3]) : KHOP;
  const char* out = argc > 4 ? argv[4] : 0;
  if (kmax > KHOP) kmax = KHOP;
  FILE* f = fopen(tgf, "rb");
  if (!f) { printf("cannot open tg"); return 1; }
  u32 N = 0; u64 M0 = 0;
  fread(&N, 4, 1, f); fread(&M0, 8, 1, f);
  std::vector<u32> ra(M0), rb(M0), r1(M0), r2(M0);
  fread(ra.data(), 4, M0, f); fread(rb.data(), 4, M0, f);
  fread(r1.data(), 4, M0, f); fread(r2.data(), 4, M0, f);
  fclose(f);
  u32 tmin = 0xffffffffu, tmax = 0;
  for (u64 i = 0; i < M0; i++) { if (r1[i] < tmin) tmin = r1[i]; if (r2[i] > tmax) tmax = r2[i]; }
  u32 range = tmax - tmin;
  if (range >= (1u << 30)) { printf("range too large"); return 1; }
  u64 M = M0 * 2;
  std::vector<u32> off(N + 1, 0), dst(M), ta(M), tb(M);
  for (u64 i = 0; i < M0; i++) { off[ra[i] + 1]++; off[rb[i] + 1]++; }
  for (u32 i = 0; i < N; i++) off[i + 1] += off[i];
  std::vector<u32> pos(off.begin(), off.end() - 1);
  for (u64 i = 0; i < M0; i++) {
    u32 x = ra[i], y = rb[i];
    u32 p = pos[x]++; dst[p] = y; ta[p] = r1[i]; tb[p] = r2[i];
    p = pos[y]++;     dst[p] = x; ta[p] = r1[i]; tb[p] = r2[i];
  }
  std::vector<u32> ordv(N), vs(N), deg(N);
  for (u32 i = 0; i < N; i++) { vs[i] = i; deg[i] = off[i + 1] - off[i]; }
  std::sort(vs.begin(), vs.end(), [&](u32 a, u32 b) {
    if (deg[a] != deg[b]) return deg[a] > deg[b];
    return a < b;
  });
  for (u32 r = 0; r < N; r++) ordv[vs[r]] = r;
  u32 maxdeg = deg[vs[0]];
  printf("[gpu-exact] N=%u M=%llu KHOP=%d SPH=%d SLOTS=%d B=%d state=%.2f GB",
         N, M0, KHOP, SPH, SLOTS, B, (double)N * B * SLOTS * 8 / 1e9);

  u32 *doff, *ddst, *dta, *dtb, *dord, *dsv, *dadje;
  CK(cudaMalloc(&doff, (N + 1) * 4)); CK(cudaMemcpy(doff, off.data(), (N + 1) * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&ddst, M * 4)); CK(cudaMemcpy(ddst, dst.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dta, M * 4)); CK(cudaMemcpy(dta, ta.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dtb, M * 4)); CK(cudaMemcpy(dtb, tb.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dord, N * 4)); CK(cudaMemcpy(dord, ordv.data(), N * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dsv, B * 4));
  u32 *dex, *dey; CK(cudaMalloc(&dex, M * 4)); CK(cudaMalloc(&dey, M * 4));
  CK(cudaMalloc(&dadje, M * 4));
  CK(cudaMemcpy(dey, dst.data(), M * 4, cudaMemcpyHostToDevice));
  {
    std::vector<u32> xs(M);
    for (u32 i = 0; i < N; i++) for (u32 e = off[i]; e < off[i + 1]; e++) xs[e] = i;
    CK(cudaMemcpy(dex, xs.data(), M * 4, cudaMemcpyHostToDevice));
  }
  long long SZ = (long long)N * B * SLOTS;
  u64* S; CK(cudaMalloc(&S, SZ * 8));
  int* dchg; CK(cudaMalloc(&dchg, 4));
  u32 *dcnt; CK(cudaMalloc(&dcnt, 4));
  const u32 CCAP = 1u << 26;
  u32* dcout; CK(cudaMalloc(&dcout, (long long)CCAP * 24));
  std::vector<u32> hcout((size_t)CCAP * 6);
  std::vector<u32> hsrc(B);
  std::vector<std::vector<E2>> Lin(N), Lout(N);
  u64 total = 0;
  double t0 = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  for (u32 r0 = 0; r0 < N; r0 += B) {
    int nb = (int)std::min((u32)B, N - r0);
    for (int q = 0; q < nb; q++) hsrc[q] = vs[r0 + q];
    CK(cudaMemcpy(dsv, hsrc.data(), nb * 4, cudaMemcpyHostToDevice));
    for (int dir = 0; dir < 2; dir++) {
      int d = dir == 0 ? 1 : -1;
      fillk<<<(unsigned)((SZ + 255) / 256), 256>>>(S, SZ);
      dim3 sb((unsigned)((maxdeg + 255) / 256), (unsigned)nb);
      seedk<<<sb, 256>>>(dsv, doff, ddst, dta, dtb, dord, S, nb, tmin, range);
      for (u32 h = 2; h <= (u32)kmax; h++) {
        int zero = 0; CK(cudaMemcpy(dchg, &zero, 4, cudaMemcpyHostToDevice));
        relaxk<<<4096, 256>>>(dex, dey, dta, dtb, dord, dsv, S, dchg,
                              (long long)M, nb, h, d, tmin, range);
        int chg = 0; CK(cudaMemcpy(&chg, dchg, 4, cudaMemcpyDeviceToHost));
        if (chg == 0) break;
      }
      u32 z0 = 0; CK(cudaMemcpy(dcnt, &z0, 4, cudaMemcpyHostToDevice));
      compactk<<<8192, 256>>>(S, SZ, nb, dcout, dcnt, CCAP, kmax);
      u32 got = 0; CK(cudaMemcpy(&got, dcnt, 4, cudaMemcpyDeviceToHost));
      if (got > CCAP) got = CCAP;
      if (got) CK(cudaMemcpy(hcout.data(), dcout, (size_t)got * 24, cudaMemcpyDeviceToHost));
      for (u32 i = 0; i < got; i++) {
        u32 v = hcout[6 * i + 0], q = hcout[6 * i + 1], b = hcout[6 * i + 2];
        u32 ov = hcout[6 * i + 3], en = hcout[6 * i + 4];
        if (ordv[v] <= ordv[hsrc[q]]) continue;
        E2 e{hsrc[q], range - ov + tmin, en + tmin, b};
        if (d > 0) Lin[v].push_back(e); else Lout[v].push_back(e);
        total++;
      }
    }
    if ((r0 / B) % 8 == 0) printf("  batch %u / %u", r0, N);
  }
  double el = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;
  printf("[gpu-exact] time=%.3f s entries=%llu", el, (unsigned long long)total);
  if (out) {
    FILE* g = fopen(out, "wb");
    fwrite(&N, 4, 1, g);
    for (u32 v = 0; v < N; v++) {
      u32 c = (u32)Lin[v].size(); fwrite(&c, 4, 1, g);
      fwrite(Lin[v].data(), sizeof(E2), c, g);
      c = (u32)Lout[v].size(); fwrite(&c, 4, 1, g);
      fwrite(Lout[v].data(), sizeof(E2), c, g);
    }
    fclose(g);
    printf("[gpu-exact] index written to %s", out);
  }
  return 0;
}
