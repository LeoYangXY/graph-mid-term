// GPU index builder that reproduces the paper indexBuild+ EXACTLY.
//
// Why this file exists: src/ifront.cu runs ONE BFS per vertex and keeps PP
// Pareto slots per pair, which loses a few percent of the answers.  The paper
// algorithm runs a SEPARATE BFS from every adjacent edge of a vertex
// (2*deg(u) traversals per vertex).  Those runs are completely independent of
// each other, which is exactly the kind of parallelism a GPU wants, so we batch
// them: every source here is a (vertex, adjacent edge, direction) triple.
//
// Per source the start timestamp ts (forward) or the end timestamp te
// (backward) is CONSTANT, so the only thing the traversal carries is
// (hop, current_time).  Two u64 values are never needed.
//
// Visited: the paper marks each edge visited once per BFS run.  On the GPU we
// use a stamp array: stamp[edge] == source_index means "visited by this
// source".  Source indices are globally increasing, so there is nothing to
// clear between batches.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
using u32 = unsigned int;
using u64 = unsigned long long;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %d line %d", (int)e, __LINE__); exit(1); } } while (0)

#ifndef FPP
#define FPP 8
#endif
static const u64 EMPTY = 0xffffffffffffffffULL;

// packed state: (hop << 32) | cur      cur = te (forward) or ts (backward)
__device__ __forceinline__ u64 pk(u32 h, u32 cur) { return ((u64)h << 32) | cur; }
__device__ __forceinline__ u32 ph(u64 v) { return (u32)(v >> 32); }
__device__ __forceinline__ u32 pc(u64 v) { return (u32)(v & 0xffffffffu); }

static inline u32 ph_h(u64 v) { return (u32)(v >> 32); }
static inline u32 pc_h(u64 v) { return (u32)(v & 0xffffffffu); }

__global__ void fillk(u64* d, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = EMPTY;
}
__global__ void zero32(u32* d, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = 0;
}

// insert (hop, curPacked) into the FPP slots of one (vertex, source).
// dominated  <=>  other.hop <= hop  and  other.cur <= cur
__device__ __forceinline__ bool fins(u64* base, int st, u64 c) {
  u32 ch = ph(c), cc = pc(c);
  for (int i = 0; i < FPP; i++) {
    u64 old = base[i * st];
    if (old == EMPTY) {
      u64 prev = atomicCAS((unsigned long long*)&base[i * st], EMPTY, c);
      if (prev == EMPTY) return true;
      if (ph(prev) <= ch && pc(prev) <= cc) return false;
      continue;
    }
    if (ph(old) <= ch && pc(old) <= cc) return false;
    if (ch <= ph(old) && cc <= pc(old)) {
      if (atomicCAS((unsigned long long*)&base[i * st], old, c) == old) return true;
    }
  }
  return false;
}

// seed: every source starts at v0 with hop 1
__global__ void seedk(const u32* sv, const u32* st0, const u32* se, const u32* ecid,
                      u32* stamp, const u32* skey, u64* S, u32* frV, u32* frS,
                      u32* frCnt, int B, u32 range, int dir, int nb) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= nb) return;
  atomicExch(&stamp[ecid[se[q]]], skey[q]);   // Algorithm 4 line 11
  u32 v0 = sv[q], t0 = st0[q];
  u32 cur = dir > 0 ? t0 : (range - t0);
  u64* base = &S[(long long)v0 * FPP * B + q];
  if (fins(base, B, pk(1, cur))) {
    u32 p = atomicAdd(frCnt, 1u);
    frV[p] = v0; frS[p] = q;
  }
}

// expand one hop level.  frV/frS hold the active (vertex, source) pairs.
__global__ void expandk(const u32* off, const u32* dst, const u32* ta, const u32* tb,
                        const u32* ordv, const u32* su, u32* stamp, const u32* ecid, const u32* skey, u64* S,
                        const u32* frV, const u32* frS, const u32* frCnt,
                        u32* nfV, u32* nfS, u32* nfCnt,
                        u32 h, int dir, u32 tmin, u32 range, int B, u32 sbase) {
  long long stride = (long long)gridDim.x * blockDim.x;
  long long total = (long long)(*frCnt) * FPP;
  for (long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       tid < total; tid += stride) {
    int p = (int)(tid % FPP);
    long long fi = tid / FPP;
    u32 v = frV[fi], q = frS[fi];
    u64 st = S[((long long)v * FPP + p) * B + q];
    if (st == EMPTY) continue;
    if (ph(st) != h - 1) continue;
    u32 cur = pc(st);
    u32 u = su[q];
    u32 sv = skey[q];                 // one stamp value per processed vertex
    for (u32 e = off[v]; e < off[v + 1]; e++) {
      u32 prev = atomicExch(&stamp[ecid[e]], sv);
      if (prev == sv) continue;       // already used in this BFS run
      u32 y = dst[e];
      if (ordv[y] <= ordv[u]) continue;
      u32 t1 = ta[e], t2 = tb[e];
      u32 nc;
      if (dir > 0) {
        if ((u64)t1 < (u64)(cur + tmin)) continue;
        nc = t2 - tmin;
      } else {
        if ((u64)t2 > (u64)((range - cur) + tmin)) continue;
        nc = range - (t1 - tmin);
      }
      u64* base = &S[((long long)y * FPP) * B + q];
      bool isnew = (base[0] == EMPTY);
      if (fins(base, B, pk(h, nc))) {
        if (isnew) {
          u32 w = atomicAdd(nfCnt, 1u);
          nfV[w] = y; nfS[w] = q;
        }
      }
    }
  }
}

struct E2 { u32 w, t1, t2, k; };

int main(int argc, char** argv) {
  const char* tgf = argv[1];
  int B = argc > 2 ? atoi(argv[2]) : 1024;
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
  for (u64 i = 0; i < M0; i++) { if (r1[i] < tmin) tmin = r1[i]; if (r2[i] > tmax) tmax = r2[i]; }
  u32 range = tmax - tmin;
  if (range >= (1u << 30)) { printf("range too large"); return 1; }
  u64 M = M0 * 2;
  std::vector<u32> ex(M), ey(M), e1(M), e2(M), off(N + 1, 0);
  for (u64 i = 0; i < M0; i++) { off[ra[i] + 1]++; off[rb[i] + 1]++; }
  for (u32 i = 0; i < N; i++) off[i + 1] += off[i];
  std::vector<u32> pos(off.begin(), off.end() - 1);
  std::vector<u32> ecid(M);
  for (u64 i = 0; i < M0; i++) {
    u32 x = ra[i], y = rb[i];
    u32 p = pos[x]++; ex[p] = x; ey[p] = y; e1[p] = r1[i]; e2[p] = r2[i]; ecid[p] = (u32)i;
    p = pos[y]++;     ex[p] = y; ey[p] = x; e1[p] = r1[i]; e2[p] = r2[i]; ecid[p] = (u32)i;
  }
  std::vector<u32> ordv(N), vs(N), deg(N);
  for (u32 i = 0; i < N; i++) { vs[i] = i; deg[i] = off[i + 1] - off[i]; }
  std::sort(vs.begin(), vs.end(), [&](u32 a, u32 b) {
    if (deg[a] != deg[b]) return deg[a] > deg[b];
    return a < b;
  });
  for (u32 r = 0; r < N; r++) ordv[vs[r]] = r;

  // enumerate every (vertex, adjacent edge, direction) source
  std::vector<u32> asU, asV0, asT0, asFix, asDir, asRank, asE;
  for (int d = 1; d >= 0; d--) {
    for (u32 r = 0; r < N; r++) {
      u32 u = vs[r];
      for (u32 e = off[u]; e < off[u + 1]; e++) {
        asU.push_back(u); asV0.push_back(ey[e]);
        if (d == 1) { asT0.push_back(e2[e]); asFix.push_back(e1[e]); }
        else        { asT0.push_back(e1[e]); asFix.push_back(e2[e]); }
        asDir.push_back(d); asRank.push_back(r + 1); asE.push_back(e);
      }
    }
  }
  u64 NS = asU.size();
  printf("[gpu-faithful] FPP=%d N=%u M=%llu sources=%llu B=%d state=%.2f GB",
         FPP, N, M0, (unsigned long long)NS, B, (double)N * B * FPP * 8 / 1e9);

  u32 *doff, *ddst, *dta, *dtb, *dord, *dsu, *dsv, *dst0, *dcid, *dsk, *dse;
  CK(cudaMalloc(&doff, (N + 1) * 4)); CK(cudaMalloc(&ddst, M * 4));
  CK(cudaMalloc(&dta, M * 4)); CK(cudaMalloc(&dtb, M * 4));
  CK(cudaMalloc(&dord, N * 4)); CK(cudaMalloc(&dsu, B * 4));
  CK(cudaMalloc(&dsv, B * 4)); CK(cudaMalloc(&dsk, B * 4)); CK(cudaMalloc(&dse, B * 4)); CK(cudaMalloc(&dst0, B * 4));
  CK(cudaMalloc(&dcid, M * 4));
  CK(cudaMemcpy(dcid, ecid.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(doff, off.data(), (N + 1) * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(ddst, ey.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dta, M * 4));
  CK(cudaMemcpy(dta, e1.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dtb, M * 4));
  CK(cudaMemcpy(dtb, e2.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dord, ordv.data(), N * 4, cudaMemcpyHostToDevice));
  u32* dstamp; CK(cudaMalloc(&dstamp, M0 * 4 + 4));
  zero32<<<(unsigned)((M0 + 255) / 256), 256>>>(dstamp, (long long)M0 + 1);
  long long SZ = (long long)N * B * FPP;
  u64* S; CK(cudaMalloc(&S, SZ * 8));
  u32 *frV, *frS, *nfV, *nfS, *frC, *nfC;
  long long cap = (long long)N * B;
  CK(cudaMalloc(&frV, cap * 4)); CK(cudaMalloc(&frS, cap * 4));
  CK(cudaMalloc(&nfV, cap * 4)); CK(cudaMalloc(&nfS, cap * 4));
  CK(cudaMalloc(&frC, 4)); CK(cudaMalloc(&nfC, 4));
  std::vector<u64> hres(SZ);
  std::vector<u32> hv(cap), hs(cap);
  std::vector<E2> gIn(N * 0), gOut(N * 0);
  std::vector<std::vector<E2>> Lin(N), Lout(N);
  u64 total = 0;
  double t0 = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  for (u64 b0 = 0; b0 < NS; b0 += B) {
    int nb = (int)std::min((u64)B, NS - b0);
    int dir = asDir[b0];
    CK(cudaMemcpy(dsu, &asU[b0], nb * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsv, &asV0[b0], nb * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dst0, &asT0[b0], nb * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsk, &asRank[b0], nb * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dse, &asE[b0], nb * 4, cudaMemcpyHostToDevice));
    fillk<<<(unsigned)((SZ + 255) / 256), 256>>>(S, SZ);
    u32 z = 0;
    CK(cudaMemcpy(frC, &z, 4, cudaMemcpyHostToDevice));
    seedk<<<(unsigned)((nb + 255) / 256), 256>>>(dsv, dst0, dse, dcid, dstamp, dsk, S, frV, frS, frC, nb, range, dir, nb);
    for (u32 h = 2; h <= (u32)kmax; h++) {
      u32 fc = 0; CK(cudaMemcpy(&fc, frC, 4, cudaMemcpyDeviceToHost));
      if (fc == 0) break;
      CK(cudaMemcpy(nfC, &z, 4, cudaMemcpyHostToDevice));
      expandk<<<4096, 256>>>(doff, ddst, dta, dtb, dord, dsu, dstamp, dcid, dsk, S,
                             frV, frS, frC, nfV, nfS, nfC, h, dir, tmin, range, nb, 0u);
      u32 nc = 0; CK(cudaMemcpy(&nc, nfC, 4, cudaMemcpyDeviceToHost));
      if (nc == 0) break;
      std::swap(frV, nfV); std::swap(frS, nfS); std::swap(frC, nfC);
      if (nc > (u32)cap) { printf("frontier overflow"); return 1; }
    }
    CK(cudaMemcpy(hres.data(), S, SZ * 8, cudaMemcpyDeviceToHost));
    for (long long v = 0; v < N; v++)
      for (int q = 0; q < nb; q++) {
        u32 u = asU[b0 + q], fx = asFix[b0 + q];
        for (int p = 0; p < FPP; p++) {
          u64 val = hres[((long long)v * FPP + p) * B + q];
          if (val == EMPTY) continue;
          u32 k = ph_h(val), cur = pc_h(val);
          u32 ts, te;
          if (dir > 0) { ts = fx; te = cur + tmin; }
          else         { ts = (range - cur) + tmin; te = fx; }
          E2 e{u, ts, te, k};
          if (dir > 0) Lin[v].push_back(e); else Lout[v].push_back(e);
          total++;
        }
      }
  }
  double el = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;
  printf("[gpu-faithful] time=%.3f s entries=%llu", el, (unsigned long long)total);
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
    printf("[gpu-faithful] index written to %s", out);
  }
  return 0;
}
