// GPU batched k-hop temporal reachability.  Semantics identical to the paper's
// Algorithm 2 (onlineQuery): a path is valid when consecutive contact intervals
// are non-decreasing in time (next.t1 >= cur_end) and stay inside [ts,te].
// The batch dimension is the query: S queries are answered simultaneously.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
using u32 = unsigned int;
using u64 = unsigned long long;
static const u32 INF = 0x7fffffffu;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %d line %d", (int)e, __LINE__); exit(1); } } while (0)

__global__ void copyk(const u32* s, u32* d, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = s[i];
}

// state A[vertex * S + q]; S is a power of two
__global__ void relaxk(const u32* xa, const u32* ya, const u32* ta,
                       const u32* tb, const u32* qte,
                       const u32* cur, u32* nxt,
                       long long M, u32 Smask, u32 logS) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       tid < (M << logS); tid += stride) {
  long long c = tid >> logS;
  u32 q = (u32)(tid & Smask);
  long long x = (long long)xa[c] * (Smask + 1) + q;
  u32 ce = cur[x];
  if (ce == INF) continue;
  u32 t1 = ta[c], t2 = tb[c];
  if (t1 < ce || t2 > qte[q]) continue;
  u32* p = &nxt[(long long)ya[c] * (Smask + 1) + q];
  if (t2 < *p) atomicMin(p, t2);
  }
}

int main(int argc, char** argv) {
  const char* tg = argv[1];
  const char* qf = argv[2];
  u32 k = argc > 3 ? atoi(argv[3]) : 3;
  int S = argc > 4 ? atoi(argv[4]) : 1024;
  int reps = argc > 5 ? atoi(argv[5]) : 3;
  double peakBW = argc > 6 ? atof(argv[6]) : 0.0;
  u32 logS = 0; while ((1 << logS) < S) logS++;
  S = 1 << logS;

  FILE* f = fopen(tg, "rb");
  if (!f) { printf("cannot open tg"); return 1; }
  u32 N = 0; u64 M = 0;
  fread(&N, 4, 1, f); fread(&M, 8, 1, f);
  std::vector<u32> ra(M), rb(M), rt1(M), rt2(M);
  fread(ra.data(), 4, M, f); fread(rb.data(), 4, M, f);
  fread(rt1.data(), 4, M, f); fread(rt2.data(), 4, M, f);
  fclose(f);
  // contacts (wedges) are symmetric: expand to both directions
  u64 M2 = M * 2;
  std::vector<u32> ha(M2), hb(M2), ht1(M2), ht2(M2);
  for (u64 i = 0; i < M; i++) {
    ha[i] = ra[i]; hb[i] = rb[i]; ht1[i] = rt1[i]; ht2[i] = rt2[i];
    ha[M + i] = rb[i]; hb[M + i] = ra[i]; ht1[M + i] = rt1[i]; ht2[M + i] = rt2[i];
  }
  M = M2;
  f = fopen(qf, "rb");
  if (!f) { printf("cannot open queries"); return 1; }
  std::vector<u32> qs(S), qt(S), qts(S), qte(S);
  int got = 0;
  { u32 a, b, s, e;
    while (got < S && fscanf(f, "%u %u %u %u", &a, &b, &s, &e) == 4) {
      qs[got] = a; qt[got] = b;
      if (s <= e) { qts[got] = s; qte[got] = e; } else { qts[got] = e; qte[got] = s; }
      got++;
    }
  }
  fclose(f);
  if (got < S) { S = 1; while (S <= got / 2) S <<= 1; logS = 0; while ((1 << logS) < S) logS++; S = 1 << logS; }
  printf("[gpu-query] N=%u M=%llu S=%d k=%u state=%.1fMB", N, M, S, k, (double)N * S * 4 * 2 / 1e6);

  u32 *dx, *dy, *da, *db, *dqe;
  CK(cudaMalloc(&dx, M * 4)); CK(cudaMalloc(&dy, M * 4));
  CK(cudaMalloc(&da, M * 4)); CK(cudaMalloc(&db, M * 4));
  CK(cudaMalloc(&dqe, S * 4));
  CK(cudaMemcpy(dx, ha.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dy, hb.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(da, ht1.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, ht2.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dqe, qte.data(), S * 4, cudaMemcpyHostToDevice));
  long long SZ = (long long)N * S;
  u32 *cur, *nxt;
  CK(cudaMalloc(&cur, SZ * 4)); CK(cudaMalloc(&nxt, SZ * 4));
  std::vector<u32> h0(SZ, INF);
  for (int q = 0; q < S; q++) h0[(long long)qs[q] * S + q] = qts[q];
  int nt = 256;
  long long sblocks = (SZ + nt - 1) / nt;
  long long tot = M << logS;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    CK(cudaMemcpy(cur, h0.data(), SZ * 4, cudaMemcpyHostToDevice));
    cudaEventRecord(e0);
    for (u32 h = 0; h < k; h++) {
      copyk<<<(unsigned)sblocks, nt>>>(cur, nxt, SZ);
      long long g = (tot + nt - 1) / nt;
      if (g > 2000000LL) g = 2000000LL;
      relaxk<<<(unsigned)g, nt>>>(dx, dy, da, db, dqe, cur, nxt, (long long)M, S - 1, logS);
      u32* t = cur; cur = nxt; nxt = t;
    }
    cudaEventRecord(e1);
    CK(cudaDeviceSynchronize());
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    if (ms / 1000.0 < best) best = ms / 1000.0;
  }
  // harvest results from the buffer that holds the last round
  std::vector<u32> hres(SZ);
  CK(cudaMemcpy(hres.data(), cur, SZ * 4, cudaMemcpyDeviceToHost));
  int pos = 0;
  for (int q = 0; q < S; q++)
    if (qs[q] == qt[q] || hres[(long long)qt[q] * S + q] != INF) pos++;
  printf("[gpu-query] best %.4f s  per-query %.1f us  throughput %.0f q/s  positive=%d/%d",
         best, best * 1e6 / S, S / best, pos, S);
  if (peakBW > 0.0)
    printf("[gpu-query] achievedBW %.0f GB/s (%.0f pct of peak)",
           (double)M * S * k * 8.0 / best / 1e9,
           100.0 * ((double)M * S * k * 8.0 / best / 1e9) / peakBW);
  return 0;
}
