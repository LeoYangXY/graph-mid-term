#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
using u32 = unsigned int;
using u64 = unsigned long long;
static const u32 INF = 0x7fffffffu;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %d at %d", (int)e, __LINE__); exit(1); } } while (0)

__global__ void copyk(const u32* __restrict__ src, u32* __restrict__ dst, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

// state layout: A[vertex * S + src];  S must be a power of two
__global__ void relaxk(const u32* __restrict__ xs, const u32* __restrict__ ys,
                       const u32* __restrict__ as, const u32* __restrict__ bs,
                       const u32* __restrict__ cur, u32* __restrict__ nxt,
                       long long M, u32 Smask, u32 logS) {
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long c = tid >> logS;
  if (c >= M) return;
  u32 s = (u32)(tid & Smask);
  u32 x = xs[c], y = ys[c], a = as[c], b = bs[c];
  u32 u = cur[(long long)x * (Smask + 1) + s];
  if (u <= b) {
    u32 v = u > a ? u : a;
    u32* p = &nxt[(long long)y * (Smask + 1) + s];
    if (v < *p) atomicMin(p, v);
  }
}

int main(int argc, char** argv) {
  const char* path = argv[1];
  u32 S = argc > 2 ? atoi(argv[2]) : 256;
  u32 k = argc > 3 ? atoi(argv[3]) : 3;
  u32 ta = argc > 4 ? atoi(argv[4]) : 0;
  int reps = argc > 5 ? atoi(argv[5]) : 1;
  u32 logS = 0; while ((1u << logS) < S) logS++;
  S = 1u << logS;

  FILE* f = fopen(path, "rb");
  if (!f) { printf("open failed"); return 1; }
  u32 N = 0; u64 M = 0;
  fread(&N, 4, 1, f); fread(&M, 8, 1, f);
  std::vector<u32> hs(M), hd(M), ha(M), hb(M);
  fread(hs.data(), 4, M, f);
  fread(hd.data(), 4, M, f);
  fread(ha.data(), 4, M, f);
  fread(hb.data(), 4, M, f);
  fclose(f);
  printf("[gpu] N=%u M=%llu S=%u k=%u state=%.1fMB contacts=%.1fMB",
         N, M, S, k, (double)N * S * 4 * 2 / 1e6, (double)M * 16 / 1e6);

  int dev = 0; cudaGetDevice(&dev);
  cudaDeviceProp pr; cudaGetDeviceProperties(&pr, dev);
  double peakBW = (argc > 6) ? atof(argv[6]) : 0.0;
  printf("[gpu] device=%s sm=%d.%d peakBW=%.0f GB/s SMs=%d", pr.name,
         pr.major, pr.minor, peakBW, pr.multiProcessorCount);
  u32 *dx, *dy, *da, *db;
  CK(cudaMalloc(&dx, M * 4)); CK(cudaMalloc(&dy, M * 4));
  CK(cudaMalloc(&da, M * 4)); CK(cudaMalloc(&db, M * 4));
  CK(cudaMemcpy(dx, hs.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dy, hd.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(da, ha.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, hb.data(), M * 4, cudaMemcpyHostToDevice));
  long long SZ = (long long)N * S;
  u32 *cur, *nxt;
  CK(cudaMalloc(&cur, SZ * 4)); CK(cudaMalloc(&nxt, SZ * 4));
  std::vector<u32> h0(SZ, INF);
  srand(12345);
  for (u32 i = 0; i < S; i++) h0[(long long)((u64)rand() % N) * S + i] = ta;
  CK(cudaMemcpy(cur, h0.data(), SZ * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(nxt, h0.data(), SZ * 4, cudaMemcpyHostToDevice));

  int nt = 256;
  long long tot = M << logS;
  long long cblocks = (M + nt - 1) / nt;
  long long sblocks = (SZ + nt - 1) / nt;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    CK(cudaMemcpy(cur, h0.data(), SZ * 4, cudaMemcpyHostToDevice));
    cudaEventRecord(e0);
    for (u32 h = 0; h < k; h++) {
      copyk<<<(unsigned)sblocks, nt>>>(cur, nxt, SZ);
      relaxk<<<(unsigned)((tot + nt - 1) / nt), nt>>>(dx, dy, da, db, cur, nxt, (long long)M, S - 1, logS);
      u32* t = cur; cur = nxt; nxt = t;
    }
    cudaEventRecord(e1);
    CK(cudaDeviceSynchronize());
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    if (ms / 1000.0 < best) best = ms / 1000.0;
    printf("[gpu] rep%d %.4fs", r, ms / 1000.0);
  }
  double achBW = (double)M * S * k * 8.0 / best / 1e9;
  if (peakBW > 0.0)
    printf("[gpu] best %.4fs -> %.2f G edge*src/s (%.2f ms/hop) achievedBW=%.0f GB/s (%.0f pct of peak)",
           best, (double)M * S * k / best / 1e9, best * 1000.0 / k, achBW, 100.0 * achBW / peakBW);
  else
    printf("[gpu] best %.4fs -> %.2f G edge*src/s (%.2f ms/hop) achievedBW=%.0f GB/s",
           best, (double)M * S * k / best / 1e9, best * 1000.0 / k, achBW);
  return 0;
}
