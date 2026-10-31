// GPU batched k-hop temporal reachability (onlineQuery, not a label query).
// At most k contacts; next.t1 >= cur_end; all intervals stay inside [ts,te].
// Contacts are undirected; s == t is reachable with zero hops.
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cmath>
#include <exception>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
using u32 = unsigned int;
using u64 = unsigned long long;
// Arrival times need one extra value beyond the full uint32 input domain.
static const u64 INF = ULLONG_MAX;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error at line %d (%s): %s\n", __LINE__, #x, cudaGetErrorString(e)); exit(1); } } while (0)

static void fail(const char* message) {
  fprintf(stderr, "gpu-query: %s\n", message);
  exit(1);
}

static u32 integer(const char* text, u32 lo, u32 hi, const char* name) {
  char* end = nullptr;
  errno = 0;
  u64 value = strtoull(text, &end, 10);
  if (text[0] < '0' || text[0] > '9' || *end || errno == ERANGE || value < lo || value > hi) {
    fprintf(stderr, "gpu-query: invalid %s '%s' (expected %u..%u)\n", name, text, lo, hi);
    exit(1);
  }
  return (u32)value;
}

__global__ void copyk(const u64* s, u64* d, u64 n) {
  u64 stride = (u64)gridDim.x * blockDim.x;
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    d[i] = s[i];
}

// state A[vertex * S + q]; S is a power-of-two stride, got is the real count.
__global__ void relaxk(const u32* xa, const u32* ya, const u32* ta,
                       const u32* tb, const u32* qte,
                       const u64* cur, u64* nxt,
                       u64 M, u32 Smask, u32 logS, u32 got) {
  u64 stride = (u64)gridDim.x * blockDim.x;
  for (u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
       tid < (M << logS); tid += stride) {
    u64 c = tid >> logS;
    u32 q = (u32)(tid & Smask);
    if (q >= got) continue;
    u64 ce = cur[(u64)xa[c] * (Smask + 1) + q];
    if (ce == INF) continue;
    u32 t1 = ta[c], t2 = tb[c];
    if (t1 < ce || t2 > qte[q]) continue;
    // All concurrent accesses to nxt during relaxation must be atomic.
    atomicMin(&nxt[(u64)ya[c] * (Smask + 1) + q], (u64)t2);
  }
}

int main(int argc, char** argv) try {
  if (argc < 3 || argc > 8) {
    fprintf(stderr, "Usage: %s graph.tg queries.txt [k [S [reps [peakGBps [answers.txt]]]]]\n"
                    "  k=3 (0..UINT32_MAX); S=1024 (1..2^30); reps=3 (1..INT_MAX).\n"
                    "  Only the first S queries are read; no automatic multi-batch traversal.\n"
                    "  Vertices must be < N; timestamps are uint32, contact t1 <= t2.\n"
                    "  Reversed query windows are normalized; empty queries/N=0 are errors.\n"
                    "  Optional answers.txt contains one 0/1 per processed query, in input order.\n", argv[0]);
    return 1;
  }
  const char* tg = argv[1];
  const char* qf = argv[2];
  u32 k = argc > 3 ? integer(argv[3], 0, UINT_MAX, "k") : 3;
  int limit = argc > 4 ? (int)integer(argv[4], 1, 1u << 30, "S") : 1024;
  int reps = argc > 5 ? (int)integer(argv[5], 1, INT_MAX, "reps") : 3;
  double peakBW = 0.0;
  if (argc > 6) {
    char* end = nullptr;
    errno = 0;
    peakBW = strtod(argv[6], &end);
    if (end == argv[6] || *end || errno == ERANGE || !std::isfinite(peakBW) || peakBW < 0)
      fail("peakGBps must be a finite non-negative number");
  }

  FILE* f = fopen(tg, "rb");
  if (!f) { perror(tg); return 1; }
  u32 N = 0; u64 M = 0;
  if (fread(&N, sizeof(N), 1, f) != 1 || fread(&M, sizeof(M), 1, f) != 1)
    fail("truncated/unreadable graph header");
  if (!N) fail("graph must have at least one vertex");
  // Check the exact binary payload before allocating from an untrusted header.
  if (fseek(f, 0, SEEK_END) != 0) fail("cannot seek graph file");
  long bytes = ftell(f);
  if (bytes < 12 || (bytes - 12) % 16 || (u64)(bytes - 12) / 16 != M)
    fail("graph size does not match N,u64 M and four uint32[M] arrays");
  if (fseek(f, 12, SEEK_SET) != 0) fail("cannot seek graph payload");
  if (M > (u64)SIZE_MAX / (2 * sizeof(u32))) fail("expanded graph is too large");
  std::vector<u32> ra(M), rb(M), rt1(M), rt2(M);
  if (fread(ra.data(), sizeof(u32), M, f) != M ||
      fread(rb.data(), sizeof(u32), M, f) != M ||
      fread(rt1.data(), sizeof(u32), M, f) != M ||
      fread(rt2.data(), sizeof(u32), M, f) != M)
    fail("truncated/unreadable graph payload");
  if (fclose(f) != 0) fail("cannot close graph input");
  // Contacts (wedges) are symmetric: expand to both directions.
  u64 M2 = M * 2;
  std::vector<u32> ha(M2), hb(M2), ht1(M2), ht2(M2);
  for (u64 i = 0; i < M; i++) {
    if (ra[i] >= N || rb[i] >= N || rt1[i] > rt2[i]) {
      fprintf(stderr, "gpu-query: invalid contact %llu (vertex >= N or t1 > t2)\n", i);
      return 1;
    }
    ha[i] = ra[i]; hb[i] = rb[i]; ht1[i] = rt1[i]; ht2[i] = rt2[i];
    ha[M + i] = rb[i]; hb[M + i] = ra[i]; ht1[M + i] = rt1[i]; ht2[M + i] = rt2[i];
  }
  M = M2;
  std::ifstream queries(qf);
  if (!queries) { fprintf(stderr, "gpu-query: cannot open queries %s\n", qf); return 1; }
  std::vector<u32> qs, qt, qts, qte;
  int got = 0;
  while (got < limit) {
    std::string field[4];
    if (!(queries >> field[0])) {
      if (queries.eof() && !queries.bad()) break;
      fail("error reading queries");
    }
    if (!(queries >> field[1] >> field[2] >> field[3]))
      fail("incomplete/unreadable query (expected four unsigned integers)");
    u32 a = integer(field[0].c_str(), 0, N - 1, "query source");
    u32 b = integer(field[1].c_str(), 0, N - 1, "query target");
    u32 s = integer(field[2].c_str(), 0, UINT_MAX, "query start");
    u32 e = integer(field[3].c_str(), 0, UINT_MAX, "query end");
    qs.push_back(a); qt.push_back(b);
    qts.push_back(s <= e ? s : e); qte.push_back(s <= e ? e : s);
    got++;
  }
  if (!got) fail("query file is empty");
  int S = 1;
  u32 logS = 0;
  while (S < got) { S <<= 1; logS++; }
  qte.resize(S);
  u64 SZ = (u64)N * S;
  if (SZ > (u64)SIZE_MAX / sizeof(u64) || M > (u64)LLONG_MAX / S)
    fail("graph times batch dimensions overflow address/iteration range");
  printf("[gpu-query] N=%u M=%llu queries=%d S=%d query_limit=%d k=%u state=%.1fMB\n",
         N, M, got, S, limit, k, (double)SZ * sizeof(u64) * 2 / 1e6);
  printf("[gpu-query] first batch only (up to query_limit); S is padded state stride, remainder not read\n");

  u32 *dx = nullptr, *dy = nullptr, *da = nullptr, *db = nullptr, *dqe;
  if (M) {
    CK(cudaMalloc(&dx, M * sizeof(u32))); CK(cudaMalloc(&dy, M * sizeof(u32)));
    CK(cudaMalloc(&da, M * sizeof(u32))); CK(cudaMalloc(&db, M * sizeof(u32)));
    CK(cudaMemcpy(dx, ha.data(), M * sizeof(u32), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dy, hb.data(), M * sizeof(u32), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da, ht1.data(), M * sizeof(u32), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, ht2.data(), M * sizeof(u32), cudaMemcpyHostToDevice));
  }
  CK(cudaMalloc(&dqe, S * sizeof(u32)));
  CK(cudaMemcpy(dqe, qte.data(), S * sizeof(u32), cudaMemcpyHostToDevice));
  u64 *cur, *nxt;
  CK(cudaMalloc(&cur, SZ * sizeof(u64))); CK(cudaMalloc(&nxt, SZ * sizeof(u64)));
  std::vector<u64> h0(SZ, INF);
  for (int q = 0; q < got; q++) h0[(u64)qs[q] * S + q] = qts[q];
  int nt = 256;
  u64 sblocks = (SZ - 1) / nt + 1;
  if (sblocks > 2000000ULL) sblocks = 2000000ULL;
  u64 tot = M * S;
  u64 g = M ? (tot - 1) / nt + 1 : 0;
  if (g > 2000000ULL) g = 2000000ULL;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    CK(cudaMemcpy(cur, h0.data(), SZ * sizeof(u64), cudaMemcpyHostToDevice));
    CK(cudaEventRecord(e0));
    for (u32 h = 0; h < k && M; h++) {
      copyk<<<(unsigned)sblocks, nt>>>(cur, nxt, SZ);
      CK(cudaGetLastError());
      relaxk<<<(unsigned)g, nt>>>(dx, dy, da, db, dqe, cur, nxt, M, S - 1, logS, got);
      CK(cudaGetLastError());
      u64* t = cur; cur = nxt; nxt = t;
    }
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms / 1000.0 < best) best = ms / 1000.0;
  }
  // Harvest the last repetition; emit only real queries, never padded lanes.
  CK(cudaMemcpy(h0.data(), cur, SZ * sizeof(u64), cudaMemcpyDeviceToHost));
  FILE* answers = nullptr;
  if (argc > 7) {
    answers = fopen(argv[7], "w");
    if (!answers) { perror(argv[7]); return 1; }
  }
  int pos = 0;
  for (int q = 0; q < got; q++) {
    bool reachable = h0[(u64)qt[q] * S + q] != INF;
    pos += reachable;
    if (answers && fprintf(answers, "%d\n", (int)reachable) < 0) fail("cannot write answers");
  }
  if (answers && fclose(answers) != 0) fail("cannot close answers output");
  printf("[gpu-query] kernel-only best %.6f s  per-query %.3f us  throughput %.0f q/s  positive=%d/%d\n",
         best, best * 1e6 / got, best > 0 ? got / best : 0.0, pos, got);
  if (peakBW > 0.0 && best > 0.0) {
    // A logical work proxy, not measured memory traffic or achieved bandwidth.
    double modeled = (double)M * got * k * 8.0 / best / 1e9;
    printf("[gpu-query] logical-work rate (8B/contact/query/hop model): %.0f GB/s-equivalent, %.0f pct of supplied peak; NOT measured bandwidth\n",
           modeled, 100.0 * modeled / peakBW);
  }
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(da)); CK(cudaFree(db));
  CK(cudaFree(dqe)); CK(cudaFree(cur)); CK(cudaFree(nxt));
  return 0;
} catch (const std::exception& error) {
  fprintf(stderr, "gpu-query: %s\n", error.what());
  return 1;
}
