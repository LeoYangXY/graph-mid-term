// Bounded-hop temporal 2-hop labels. Each successful build preserves every
// nondominated interval through K hops in the rank-restricted graph.
// SPH is an initial capacity, NOT an accuracy parameter: retry overflowing
// batches with more slots, or fail without publishing an incomplete index.
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
using u32 = unsigned int;
using u64 = unsigned long long;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA line %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#ifndef KHOP
#define KHOP 5
#endif
#ifndef SPH
#define SPH 128
#endif
static_assert(KHOP > 0 && SPH > 0, "positive build limits required");
static const u64 EMPTY = ULLONG_MAX;
static void fail(const char* s) { fprintf(stderr, "%s\n", s); exit(1); }
static u32 number(const char* s) {
  char* end; errno = 0; unsigned long long v = strtoull(s, &end, 10);
  if (errno || *s == '-' || *s == 0 || *end || v == 0 || v > INT_MAX)
    fail("expected a positive integer <= INT_MAX");
  return (u32)v;
}
static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
__device__ bool dom(u64 a, u64 b) {
  return (u32)(a >> 32) <= (u32)(b >> 32) && (u32)a <= (u32)b;
}
// Every replacement dominates the old value. Therefore a successful insertion
// or a witnessed dominator remains covered after concurrent replacements.
// Failed CAS retries the SAME slot; a full scan without coverage forces retry.
__device__ bool insert(u64* base, u32 stride, u32 cap, u64 candidate, u32* overflow) {
  for (u32 i = 0; i < cap; ++i) {
    u64* cell = base + (u64)i * stride;
    u64 old = atomicCAS(cell, EMPTY, EMPTY);
    for (;;) {
      if (old != EMPTY && dom(old, candidate)) return false;
      if (old != EMPTY && !dom(candidate, old)) break;
      u64 observed = atomicCAS(cell, old, candidate);
      if (observed == old) return true;
      old = observed;
    }
  }
  atomicExch(overflow, 1u);
  return false;
}
__global__ void seed(const u32* src, const u32* off, const u32* dst,
                     const u32* ta, const u32* tb, const u32* rank,
                     u64* state, u32 B, u32 P, u32 K, u32 top, u32* overflow) {
  u32 q = blockIdx.y;
  u32 u = src[q];
  for (u64 e = (u64)off[u] + blockIdx.x * blockDim.x + threadIdx.x;
       e < off[u + 1]; e += (u64)gridDim.x * blockDim.x) {
    u32 v = dst[e];
    if (rank[v] <= rank[u]) continue;
    u64 c = ((u64)(top - ta[e]) << 32) | tb[e];
    insert(state + (u64)v * K * P * B + q, B, P, c, overflow);
  }
}
__global__ void relax(const u32* x, const u32* y, const u32* ta, const u32* tb,
                      const u32* rank, const u32* src, u64* state,
                      u64 M, u32 B, u32 P, u32 K, u32 h, int dir, u32 top,
                      u32* changed, u32* overflow) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
       i < M * B; i += (u64)gridDim.x * blockDim.x) {
    u64 e = i / B; u32 q = i % B, v = y[e];
    if (rank[v] <= rank[src[q]]) continue;
    const u64* from = state + ((u64)x[e] * K + h - 2) * P * B + q;
    u64* to = state + ((u64)v * K + h - 1) * P * B + q;
    for (u32 p = 0; p < P; ++p) {
      u64 a = from[(u64)p * B]; // previous hop is immutable during this kernel
      if (a == EMPTY) continue;
      u32 startOff = a >> 32, end = (u32)a;
      u64 c;
      if (dir > 0) {
        if (ta[e] < end) continue;
        c = ((u64)startOff << 32) | tb[e];
      } else {
        if (tb[e] > top - startOff) continue;
        c = ((u64)(top - ta[e]) << 32) | end;
      }
      if (insert(to, B, P, c, overflow)) atomicExch(changed, 1u);
    }
  }
}
struct Record { u32 v, q, k, start, end; };
struct Entry { u32 w, t1, t2, k; };
// Count first, then allocate exactly enough space and emit from immutable state.
__global__ void compact(const u64* state, u64 size, u32 B, u32 P, u32 K,
                        u32 top, Record* output, u64* count) {
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
       i < size; i += (u64)gridDim.x * blockDim.x) {
    u64 a = state[i];
    if (a == EMPTY) continue;
    u64 ix = atomicAdd(count, 1ULL);
    if (output) {
      u64 slot = i / B;
      output[ix] = Record{(u32)(slot / ((u64)K * P)), (u32)(i % B),
                           (u32)((slot / P) % K + 1), top - (u32)(a >> 32), (u32)a};
    }
  }
}
int main(int argc, char** argv) {
  if (argc < 2 || argc > 5) fail("usage: iex graph.bin [batch [kmax [out.idx]]]");
  const double started = now();
  u32 B = argc > 2 ? number(argv[2]) : 32;
  u32 K = argc > 3 ? number(argv[3]) : KHOP;
  if (B > 65535 || K > KHOP) fail("batch must be <=65535; kmax must be <= compiled KHOP (no truncation)");
  FILE* f = fopen(argv[1], "rb");
  if (!f) fail("cannot open graph");
  if (fseek(f, 0, SEEK_END) != 0) fail("cannot size graph");
  long bytes = ftell(f); rewind(f);
  u32 N; u64 M0;
  if (fread(&N, 4, 1, f) != 1 || fread(&M0, 8, 1, f) != 1 || N == 0 || N == UINT_MAX ||
      M0 > UINT_MAX / 2 || bytes < 12 || (u64)(bytes - 12) != M0 * 16)
    fail("invalid/truncated graph or CSR size exceeds uint32 range");
  std::vector<u32> a(M0), b(M0), t1(M0), t2(M0);
  for (auto* v : {&a, &b, &t1, &t2})
    if (fread(v->data(), 4, M0, f) != M0) fail("truncated graph");
  fclose(f);
  u32 top = 0;
  u64 M = M0 * 2;
  std::vector<u32> off((size_t)N + 1, 0), x(M), y(M), ta(M), tb(M);
  for (u64 e = 0; e < M0; ++e) {
    if (a[e] >= N || b[e] >= N || t1[e] >= t2[e] || t2[e] == UINT_MAX)
      fail("invalid vertex/interval (require start<end<UINT32_MAX)");
    ++off[a[e] + 1]; ++off[b[e] + 1]; top = std::max(top, t2[e]);
  }
  for (u32 v = 0; v < N; ++v) off[v + 1] += off[v];
  auto pos = off;
  for (u64 e = 0; e < M0; ++e) {
    u32 p = pos[a[e]]++; x[p] = a[e]; y[p] = b[e]; ta[p] = t1[e]; tb[p] = t2[e];
    p = pos[b[e]]++; x[p] = b[e]; y[p] = a[e]; ta[p] = t1[e]; tb[p] = t2[e];
  }
  std::vector<u32> rank(N), vs(N);
  for (u32 v = 0; v < N; ++v) vs[v] = v;
  std::sort(vs.begin(), vs.end(), [&](u32 u, u32 v) {
    u32 du = off[u + 1] - off[u], dv = off[v + 1] - off[v];
    return du != dv ? du > dv : u < v;
  });
  for (u32 r = 0; r < N; ++r) rank[vs[r]] = r;
  auto upload = [](const std::vector<u32>& v) {
    u32* p; size_t bytes = v.size() * sizeof(u32);
    CK(cudaMalloc(&p, std::max(bytes, sizeof(u32))));
    if (bytes) CK(cudaMemcpy(p, v.data(), bytes, cudaMemcpyHostToDevice));
    return p;
  };
  u32 *doff = upload(off), *dx = upload(x), *dy = upload(y), *dta = upload(ta),
      *dtb = upload(tb), *drank = upload(rank), *dsrc;
  CK(cudaMalloc(&dsrc, (size_t)B * 4));
  u32 *overflow, *changed; u64* count;
  CK(cudaMalloc(&overflow, 4)); CK(cudaMalloc(&changed, 4)); CK(cudaMalloc(&count, 8));
  u64* state = nullptr; u64 stateCapacity = 0;
  std::vector<std::vector<Entry>> Lin(N), Lout(N);
  u64 total = 0; u32 maxSlots = SPH, retries = 0;
  const double buildStart = now();
  for (u32 first = 0; first < N;) {
    u32 nb = std::min(B, N - first), P = SPH;
    CK(cudaMemcpy(dsrc, vs.data() + first, (size_t)nb * 4, cudaMemcpyHostToDevice));
    for (int dir : {1, -1}) {
      u64 size;
      for (;;) {
        if ((u64)N * K > (SIZE_MAX / 8) / nb / P) fail("state allocation size overflow");
        size = (u64)N * K * P * nb;
        if (size > stateCapacity) {
          if (state) CK(cudaFree(state));
          state = nullptr;
          size_t available, deviceTotal; CK(cudaMemGetInfo(&available, &deviceTotal));
          // Shrink the batch before giving up: batches are independent, so a
          // smaller nb is still exact, just less parallel.
          if (size * 8 > available) {
            if (nb > 1) { nb /= 2; fprintf(stderr, "shrink batch to %u (%.2f GB free)\n", nb, (double)available / 1e9); continue; }
            fail("exact frontier does not fit even at batch 1; reduce KHOP or free GPU memory. No new index published");
          }
          CK(cudaMalloc(&state, size * 8)); stateCapacity = size;
        }
        CK(cudaMemset(state, 0xff, size * 8)); CK(cudaMemset(overflow, 0, 4));
        if (M) {
          seed<<<dim3(32, nb), 256>>>(dsrc, doff, dy, dta, dtb, drank, state, nb, P, K, top, overflow);
          CK(cudaGetLastError());
        }
        for (u32 h = 2; h <= K && M; ++h) {
          CK(cudaMemset(changed, 0, 4));
          relax<<<4096, 256>>>(dx, dy, dta, dtb, drank, dsrc, state, M, nb, P, K, h, dir, top, changed, overflow);
          CK(cudaGetLastError());
          u32 didChange; CK(cudaMemcpy(&didChange, changed, 4, cudaMemcpyDeviceToHost));
          if (!didChange) break;
        }
        u32 lost; CK(cudaMemcpy(&lost, overflow, 4, cudaMemcpyDeviceToHost));
        if (!lost) break;
        if (P > UINT_MAX / 2) fail("frontier capacity overflow; no new index published");
        P *= 2; ++retries; maxSlots = std::max(maxSlots, P);
        fprintf(stderr, "retry batch=%u direction=%d slots=%u\n", first, dir, P);
      }
      CK(cudaMemset(count, 0, 8));
      compact<<<4096, 256>>>(state, size, nb, P, K, top, nullptr, count);
      CK(cudaGetLastError());
      u64 got; CK(cudaMemcpy(&got, count, 8, cudaMemcpyDeviceToHost));
      if (got) {
        if (got > SIZE_MAX / sizeof(Record)) fail("output size overflow");
        Record* output; CK(cudaMalloc(&output, got * sizeof(Record)));
        CK(cudaMemset(count, 0, 8));
        compact<<<4096, 256>>>(state, size, nb, P, K, top, output, count);
        CK(cudaGetLastError());
        std::vector<Record> records(got);
        CK(cudaMemcpy(records.data(), output, got * sizeof(Record), cudaMemcpyDeviceToHost));
        CK(cudaFree(output));
        for (const auto& r : records) {
          auto& list = dir > 0 ? Lin[r.v] : Lout[r.v];
          list.push_back(Entry{vs[first + r.q], r.start, r.end, r.k});
        }
        total += got;
      }
    }
    first += nb;
  }
  const double buildSeconds = now() - buildStart;
  if (argc > 4) {
    for (u32 v = 0; v < N; ++v)
      if (Lin[v].size() > UINT_MAX || Lout[v].size() > UINT_MAX) fail("label count exceeds file format");
    // Versioned header records the query horizon; readers must reject k>K.
    // Publish only after all writes succeed, preserving any previous index on failure.
    std::string temporary = std::string(argv[4]) + ".partial";
    FILE* out = fopen(temporary.c_str(), "wb");
    if (!out) fail("cannot open index output");
    u32 header[] = {UINT_MAX, 1, N, K};
    if (fwrite(header, 4, 4, out) != 4) fail("index header write failed");
    for (u32 v = 0; v < N; ++v) for (auto* list : {&Lin[v], &Lout[v]}) {
      u32 c = list->size();
      if (fwrite(&c, 4, 1, out) != 1 || fwrite(list->data(), sizeof(Entry), c, out) != c)
        fail("index write failed; partial file is not a valid index");
    }
    if (fclose(out) != 0 || rename(temporary.c_str(), argv[4]) != 0) fail("index publication failed");
  }
  printf("[gpu-index] N=%u M=%llu K=%u slots=%u retries=%u entries=%llu build_s=%.6f total_s=%.6f\n",
         N, M0, K, maxSlots, retries, total, buildSeconds, now() - started);
  return 0;
}
