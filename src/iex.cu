// Sparse-frontier index builder for bounded-hop temporal 2-hop labels.
//
// The dense version allocated N*K*P*B u64 and memset+scanned all of it for every
// batch and direction: 16*N^2*K*P bytes of traffic, which dominated the runtime
// (FR: 258 TB for a 119 s build). Almost none of those cells is ever written.
//
// This version keeps two hop buffers and drives everything from a frontier list
// of the (vertex, source) pairs that actually hold data, so clearing, emitting
// and expanding all cost O(frontier) instead of O(N).
//
// Invariant: both hop buffers are empty when a batch starts. Every frontier is
// cleared exactly once, after the hop that produced it has been consumed, so no
// per-batch memset of the whole array is needed.
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
struct Entry { u32 w, t1, t2, k; };
struct Record { u32 v, q, k, start, end; };
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
__device__ __forceinline__ bool dom(u64 a, u64 b) {
  return (u32)(a >> 32) <= (u32)(b >> 32) && (u32)a <= (u32)b;
}
// A replacement always dominates what it replaces, so a witnessed dominator stays
// valid across concurrent updates, and a full scan with no dominator means the
// frontier really is full.
__device__ __forceinline__ bool insert(u64* base, u32 stride, u32 cap, u64 candidate, u32* overflow) {
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
// generation tag replaces "was this cell cleared": no memset per hop
__device__ __forceinline__ void touch(u32 v, u32 q, u32 B, u32* mark, u32 gen,
                                      u32* nV, u32* nQ, u32* nCnt) {
  u64 idx = (u64)v * B + q;
  if (atomicExch(&mark[idx], gen) != gen) {
    u32 i = atomicAdd(nCnt, 1u);
    nV[i] = v; nQ[i] = q;
  }
}

__global__ void seed(const u32* src, const u32* off, const u32* dst, const u32* ta,
                     const u32* tb, const u32* rank, u64* st, u32 B, u32 P, u32 top,
                     u32 gen, u32* fV, u32* fQ, u32* fCnt, u32* mark, u32* overflow) {
  u32 q = blockIdx.y;
  u32 u = src[q];
  for (u64 e = (u64)off[u] + blockIdx.x * blockDim.x + threadIdx.x;
       e < off[u + 1]; e += (u64)gridDim.x * blockDim.x) {
    u32 v = dst[e];
    if (rank[v] <= rank[u]) continue;
    u64 c = ((u64)(top - ta[e]) << 32) | tb[e];
    if (insert(st + (u64)v * P * B + q, B, P, c, overflow))
      touch(v, q, B, mark, gen, fV, fQ, fCnt);
  }
}
// expand the frontier over its out-edges; this replaces the full-edge scan
__global__ void relax(const u32* off, const u32* dst, const u32* ta, const u32* tb,
                      const u32* rank, const u32* src, const u64* from, u64* to,
                      const u32* fV, const u32* fQ, const u32* fCnt,
                      u32* nV, u32* nQ, u32* nCnt, u32* mark, u32 gen,
                      u32 B, u32 P, u32 top, int dir, u32* overflow) {
  u64 total = (u64)(*fCnt) * P;
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (u64)gridDim.x * blockDim.x) {
    u32 item = (u32)(i / P), p = (u32)(i % P);
    u32 v = fV[item], q = fQ[item];
    u64 a = from[(u64)v * P * B + (u64)p * B + q];
    if (a == EMPTY) continue;
    u32 startOff = (u32)(a >> 32), end = (u32)a;
    u32 need = rank[src[q]];
    for (u32 e = off[v]; e < off[v + 1]; ++e) {
      u32 y = dst[e];
      if (rank[y] <= need) continue;
      u64 c;
      if (dir > 0) {
        if (ta[e] < end) continue;
        c = ((u64)startOff << 32) | tb[e];
      } else {
        if (tb[e] > top - startOff) continue;
        c = ((u64)(top - ta[e]) << 32) | end;
      }
      if (insert(to + (u64)y * P * B + q, B, P, c, overflow))
        touch(y, q, B, mark, gen, nV, nQ, nCnt);
    }
  }
}
__global__ void emit(const u64* st, const u32* fV, const u32* fQ, const u32* fCnt,
                     u32 B, u32 P, u32 top, u32 hop, Record* out, u64* count, u64 cap) {
  u64 total = (u64)(*fCnt) * P;
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (u64)gridDim.x * blockDim.x) {
    u32 item = (u32)(i / P), p = (u32)(i % P);
    u64 a = st[(u64)fV[item] * P * B + (u64)p * B + fQ[item]];
    if (a == EMPTY) continue;
    u64 ix = atomicAdd(count, 1ULL);
    if (ix < cap) out[ix] = Record{fV[item], fQ[item], hop, top - (u32)(a >> 32), (u32)a};
  }
}
__global__ void clearCells(u64* st, const u32* fV, const u32* fQ, const u32* fCnt, u32 B, u32 P) {
  u64 total = (u64)(*fCnt) * P;
  for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (u64)gridDim.x * blockDim.x) {
    u32 item = (u32)(i / P), p = (u32)(i % P);
    st[(u64)fV[item] * P * B + (u64)p * B + fQ[item]] = EMPTY;
  }
}
int main(int argc, char** argv) {
  if (argc < 2 || argc > 5) fail("usage: iex graph.bin [batch [kmax [out.idx]]]");
  const double started = now();
  u32 B = argc > 2 ? number(argv[2]) : 16;
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
  std::vector<u32> off((size_t)N + 1, 0), dst(M0 * 2), ta(M0 * 2), tb(M0 * 2);
  for (u64 e = 0; e < M0; ++e) {
    if (a[e] >= N || b[e] >= N || t1[e] >= t2[e] || t2[e] == UINT_MAX)
      fail("invalid vertex/interval (require start<end<UINT32_MAX)");
    ++off[a[e] + 1]; ++off[b[e] + 1]; top = std::max(top, t2[e]);
  }
  for (u32 v = 0; v < N; ++v) off[v + 1] += off[v];
  auto pos = off;
  for (u64 e = 0; e < M0; ++e) {
    u32 p = pos[a[e]]++; dst[p] = b[e]; ta[p] = t1[e]; tb[p] = t2[e];
    p = pos[b[e]]++;     dst[p] = a[e]; ta[p] = t1[e]; tb[p] = t2[e];
  }
  std::vector<u32> rank(N), vs(N);
  for (u32 v = 0; v < N; ++v) vs[v] = v;
  std::sort(vs.begin(), vs.end(), [&](u32 u, u32 v) {
    u32 du = off[u + 1] - off[u], dv = off[v + 1] - off[v];
    return du != dv ? du > dv : u < v;
  });
  for (u32 r = 0; r < N; ++r) rank[vs[r]] = r;
  auto upload = [](const std::vector<u32>& v) {
    u32* p; size_t nb = v.size() * sizeof(u32);
    CK(cudaMalloc(&p, std::max(nb, sizeof(u32))));
    if (nb) CK(cudaMemcpy(p, v.data(), nb, cudaMemcpyHostToDevice));
    return p;
  };
  u32 *doff = upload(off), *ddst = upload(dst), *dta = upload(ta),
      *dtb = upload(tb), *drank = upload(rank), *dsrc;
  CK(cudaMalloc(&dsrc, (size_t)B * 4));
  u64* buf[2] = {nullptr, nullptr};
  u64 bufCells = 0;
  u32 *fV[3], *fQ[3], *fC[3];
  for (int i = 0; i < 3; ++i) {
    CK(cudaMalloc(&fV[i], (size_t)N * B * 4));
    CK(cudaMalloc(&fQ[i], (size_t)N * B * 4));
    CK(cudaMalloc(&fC[i], 4));
  }
  u32 *mark; CK(cudaMalloc(&mark, (size_t)N * B * 4));
  CK(cudaMemset(mark, 0, (size_t)N * B * 4));
  u32 *overflow; CK(cudaMalloc(&overflow, 4));
  const u64 cap = 1ULL << 24;
  Record* rec; CK(cudaMalloc(&rec, cap * sizeof(Record)));
  u64* recCnt; CK(cudaMalloc(&recCnt, 8));
  std::vector<Record> hostRec;
  std::vector<std::vector<Entry>> Lin(N), Lout(N);
  u64 totalEntries = 0; u32 maxSlots = SPH, retries = 0, gen = 0;
  const int lastSlot = (int)(K % 3);
  const double buildStart = now();
  const int trace = getenv("IEX_TRACE") ? 1 : 0;
  for (u32 first = 0; first < N;) {
    u32 nb = std::min(B, N - first), P = SPH;
    CK(cudaMemcpy(dsrc, vs.data() + first, (size_t)nb * 4, cudaMemcpyHostToDevice));
    for (int dir : {1, -1}) {
      for (;;) {
        if ((u64)N * nb > (SIZE_MAX / 8) / P) fail("state allocation size overflow");
        u64 cells = (u64)N * P * nb;
        if (cells > bufCells) {
          for (int i = 0; i < 2; ++i) if (buf[i]) { CK(cudaFree(buf[i])); buf[i] = nullptr; }
          bufCells = 0;
          size_t available, totalBytes; CK(cudaMemGetInfo(&available, &totalBytes));
          if (cells * 16 > available) {
            if (nb > 1) { nb /= 2; fprintf(stderr, "shrink batch to %u (%.2f GB free)\n", nb, (double)available / 1e9); continue; }
            fail("buffers do not fit even at batch 1; reduce KHOP or free GPU memory. No new index published");
          }
          for (int i = 0; i < 2; ++i) {
            CK(cudaMalloc(&buf[i], cells * 8));
            CK(cudaMemset(buf[i], 0xff, cells * 8));  // only on (re)allocation
          }
          bufCells = cells;
        }
        CK(cudaMemset(overflow, 0, 4));
        int cur = 0, nxt = 1;
        u32 s1 = 1 % 3;
        CK(cudaMemset(fC[s1], 0, 4));
        seed<<<dim3(32, nb), 256>>>(dsrc, doff, ddst, dta, dtb, drank, buf[cur], nb, P, top,
                                    ++gen, fV[s1], fQ[s1], fC[s1], mark, overflow);
        CK(cudaGetLastError());
        CK(cudaMemset(recCnt, 0, 8));
        if (M0) {
          emit<<<2048, 256>>>(buf[cur], fV[s1], fQ[s1], fC[s1], nb, P, top, 1, rec, recCnt, cap);
          CK(cudaGetLastError());
        }
        for (u32 h = 2; h <= K && M0; ++h) {
          u32 rs = (h - 1) % 3, ws = h % 3;  // read hop h-1, write hop h
          CK(cudaMemset(fC[ws], 0, 4));
          relax<<<8192, 256>>>(doff, ddst, dta, dtb, drank, dsrc, buf[cur], buf[nxt],
                               fV[rs], fQ[rs], fC[rs], fV[ws], fQ[ws], fC[ws], mark, ++gen,
                               nb, P, top, dir, overflow);
          CK(cudaGetLastError());
          emit<<<2048, 256>>>(buf[nxt], fV[ws], fQ[ws], fC[ws], nb, P, top, h, rec, recCnt, cap);
          CK(cudaGetLastError());
          clearCells<<<2048, 256>>>(buf[cur], fV[rs], fQ[rs], fC[rs], nb, P);
          CK(cudaGetLastError());
          std::swap(cur, nxt);
        }
        if (M0) { // last hop is still resident: clear it so the buffers are empty again
          clearCells<<<2048, 256>>>(buf[cur], fV[lastSlot], fQ[lastSlot], fC[lastSlot], nb, P);
          CK(cudaGetLastError());
        }
        CK(cudaDeviceSynchronize());
        u32 lost; CK(cudaMemcpy(&lost, overflow, 4, cudaMemcpyDeviceToHost));
        if (!lost) break;
        if (P > UINT_MAX / 2) fail("frontier capacity overflow; no new index published");
        P *= 2; ++retries; maxSlots = std::max(maxSlots, P);
        fprintf(stderr, "retry batch=%u direction=%d slots=%u\n", first, dir, P);
        for (int i = 0; i < 2; ++i) { CK(cudaFree(buf[i])); buf[i] = nullptr; }
        bufCells = 0;
      }
      u64 got; CK(cudaMemcpy(&got, recCnt, 8, cudaMemcpyDeviceToHost));
      if (got > cap) fail("record buffer overflow; lower the batch size");
      if (got) {
        hostRec.resize(got);
        CK(cudaMemcpy(hostRec.data(), rec, got * sizeof(Record), cudaMemcpyDeviceToHost));
        for (const auto& r : hostRec) {
          auto& list = dir > 0 ? Lin[r.v] : Lout[r.v];
          list.push_back(Entry{vs[first + r.q], r.start, r.end, r.k});
        }
        totalEntries += got;
      }
      if (trace) fprintf(stderr, "batch %u dir %d entries %llu\n", first, dir, (unsigned long long)got);
    }
    first += nb;
  }
  const double buildSeconds = now() - buildStart;
  if (argc > 4) {
    for (u32 v = 0; v < N; ++v)
      if (Lin[v].size() > UINT_MAX || Lout[v].size() > UINT_MAX) fail("label count exceeds file format");
    std::string temporary = std::string(argv[4]) + ".partial";
    FILE* out = fopen(temporary.c_str(), "wb");
    if (!out) fail("cannot open index output");
    u32 header[] = {UINT_MAX, 1, N, K};
    if (fwrite(header, 4, 4, out) != 4) fail("index header write failed");
    for (u32 v = 0; v < N; ++v) for (auto* list : {&Lin[v], &Lout[v]}) {
      u32 c = (u32)list->size();
      if (fwrite(&c, 4, 1, out) != 1 || fwrite(list->data(), sizeof(Entry), c, out) != c)
        fail("index write failed; partial file is not a valid index");
    }
    if (fclose(out) != 0 || rename(temporary.c_str(), argv[4]) != 0) fail("index publication failed");
  }
  printf("[gpu-index] N=%u M=%llu K=%u slots=%u retries=%u entries=%llu build_s=%.6f total_s=%.6f\n",
         N, M0, K, maxSlots, retries, totalEntries, buildSeconds, now() - started);
  return 0;
}
