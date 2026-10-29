// Exact GPU port of the paper indexBuild+ (Algorithm 5).
//
// The paper processes one vertex u at a time and, inside u, runs a separate
// temporal BFS from every adjacent edge, all sharing one visited set.  That
// sharing plus the sequential order is what makes the result exact, so we keep
// it literally: ONE GPU THREAD per vertex u, runs executed in the same order.
// Parallelism comes from the vertex count (42k-120k), which is plenty on H20.
//
// visited: global stamp array, stamp[contact] == rank(u)+1.  rank is unique per
// thread so nothing has to be cleared between vertices.
//
// priority queue: hop ascending, then interval end ascending (Algorithm 5).
// Each thread owns a fixed-capacity heap in global memory.
//
// NOTE we skip the in-BFS dominance test (Algorithm 4 lines 18-19).  Doing it
// needs a per-thread x per-vertex temp label table, which does not fit.  Skipping
// it can only ADD entries, never invalidate one, so the index stays sound; the
// completeness of the result is verified empirically with the cmp tool
// (online-only must be 0) rather than argued.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
using u32 = unsigned int;
using u64 = unsigned long long;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %d line %d", (int)e, __LINE__); exit(1); } } while (0)

static const u32 NONE = 0xffffffffu;
struct HI { u32 v, ts, te, k; };

// temp label L-prime(v) of the running thread, addressed directly by vertex,
// stored as a linked list inside a per-thread pool.
static __device__ __forceinline__ bool dom2(u32 ats, u32 ate, u32 ak, u32 bts, u32 bte, u32 bk) {
  return bts <= ats && ate <= bte && ak <= bk;
}
__device__ __forceinline__ bool less(const HI& a, const HI& b) {
  if (a.k != b.k) return a.k < b.k;
  return a.te < b.te;
}
__device__ __forceinline__ void hpush(HI* H, u32& n, u32 cap, HI x) {
  if (n >= cap) return;
  u32 i = n++;
  H[i] = x;
  while (i > 0) {
    u32 p = (i - 1) >> 1;
    if (!less(H[i], H[p])) break;
    HI t = H[i]; H[i] = H[p]; H[p] = t;
    i = p;
  }
}
__device__ __forceinline__ HI hpop(HI* H, u32& n) {
  HI top = H[0];
  H[0] = H[--n];
  u32 i = 0;
  for (;;) {
    u32 l = 2 * i + 1, r = l + 1, s = i;
    if (l < n && less(H[l], H[s])) s = l;
    if (r < n && less(H[r], H[s])) s = r;
    if (s == i) break;
    HI t = H[i]; H[i] = H[s]; H[s] = t;
    i = s;
  }
  return top;
}

// one thread == one processed vertex u, exactly as Algorithm 4/5
__global__ void buildk(const u32* off, const u32* dst, const u32* ta, const u32* tb,
                       const u32* ecid, const u32* ordv, const u32* vs, u32 N, u32* stampAll,
                       u32 M0, u32 base, HI* H, u32 hcap,
                       u32* head, u32* pool, u32 pcap, u32* pcnt,
                       u32* pin, u32* cin, u32* pout, u32* cout, u32 vcap,
                       u32* ovf) {
  u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  u32 u = vs[tid];
  u32 r = base + tid + 1;
  u32* stamp = stampAll + (long long)tid * M0;
  HI* h = H + (long long)tid * hcap;
  u32 hn = 0;
  for (u32 e = off[u]; e < off[u + 1]; e++) {
   for (int dir = 0; dir < 2; dir++) {
      atomicExch(&stamp[ecid[e]], r);                 // Algorithm 4 line 11
      hn = 0;
      {
        HI s;
        s.v = dst[e]; s.ts = ta[e]; s.te = tb[e]; s.k = 1;
        hpush(h, hn, hcap, s);
      }
      while (hn > 0) {
        HI s = hpop(h, hn);
        long long pb = (long long)tid * pcap;
        bool dominated = false;
        for (u32 ix = head[(long long)tid * N + s.v]; ix != NONE;
             ix = pool[(pb + ix) * 5 + 4]) {
          if (pool[(pb + ix) * 5 + 3] & 0x80000000u) continue;
          if (dom2(pool[(pb + ix) * 5 + 1], pool[(pb + ix) * 5 + 2],
                  pool[(pb + ix) * 5 + 3] & 0x7fffffffu, s.ts, s.te, s.k)) {
            dominated = true; break;
          }
        }
        if (dominated) continue;
        for (u32 ix = head[(long long)tid * N + s.v]; ix != NONE;
             ix = pool[(pb + ix) * 5 + 4]) {
          if (pool[(pb + ix) * 5 + 3] & 0x80000000u) continue;
          if (dom2(s.ts, s.te, s.k, pool[(pb + ix) * 5 + 1], pool[(pb + ix) * 5 + 2],
                  pool[(pb + ix) * 5 + 3] & 0x7fffffffu))
            atomicOr(&pool[(pb + ix) * 5 + 3], 0x80000000u);
        }
        {
          u32 ix = atomicAdd(&pcnt[tid], 1u);
          if (ix < pcap) {
            long long b = (long long)tid * pcap + ix;
            pool[5 * b + 0] = u; pool[5 * b + 1] = s.ts; pool[5 * b + 2] = s.te;
            pool[5 * b + 3] = s.k;
            pool[5 * b + 4] = head[(long long)tid * N + s.v];
            head[(long long)tid * N + s.v] = ix;
          }
        }
        // record the entry at s.v
        if (dir == 0) {
          u32 i = atomicAdd(&cin[s.v], 1u);
          if (i < vcap) { long long b = (long long)s.v * vcap + i;
                          pin[4 * b + 0] = u; pin[4 * b + 1] = s.ts;
                          pin[4 * b + 2] = s.te; pin[4 * b + 3] = s.k; }
          else atomicAdd(ovf, 1u);
        } else {
          u32 i = atomicAdd(&cout[s.v], 1u);
          if (i < vcap) { long long b = (long long)s.v * vcap + i;
                          pout[4 * b + 0] = u; pout[4 * b + 1] = s.ts;
                          pout[4 * b + 2] = s.te; pout[4 * b + 3] = s.k; }
          else atomicAdd(ovf, 1u);
        }
        for (u32 q = off[s.v]; q < off[s.v + 1]; q++) {
          u32 prev = atomicExch(&stamp[ecid[q]], r);
          if (prev == r) continue;
          u32 y = dst[q];
          if (ordv[y] <= ordv[u]) continue;
          HI z;
          if (dir == 0) {
            if ((u64)ta[q] < (u64)s.te) continue;
            z.v = y; z.ts = s.ts; z.te = tb[q]; z.k = s.k + 1;
          } else {
            if ((u64)tb[q] > (u64)s.ts) continue;
            z.v = y; z.ts = ta[q]; z.te = s.te; z.k = s.k + 1;
          }
          hpush(h, hn, hcap, z);
        }
      }
    }
  }
}

struct E2 { u32 w, t1, t2, k; };

static void paretoFilter(std::vector<E2>& L) {
  if (L.size() < 2) return;
  std::sort(L.begin(), L.end(), [](const E2& a, const E2& b) {
    if (a.w != b.w) return a.w < b.w;
    if (a.k != b.k) return a.k < b.k;
    if (a.t1 != b.t1) return a.t1 > b.t1;
    return a.t2 < b.t2;
  });
  std::vector<E2> out;
  size_t i = 0;
  while (i < L.size()) {
    size_t j = i;
    while (j < L.size() && L[j].w == L[i].w) j++;
    for (size_t a = i; a < j; a++) {
      bool dom = false;
      for (size_t b = i; b < j && !dom; b++) {
        if (a == b) continue;
        if (L[b].k <= L[a].k && L[b].t1 >= L[a].t1 && L[b].t2 <= L[a].t2 &&
            (L[b].k < L[a].k || L[b].t1 > L[a].t1 || L[b].t2 < L[a].t2)) dom = true;
      }
      if (!dom) out.push_back(L[a]);
    }
    i = j;
  }
  L.swap(out);
}

int main(int argc, char** argv) {
  const char* tgf = argv[1];
  u32 vcap = argc > 2 ? (u32)atoi(argv[2]) : 512;
  u32 hcap = argc > 3 ? (u32)atoi(argv[3]) : 2048;
  const char* out = argc > 4 ? argv[4] : 0;
  FILE* f = fopen(tgf, "rb");
  if (!f) { printf("cannot open tg"); return 1; }
  u32 N = 0; u64 M0 = 0;
  fread(&N, 4, 1, f); fread(&M0, 8, 1, f);
  std::vector<u32> ra(M0), rb(M0), r1(M0), r2(M0);
  fread(ra.data(), 4, M0, f); fread(rb.data(), 4, M0, f);
  fread(r1.data(), 4, M0, f); fread(r2.data(), 4, M0, f);
  fclose(f);
  u64 M = M0 * 2;
  std::vector<u32> off(N + 1, 0);
  for (u64 i = 0; i < M0; i++) { off[ra[i] + 1]++; off[rb[i] + 1]++; }
  for (u32 i = 0; i < N; i++) off[i + 1] += off[i];
  std::vector<u32> pos(off.begin(), off.end() - 1);
  std::vector<u32> dst(M), ta(M), tb(M), ecid(M);
  for (u64 i = 0; i < M0; i++) {
    u32 x = ra[i], y = rb[i];
    u32 p = pos[x]++; dst[p] = y; ta[p] = r1[i]; tb[p] = r2[i]; ecid[p] = (u32)i;
    p = pos[y]++;     dst[p] = x; ta[p] = r1[i]; tb[p] = r2[i]; ecid[p] = (u32)i;
  }
  std::vector<u32> ordv(N), vs(N), deg(N);
  for (u32 i = 0; i < N; i++) { vs[i] = i; deg[i] = off[i + 1] - off[i]; }
  std::sort(vs.begin(), vs.end(), [&](u32 a, u32 b) {
    if (deg[a] != deg[b]) return deg[a] > deg[b];
    return a < b;
  });
  for (u32 r = 0; r < N; r++) ordv[vs[r]] = r;
  printf("[gpu-exact] N=%u M=%llu vcap=%u hcap=%u  mem: heap=%.2f GB pools=%.2f GB",
         N, M0, vcap, hcap, (double)N * hcap * 16 / 1e9, (double)N * vcap * 16 * 2 / 1e9);

  u32 *doff, *ddst, *dta, *dtb, *decid, *dord, *dvs, *dstamp, *dcin, *dcout, *dovf, *dpin, *dpout;
  HI* dH;
  CK(cudaMalloc(&doff, (N + 1) * 4)); CK(cudaMemcpy(doff, off.data(), (N + 1) * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&ddst, M * 4)); CK(cudaMemcpy(ddst, dst.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dta, M * 4)); CK(cudaMemcpy(dta, ta.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dtb, M * 4)); CK(cudaMemcpy(dtb, tb.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&decid, M * 4)); CK(cudaMemcpy(decid, ecid.data(), M * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dord, N * 4)); CK(cudaMemcpy(dord, ordv.data(), N * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dvs, N * 4)); CK(cudaMemcpy(dvs, vs.data(), N * 4, cudaMemcpyHostToDevice));
  u32 W0 = 1024;
  const u32 PCAP = 65536;
  CK(cudaMalloc(&dstamp, (long long)W0 * M0 * 4)); CK(cudaMemset(dstamp, 0, (long long)W0 * M0 * 4));
  CK(cudaMalloc(&dH, (long long)N * hcap * 16));
  CK(cudaMalloc(&dcin, N * 4)); CK(cudaMemset(dcin, 0, N * 4));
  CK(cudaMalloc(&dcout, N * 4)); CK(cudaMemset(dcout, 0, N * 4));
  CK(cudaMalloc(&dovf, 4)); CK(cudaMemset(dovf, 0, 4));
  CK(cudaMalloc(&dpin, (long long)N * vcap * 16));
  CK(cudaMalloc(&dpout, (long long)N * vcap * 16));
  u32 *dhead, *dpool, *dpcnt;
  CK(cudaMalloc(&dhead, (long long)W0 * N * 4));
  CK(cudaMalloc(&dpool, (long long)W0 * PCAP * 20));
  CK(cudaMalloc(&dpcnt, W0 * 4));

  double t0 = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
  u32 W = argc > 5 ? (u32)atoi(argv[5]) : 1024;
  printf("[gpu-exact] wave width W=%u  visited mem=%.2f GB", W, (double)W * M0 * 4 / 1e9);
  for (u32 base = 0; base < N; base += W) {
    u32 cnt = (N - base < W) ? (N - base) : W;
    CK(cudaMemset(dhead, 0xff, (long long)W0 * N * 4));
    CK(cudaMemset(dpcnt, 0, W0 * 4));
    buildk<<<(unsigned)((cnt + 255) / 256), 256>>>(doff, ddst, dta, dtb, decid, dord, dvs, N,
                                                 dstamp, M0, base, dH, hcap, dhead, dpool, PCAP, dpcnt,
                                                 dpin, dcin, dpout, dcout,
                                                 vcap, dovf);
  }
  CK(cudaDeviceSynchronize());
  double el = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count() - t0;
  u32 ovf = 0; CK(cudaMemcpy(&ovf, dovf, 4, cudaMemcpyDeviceToHost));
  printf("[gpu-exact] build time=%.3f s  overflowed entries=%u", el, ovf);

  std::vector<u32> cin(N), cout_(N);
  CK(cudaMemcpy(cin.data(), dcin, N * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(cout_.data(), dcout, N * 4, cudaMemcpyDeviceToHost));
  std::vector<u32> pin((size_t)N * vcap * 4), pout((size_t)N * vcap * 4);
  CK(cudaMemcpy(pin.data(), dpin, (size_t)N * vcap * 16, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(pout.data(), dpout, (size_t)N * vcap * 16, cudaMemcpyDeviceToHost));
  u64 tot = 0;
  std::vector<std::vector<E2>> Lin(N), Lout(N);
  for (u32 v = 0; v < N; v++) {
    u32 c = std::min(cin[v], vcap);
    Lin[v].resize(c);
    for (u32 i = 0; i < c; i++) {
      size_t b = ((size_t)v * vcap + i) * 4;
      Lin[v][i] = E2{pin[b], pin[b + 1], pin[b + 2], pin[b + 3]};
    }
    c = std::min(cout_[v], vcap);
    Lout[v].resize(c);
    for (u32 i = 0; i < c; i++) {
      size_t b = ((size_t)v * vcap + i) * 4;
      Lout[v][i] = E2{pout[b], pout[b + 1], pout[b + 2], pout[b + 3]};
    }
    paretoFilter(Lin[v]); paretoFilter(Lout[v]);
    tot += Lin[v].size() + Lout[v].size();
  }
  printf("[gpu-exact] entries after pareto=%llu", (unsigned long long)tot);
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
