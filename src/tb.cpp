#include "tb.h"
#define P(...) do { printf(__VA_ARGS__); putchar(10); } while (0)
static const u32 INF = 0x7fffffffu;

static std::vector<u32> gs;

static void do_transform(const TBigraph& g, const std::string& out, bool count_only) {
  size_t M = g.m();
  std::vector<u32> idx(M);
  for (size_t i = 0; i < M; i++) idx[i] = (u32)i;
  double t0 = now_s();
  std::sort(idx.begin(), idx.end(), [&](u32 a, u32 b) {
    if (g.ev[a] != g.ev[b]) return g.ev[a] < g.ev[b];
    return g.ts[a] < g.ts[b];
  });
  double tsort = now_s() - t0;
  {
    u32 cur = 0xffffffffu;
    for (size_t p = 0; p < M; p++) {
      u32 v = g.ev[idx[p]];
      if (p == 0 || v != cur) { cur = v; gs.push_back((u32)p); }
    }
    gs.push_back((u32)M);
  }
  size_t ng = gs.size() - 1;
  t0 = now_s();
  std::vector<u64> off(ng + 1, 0);
  #pragma omp parallel for schedule(dynamic, 256)
  for (i64 gi = 0; gi < (i64)ng; gi++) {
    u32 b = gs[gi], e = gs[gi + 1];
    u64 c = 0;
    for (u32 i = b; i < e; i++) {
      u32 tei = g.te[idx[i]];
      u32 j = i + 1;
      while (j < e && g.ts[idx[j]] <= tei) j++;
      c += (j - i - 1);
    }
    off[gi + 1] = c;
  }
  for (size_t i = 0; i < ng; i++) off[i + 1] += off[i];
  u64 total = off[ng];
  P("[transform] sort %.2fs count %.2fs contacts=%llu (%.2fx edges)",
    tsort, now_s() - t0, (unsigned long long)total, (double)total / (double)M);
  if (count_only) return;
  t0 = now_s();
  TGraph tg; tg.n = g.nu;
  tg.src.resize(total); tg.dst.resize(total);
  tg.ca.resize(total); tg.cb.resize(total);
  #pragma omp parallel for schedule(dynamic, 256)
  for (i64 gi = 0; gi < (i64)ng; gi++) {
    u32 b = gs[gi], e = gs[gi + 1];
    u64 w = off[gi];
    for (u32 i = b; i < e; i++) {
      u32 ei = idx[i], tei = g.te[ei];
      u32 j = i + 1;
      while (j < e && g.ts[idx[j]] <= tei) j++;
      for (u32 p = i + 1; p < j; p++) {
        u32 ep = idx[p];
        u64 k = w++;
        tg.src[k] = g.eu[ei]; tg.dst[k] = g.eu[ep];
        tg.ca[k] = g.ts[ep]; tg.cb[k] = std::min(tei, g.te[ep]);
      }
    }
  }
  P("[transform] emit %.2fs", now_s() - t0);
  t0 = now_s();
  FILE* f = fopen(out.c_str(), "wb");
  fwrite(&tg.n, 4, 1, f);
  u64 mm = tg.m(); fwrite(&mm, 8, 1, f);
  fwrite(tg.src.data(), 4, mm, f);
  fwrite(tg.dst.data(), 4, mm, f);
  fwrite(tg.ca.data(), 4, mm, f);
  fwrite(tg.cb.data(), 4, mm, f);
  fclose(f);
  P("[transform] write %.2fs -> %s", now_s() - t0, out.c_str());
}

static TGraph load_tg(const std::string& path) {
  TGraph t; FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "open failed"); putchar(10); exit(1); }
  u64 mm = 0;
  fread(&t.n, 4, 1, f); fread(&mm, 8, 1, f);
  t.src.resize(mm); t.dst.resize(mm); t.ca.resize(mm); t.cb.resize(mm);
  fread(t.src.data(), 4, mm, f);
  fread(t.dst.data(), 4, mm, f);
  fread(t.ca.data(), 4, mm, f);
  fread(t.cb.data(), 4, mm, f);
  fclose(f); return t;
}

struct CSR {
  u32 n = 0; size_t m = 0;
  std::vector<u32> off, dst, a, b;
};
static CSR build_csr(const TGraph& t) {
  CSR c; c.n = t.n; c.m = t.m();
  c.off.assign(t.n + 1, 0);
  for (size_t i = 0; i < c.m; i++) c.off[t.src[i] + 1]++;
  for (u32 i = 0; i < t.n; i++) c.off[i + 1] += c.off[i];
  c.dst.resize(c.m); c.a.resize(c.m); c.b.resize(c.m);
  std::vector<u32> pos(c.off.begin(), c.off.end() - 1);
  for (size_t i = 0; i < c.m; i++) {
    u32 s = t.src[i], p = pos[s]++;
    c.dst[p] = t.dst[i]; c.a[p] = t.ca[i]; c.b[p] = t.cb[i];
  }
  return c;
}

static bool one_query(const CSR& c, u32 s, u32 t, u32 ta, u32 tb, u32 k,
                      std::vector<u32>& best, std::vector<u32>& tent,
                      std::vector<u32>& mark, u32 epoch,
                      std::vector<u32>& fr, std::vector<u32>& nf,
                      std::vector<u32>& touched) {
  best[s] = ta; touched.push_back(s); fr.clear(); fr.push_back(s);
  for (u32 h = 0; h < k; h++) {
    if (fr.empty()) break;
    if (best[t] <= tb) break;
    nf.clear();
    for (u32 v : fr) {
      u32 bv = best[v];
      for (u32 p = c.off[v]; p < c.off[v + 1]; p++) {
        if (bv > c.b[p]) continue;
        u32 y = c.dst[p];
        u32 nv = bv > c.a[p] ? bv : c.a[p];
        if (mark[y] != epoch) { mark[y] = epoch; tent[y] = nv; nf.push_back(y); }
        else if (nv < tent[y]) tent[y] = nv;
      }
    }
    for (u32 y : nf) {
      if (tent[y] < best[y]) { best[y] = tent[y]; touched.push_back(y); }
    }
    fr.swap(nf);
  }
  bool ok = best[t] <= tb;
  for (u32 v : touched) best[v] = INF;
  touched.clear();
  return ok;
}

static void do_online(const TGraph& t, const std::string& qfile, u32 k, int nq) {
  CSR c = build_csr(t);
  std::vector<u32> qs, qt, qa, qb;
  {
    FILE* f = fopen(qfile.c_str(), "rb");
    if (!f) { fprintf(stderr, "no query file"); putchar(10); exit(1); }
    u32 a, b, s, e;
    while (fscanf(f, "%u %u %u %u", &a, &b, &s, &e) == 4) {
      qs.push_back(a); qt.push_back(b);
      if (s <= e) { qa.push_back(s); qb.push_back(e); } else { qa.push_back(e); qb.push_back(s); }
    }
    fclose(f);
  }
  if (nq > 0 && nq < (int)qs.size()) { qs.resize(nq); qt.resize(nq); qa.resize(nq); qb.resize(nq); }
  P("[online] N=%u M=%llu k=%u queries=%zu", c.n, (unsigned long long)c.m, k, qs.size());
  std::vector<u32> best(c.n, INF), tent(c.n), mark(c.n, 0), fr, nf, touched;
  double t0 = now_s();
  size_t pos = 0;
  for (size_t i = 0; i < qs.size(); i++) {
    u32 epoch = (u32)(i + 1);
    if (epoch == 0) { std::fill(mark.begin(), mark.end(), 0); epoch = 1; }
    if (one_query(c, qs[i], qt[i], qa[i], qb[i], k, best, tent, mark, epoch, fr, nf, touched)) pos++;
  }
  double el = now_s() - t0;
  P("[online] total %.4fs per-query %.1f us positive=%zu/%zu",
    el, el * 1e6 / qs.size(), pos, qs.size());
}

static void do_batch(const TGraph& t, u32 S, u32 k, u32 ta, u32 tb, int nthreads) {
  omp_set_num_threads(nthreads);
  size_t M = t.m(); u32 N = t.n;
  size_t SZ = (size_t)N * S;
  P("[batch] N=%u M=%llu S=%u k=%u threads=%d state=%.1fMB",
    N, (unsigned long long)M, S, k, nthreads, (double)SZ * 4 / 1e6);
  std::vector<u32> cur(SZ, INF), nxt(SZ);
  srand(12345);
  for (u32 i = 0; i < S; i++) cur[(size_t)((u64)rand() % N) * S + i] = ta;
  double t0 = now_s();
  for (u32 h = 0; h < k; h++) {
    #pragma omp parallel for
    for (i64 i = 0; i < (i64)SZ; i++) nxt[i] = cur[i];
    #pragma omp parallel for schedule(static)
    for (i64 cidx = 0; cidx < (i64)M; cidx++) {
      u32 x = t.src[cidx], y = t.dst[cidx];
      u32 a = t.ca[cidx], b = t.cb[cidx];
      const u32* cx = &cur[(size_t)x * S];
      u32* ny = &nxt[(size_t)y * S];
      for (u32 s = 0; s < S; s++) {
        u32 u = cx[s];
        if (u <= b) { u32 v = u > a ? u : a; if (v < ny[s]) ny[s] = v; }
      }
    }
    cur.swap(nxt);
  }
  double el = now_s() - t0;
  size_t reach = 0;
  for (size_t i = 0; i < SZ; i++) if (cur[i] <= tb) reach++;
  P("[batch] time %.3fs edge*src/hop=%.3e -> %.3f Gops/s reached=%zu (avg %.1f/src)",
    el, (double)M * S, (double)M * S * k / el / 1e9, reach, (double)reach / S);
}

int main(int argc, char** argv) {
  std::string cmd = argc > 1 ? argv[1] : "stats";
  std::string path = argc > 2 ? argv[2] : "";
  if (cmd == "stats") {
    TBigraph g = load(path);
    P("|U|=%u |L|=%u |E|=%zu", g.nu, g.nl, g.m());
    return 0;
  }
  if (cmd == "transform") {
    TBigraph g = load(path);
    bool co = (argc > 4 && std::string(argv[4]) == "count");
    do_transform(g, argc > 3 ? argv[3] : "/tmp/tg.bin", co);
    return 0;
  }
  if (cmd == "online") {
    TGraph t = load_tg(path);
    u32 k = argc > 4 ? atoi(argv[4]) : 3;
    int nq = argc > 5 ? atoi(argv[5]) : 0;
    do_online(t, argc > 3 ? argv[3] : "", k, nq);
    return 0;
  }
  if (cmd == "batch") {
    TGraph t = load_tg(path);
    u32 S = argc > 3 ? atoi(argv[3]) : 256;
    u32 k = argc > 4 ? atoi(argv[4]) : 3;
    u32 ta = argc > 5 ? atoi(argv[5]) : 0;
    u32 tb = argc > 6 ? atoi(argv[6]) : 0x7fffffff;
    int nt = argc > 7 ? atoi(argv[7]) : 16;
    do_batch(t, S, k, ta, tb, nt);
    return 0;
  }
  fprintf(stderr, "usage: tb stats|transform|online|batch <file> [args]");
  putchar(10);
  return 1;
}
