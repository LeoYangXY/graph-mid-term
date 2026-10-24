// CPU reproduction of IP&M 2025 (Zhou et al.): k-hop reachability on
// temporal bipartite graphs. Serial / single threaded, as in the paper.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <chrono>
using u32 = unsigned int;
using u64 = unsigned long long;
#define P(...) do { printf(__VA_ARGS__); putchar(10); } while (0)

static double now_s() {
  return std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

struct TB {
  u32 nu = 0, nl = 0;
  std::vector<u32> eu, ev, ts, te;
  size_t m() const { return eu.size(); }
};

static TB load(const std::string& p) {
  TB g; FILE* f = fopen(p.c_str(), "rb");
  if (!f) { P("cannot open %s", p.c_str()); exit(1); }
  char* line = nullptr; size_t cap = 0; ssize_t n;
  while ((n = getline(&line, &cap, f)) != -1) {
    if (n == 0) continue;
    if (line[0] == '#') { sscanf(line + 1, "%u %u", &g.nu, &g.nl); continue; }
    if (line[0] == 10) continue;
    u32 a, b, s, e;
    if (sscanf(line, "%u %u %u %u", &a, &b, &s, &e) != 4) continue;
    g.eu.push_back(a); g.ev.push_back(b); g.ts.push_back(s); g.te.push_back(e);
  }
  free(line); fclose(f);
  u32 mxu = 0, mxl = 0;
  for (u32 x : g.eu) if (x > mxu) mxu = x;
  for (u32 x : g.ev) if (x > mxl) mxl = x;
  if (mxu >= g.nu) g.nu = mxu + 1;
  if (mxl >= g.nl) g.nl = mxl + 1;
  return g;
}

struct TG {
  u32 n = 0;
  std::vector<u32> a, b, ta, tb;
  size_t m() const { return a.size(); }
};

// Algorithm 1: graphConvert
static TG graphConvert(const TB& g) {
  size_t M = g.m();
  std::vector<u32> idx(M);
  for (size_t i = 0; i < M; i++) idx[i] = (u32)i;
  std::sort(idx.begin(), idx.end(), [&](u32 x, u32 y) {
    if (g.ev[x] != g.ev[y]) return g.ev[x] < g.ev[y];
    return g.ts[x] < g.ts[y];
  });
  TG t; t.n = g.nu;
  for (size_t p = 0; p < M;) {
    u32 v = g.ev[idx[p]];
    size_t q = p;
    while (q < M && g.ev[idx[q]] == v) q++;
    for (size_t i = p; i < q; i++) {
      u32 ei = idx[i], tei = g.te[ei];
      for (size_t j = i + 1; j < q; j++) {
        u32 ej = idx[j];
        if (g.ts[ej] >= tei) break;
        u32 ta = g.ts[ej];
        u32 tb = tei < g.te[ej] ? tei : g.te[ej];
        if (ta < tb) {
          t.a.push_back(g.eu[ei]); t.b.push_back(g.eu[ej]);
          t.ta.push_back(ta); t.tb.push_back(tb);
        }
      }
    }
    p = q;
  }
  return t;
}

struct CSR {
  u32 n = 0; size_t m = 0;
  std::vector<u32> off, dst, ta, tb;
};
static CSR build_csr(const TG& t) {
  CSR c; c.n = t.n; c.m = t.m();
  c.off.assign(t.n + 1, 0);
  for (size_t i = 0; i < c.m; i++) { c.off[t.a[i] + 1]++; c.off[t.b[i] + 1]++; }
  for (u32 i = 0; i < t.n; i++) c.off[i + 1] += c.off[i];
  size_t nn = c.off[t.n];
  c.dst.resize(nn); c.ta.resize(nn); c.tb.resize(nn);
  std::vector<u32> pos(c.off.begin(), c.off.end() - 1);
  for (size_t i = 0; i < c.m; i++) {
    u32 x = t.a[i], y = t.b[i];
    u32 p = pos[x]++; c.dst[p] = y; c.ta[p] = t.ta[i]; c.tb[p] = t.tb[i];
    p = pos[y]++;     c.dst[p] = x; c.ta[p] = t.ta[i]; c.tb[p] = t.tb[i];
  }
  return c;
}

static const u32 NONE = 0xffffffffu;

// Algorithm 2: onlineQuery -- level synchronous (same semantics as the GPU kernel)
static bool onlineQuery(const CSR& c, u32 s, u32 t, u32 ts, u32 te, u32 k,
                        std::vector<u32>& cur, std::vector<u32>& nxt,
                        std::vector<u32>& fr, std::vector<u32>& nf) {
  if (s == t) return true;
  cur[s] = ts; fr.clear(); fr.push_back(s);
  for (u32 h = 0; h < k; h++) {
    if (fr.empty()) break;
    nxt = cur;
    nf.clear();
    for (u32 v : fr) {
      u32 ce = cur[v];
      for (u32 p = c.off[v]; p < c.off[v + 1]; p++) {
        u32 t1 = c.ta[p], t2 = c.tb[p];
        if (t1 < ce || t2 > te) continue;
        u32 y = c.dst[p];
        if (t2 < nxt[y]) {
          if (nxt[y] == cur[y]) nf.push_back(y);
          nxt[y] = t2;
        }
      }
    }
    cur.swap(nxt);
    fr.swap(nf);
  }
  bool ok = cur[t] != NONE;
  return ok;
}

struct Ent { u32 w, t1, t2, k; };
static std::vector<std::vector<Ent>> Lin, Lout;

// Algorithm 3: indexQuery, conditions C1 / C2 / C3
static bool indexQuery(u32 s, u32 t, u32 ts, u32 te, u32 k) {
  if (s == t) return true;
  for (const Ent& e : Lout[s])
    if (e.w == t && ts <= e.t1 && e.t2 <= te && e.k <= k) return true;
  for (const Ent& e : Lin[t])
    if (e.w == s && ts <= e.t1 && e.t2 <= te && e.k <= k) return true;
  for (const Ent& a : Lout[s]) {
    if (ts > a.t1 || a.t2 > te || a.k >= k) continue;
    for (const Ent& b : Lin[t]) {
      if (b.w != a.w) continue;
      if (ts <= b.t1 && b.t2 <= te && a.t2 <= b.t1 && a.k + b.k <= k) return true;
    }
  }
  return false;
}

// A dominates B iff A interval inside B interval, and A hop not larger
static bool dominates(const Ent& A, const Ent& B) {
  return B.t1 <= A.t1 && A.t2 <= B.t2 && A.k <= B.k;
}

static void tryAdd(std::vector<Ent>& lst, u32 u, u32 ts, u32 te, u32 kk) {
  Ent ne{u, ts, te, kk};
  for (const Ent& e : lst) if (e.w == u && dominates(e, ne)) return;
  size_t w = 0;
  for (size_t i = 0; i < lst.size(); i++)
    if (!(lst[i].w == u && dominates(ne, lst[i]))) lst[w++] = lst[i];
  lst.resize(w);
  lst.push_back(ne);
}

struct St { u32 v, ts, te, k; };
struct CmpF {
  bool operator()(const St& a, const St& b) const {
    if (a.k != b.k) return a.k > b.k;
    return (a.te - a.ts) > (b.te - b.ts);
  }
};
struct CmpB {
  bool operator()(const St& a, const St& b) const {
    if (a.k != b.k) return a.k > b.k;
    return (a.te - a.ts) > (b.te - b.ts);
  }
};

static std::vector<u32> ord;
static std::vector<u32> stamp;

// one temporal BFS from u; dir>0 forward in time (fills L_in of the others),
// dir<0 backward (fills L_out of the others).  tmp[v] collects entries of u.
template <class Cmp>
static void runBFS(const CSR& c, u32 u, int dir,
                   std::vector<std::vector<Ent>>& tmp,
                   std::vector<u32>& tlist, u32 stampVal) {
  std::priority_queue<St, std::vector<St>, Cmp> pq;
  for (u32 p = c.off[u]; p < c.off[u + 1]; p++) {
    u32 y = c.dst[p];
    if (ord[y] <= ord[u]) continue;
    if (stamp[p] == stampVal) continue;
    stamp[p] = stampVal;
    pq.push(St{y, c.ta[p], c.tb[p], 1});
  }
  while (!pq.empty()) {
    St s = pq.top(); pq.pop();
    u32 v = s.v;
    if (tmp[v].empty()) tlist.push_back(v);
    tryAdd(tmp[v], u, s.ts, s.te, s.k);
    for (u32 p = c.off[v]; p < c.off[v + 1]; p++) {
      if (stamp[p] == stampVal) continue;
      u32 y = c.dst[p];
      if (ord[y] <= ord[u]) continue;
      u32 t1 = c.ta[p], t2 = c.tb[p];
      if (dir > 0) {
        if (t1 >= s.te) { stamp[p] = stampVal; pq.push(St{y, s.ts, t2, s.k + 1}); }
      } else {
        if (t2 <= s.ts) { stamp[p] = stampVal; pq.push(St{y, t1, s.te, s.k + 1}); }
      }
    }
  }
}

static int gFaithful = 0;

template <class Cmp>
static void runBFS1(const CSR& c, u32 u, u32 seedEdge, int dir,
                    std::vector<std::vector<Ent>>& tmp,
                    std::vector<u32>& tlist, u32 stampVal) {
  std::priority_queue<St, std::vector<St>, Cmp> pq;
  u32 y0 = c.dst[seedEdge];
  if (ord[y0] > ord[u]) {
    stamp[seedEdge] = stampVal;
    pq.push(St{y0, c.ta[seedEdge], c.tb[seedEdge], 1});
  }
  while (!pq.empty()) {
    St s = pq.top(); pq.pop();
    u32 v = s.v;
    if (tmp[v].empty()) tlist.push_back(v);
    tryAdd(tmp[v], u, s.ts, s.te, s.k);
    for (u32 p = c.off[v]; p < c.off[v + 1]; p++) {
      if (stamp[p] == stampVal) continue;
      u32 y = c.dst[p];
      if (ord[y] <= ord[u]) continue;
      u32 t1 = c.ta[p], t2 = c.tb[p];
      if (dir > 0) {
        if (t1 >= s.te) { stamp[p] = stampVal; pq.push(St{y, s.ts, t2, s.k + 1}); }
      } else {
        if (t2 <= s.ts) { stamp[p] = stampVal; pq.push(St{y, t1, s.te, s.k + 1}); }
      }
    }
  }
}

static void indexBuildPlus(const CSR& c, u32 n) {
  Lin.assign(n, std::vector<Ent>());
  Lout.assign(n, std::vector<Ent>());
  ord.assign(n, 0);
  std::vector<u32> vs(n), deg(n);
  for (u32 i = 0; i < n; i++) { vs[i] = i; deg[i] = c.off[i + 1] - c.off[i]; }
  std::sort(vs.begin(), vs.end(), [&](u32 a, u32 b) {
    if (deg[a] != deg[b]) return deg[a] > deg[b];
    return a < b;
  });
  for (u32 r = 0; r < n; r++) ord[vs[r]] = r;
  stamp.assign(c.off[n], 0);
  std::vector<std::vector<Ent>> tmpIn(n), tmpOut(n);
  std::vector<u32> tlist;
  for (u32 r = 0; r < n; r++) {
    u32 u = vs[r];
    u32 sv = u + 1;
    if (gFaithful) {
      u32 svc = 1000000u + r;
      for (u32 p = c.off[u]; p < c.off[u + 1]; p++) runBFS1<CmpF>(c, u, p, 1, tmpIn, tlist, svc++);
      for (u32 v : tlist) { for (Ent& e : tmpIn[v]) Lin[v].push_back(e); tmpIn[v].clear(); }
      tlist.clear();
      for (u32 p = c.off[u]; p < c.off[u + 1]; p++) runBFS1<CmpB>(c, u, p, -1, tmpOut, tlist, svc++);
      for (u32 v : tlist) { for (Ent& e : tmpOut[v]) Lout[v].push_back(e); tmpOut[v].clear(); }
    } else {
    tlist.clear();
    runBFS<CmpF>(c, u, 1, tmpIn, tlist, sv);
    for (u32 v : tlist) { for (Ent& e : tmpIn[v]) Lin[v].push_back(e); tmpIn[v].clear(); }
    tlist.clear();
    runBFS<CmpB>(c, u, -1, tmpOut, tlist, sv);
    for (u32 v : tlist) { for (Ent& e : tmpOut[v]) Lout[v].push_back(e); tmpOut[v].clear(); }
    }
    if ((r & 8191) == 0) P("  built %u / %u  (%.1f pct)", r, n, 100.0 * r / n);
  }
}

static void saveTG(const TG& t, const std::string& p) {
  FILE* f = fopen(p.c_str(), "wb");
  u64 mm = t.m();
  fwrite(&t.n, 4, 1, f); fwrite(&mm, 8, 1, f);
  fwrite(t.a.data(), 4, mm, f); fwrite(t.b.data(), 4, mm, f);
  fwrite(t.ta.data(), 4, mm, f); fwrite(t.tb.data(), 4, mm, f);
  fclose(f);
}
static TG loadTG(const std::string& p) {
  TG t; FILE* f = fopen(p.c_str(), "rb");
  if (!f) { P("cannot open %s", p.c_str()); exit(1); }
  u64 mm = 0; fread(&t.n, 4, 1, f); fread(&mm, 8, 1, f);
  t.a.resize(mm); t.b.resize(mm); t.ta.resize(mm); t.tb.resize(mm);
  fread(t.a.data(), 4, mm, f); fread(t.b.data(), 4, mm, f);
  fread(t.ta.data(), 4, mm, f); fread(t.tb.data(), 4, mm, f);
  fclose(f); return t;
}

static void writeIdx(u32 n, const char* p) {
  FILE* f = fopen(p, "wb");
  fwrite(&n, 4, 1, f);
  for (u32 v = 0; v < n; v++) {
    u32 c = (u32)Lin[v].size(); fwrite(&c, 4, 1, f);
    fwrite(Lin[v].data(), sizeof(Ent), c, f);
    c = (u32)Lout[v].size(); fwrite(&c, 4, 1, f);
    fwrite(Lout[v].data(), sizeof(Ent), c, f);
  }
  fclose(f);
}
static void readIdx(const char* p, u32& n) {
  FILE* f = fopen(p, "rb");
  if (!f) { P("cannot open index"); exit(1); }
  fread(&n, 4, 1, f);
  Lin.assign(n, std::vector<Ent>()); Lout.assign(n, std::vector<Ent>());
  for (u32 v = 0; v < n; v++) {
    u32 c = 0; fread(&c, 4, 1, f); Lin[v].resize(c);
    fread(Lin[v].data(), sizeof(Ent), c, f);
    fread(&c, 4, 1, f); Lout[v].resize(c);
    fread(Lout[v].data(), sizeof(Ent), c, f);
  }
  fclose(f);
}

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  std::string path = argc > 2 ? argv[2] : "";
  if (mode == "convert") {
    TB g = load(path);
    P("|U|=%u |L|=%u |Eb|=%zu", g.nu, g.nl, g.m());
    double t0 = now_s();
    TG t = graphConvert(g);
    double el = now_s() - t0;
    P("[convert] |V|=%u |E|=%llu  time=%.4f s", t.n, (unsigned long long)t.m(), el);
    if (argc > 3) saveTG(t, argv[3]);
    return 0;
  }
  if (mode == "build") {
    TB g = load(path);
    double t0 = now_s(); TG t = graphConvert(g); double tc = now_s() - t0;
    P("[convert] |V|=%u |E|=%llu  time=%.4f s", t.n, (unsigned long long)t.m(), tc);
    t0 = now_s(); CSR c = build_csr(t); double tcc = now_s() - t0;
    P("[csr] %.4f s", tcc);
    t0 = now_s(); indexBuildPlus(c, t.n); double tb = now_s() - t0;
    u64 tot = 0;
    for (u32 i = 0; i < t.n; i++) tot += Lin[i].size() + Lout[i].size();
    P("[indexBuild+] time=%.4f s   entries=%llu  avg-label=%.1f", tb,
      (unsigned long long)tot, (double)tot / (double)t.n);
    return 0;
  }
  if (mode == "online" || mode == "query") {
    TG t = loadTG(path);
    CSR c = build_csr(t);
    std::vector<u32> qs, qt, qa, qb;
    FILE* f = fopen(argv[3], "rb");
    u32 a, b, s, e;
    while (fscanf(f, "%u %u %u %u", &a, &b, &s, &e) == 4) {
      qs.push_back(a); qt.push_back(b);
      if (s <= e) { qa.push_back(s); qb.push_back(e); } else { qa.push_back(e); qb.push_back(s); }
    }
    fclose(f);
    int nq = argc > 5 ? atoi(argv[5]) : 0;
    u32 k = argc > 4 ? atoi(argv[4]) : 3;
    if (nq > 0 && nq < (int)qs.size()) { qs.resize(nq); qt.resize(nq); qa.resize(nq); qb.resize(nq); }
    if (mode == "query") {
      P("query mode needs the index; use build first");
      return 0;
    }
    std::vector<u32> cur(t.n, NONE), nxt(t.n, NONE), fr, nf;
    double t0 = now_s();
    size_t pos = 0;
    for (size_t i = 0; i < qs.size(); i++) {
      if (onlineQuery(c, qs[i], qt[i], qa[i], qb[i], k, cur, nxt, fr, nf)) pos++;
      std::fill(cur.begin(), cur.end(), NONE);
    }
    double el = now_s() - t0;
    P("[onlineQuery] N=%u M=%llu k=%u q=%zu  total=%.4f s  per-query=%.1f us  positive=%zu",
      t.n, (unsigned long long)t.m(), k, qs.size(), el, el * 1e6 / qs.size(), pos);
    return 0;
  }
  if (mode == "idump" || mode == "idump2") {
    gFaithful = (mode == "idump2") ? 1 : 0;
    TB g = load(path);
    TG t = graphConvert(g);
    CSR c = build_csr(t);
    double t0 = now_s(); indexBuildPlus(c, t.n); double tb = now_s() - t0;
    u64 tot = 0;
    for (u32 i = 0; i < t.n; i++) tot += Lin[i].size() + Lout[i].size();
    P("[indexBuild+] time=%.4f s entries=%llu", tb, (unsigned long long)tot);
    writeIdx(t.n, argv[3]);
    return 0;
  }
  if (mode == "iquery") {
    u32 n = 0; readIdx(path.c_str(), n);
    std::vector<u32> qs, qt, qa, qb;
    FILE* f = fopen(argv[3], "rb");
    u32 a, b, s2, e2;
    while (fscanf(f, "%u %u %u %u", &a, &b, &s2, &e2) == 4) {
      qs.push_back(a); qt.push_back(b);
      if (s2 <= e2) { qa.push_back(s2); qb.push_back(e2); } else { qa.push_back(e2); qb.push_back(s2); }
    }
    fclose(f);
    u32 k = argc > 4 ? atoi(argv[4]) : 3;
    int nq = argc > 5 ? atoi(argv[5]) : 0;
    if (nq > 0 && nq < (int)qs.size()) { qs.resize(nq); qt.resize(nq); qa.resize(nq); qb.resize(nq); }
    double t0 = now_s();
    size_t pos = 0;
    for (size_t i = 0; i < qs.size(); i++)
      if (indexQuery(qs[i], qt[i], qa[i], qb[i], k)) pos++;
    double el = now_s() - t0;
    P("[indexQuery] N=%u q=%zu k=%u total=%.4f s per-query=%.2f us positive=%zu",
      n, qs.size(), k, el, el * 1e6 / qs.size(), pos);
    return 0;
  }
  P("usage: paper convert|build|idump|iquery|online <file> [args]");
  return 1;
}
