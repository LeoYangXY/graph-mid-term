#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>

using i64 = long long;
using u32 = uint32_t;
using u64 = uint64_t;

static inline double now_s() {
  return std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

struct TBigraph {
  u32 nu = 0, nl = 0;
  std::vector<u32> eu, ev;
  std::vector<u32> ts, te;
  size_t m() const { return eu.size(); }
};

static TBigraph load(const std::string& path) {
  TBigraph g;
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s", path.c_str()); fputc(10, stderr); exit(1); }
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
  for (u32 x : g.eu) mxu = std::max(mxu, x);
  for (u32 x : g.ev) mxl = std::max(mxl, x);
  if (mxu >= g.nu) g.nu = mxu + 1;
  if (mxl >= g.nl) g.nl = mxl + 1;
  return g;
}

struct TGraph {
  u32 n = 0;
  std::vector<u32> src, dst;
  std::vector<u32> ca, cb;
  size_t m() const { return src.size(); }
};
