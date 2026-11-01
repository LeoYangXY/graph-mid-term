// Concatenate per-vertex label lists from several partial indexes, streaming so
// the merge needs no more memory than one vertex's labels.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
using u32 = unsigned int;
struct Entry { u32 w, t1, t2, k; };
static void fail(const char* s) { fprintf(stderr, "%s\n", s); exit(1); }
int main(int argc, char** argv) {
  if (argc < 4) fail("usage: mergeidx out.idx part1.idx part2.idx ...");
  int np = argc - 2;
  std::vector<FILE*> in(np);
  u32 N = 0, K = 0;
  for (int i = 0; i < np; ++i) {
    in[i] = fopen(argv[i + 2], "rb");
    if (!in[i]) fail("cannot open part");
    u32 head[4];
    if (fread(head, 4, 4, in[i]) != 4 || head[0] != 0xffffffffu || head[1] != 1)
      fail("part is not a versioned index");
    if (i == 0) { N = head[2]; K = head[3]; }
    else if (head[2] != N) fail("parts disagree on |V|");
    else if (head[3] > K) K = head[3];
  }
  std::string tmp = std::string(argv[1]) + ".partial";
  FILE* out = fopen(tmp.c_str(), "wb");
  if (!out) fail("cannot open output");
  u32 header[4] = {0xffffffffu, 1, N, K};
  if (fwrite(header, 4, 4, out) != 4) fail("write failed");
  std::vector<Entry> buf;
  for (u32 v = 0; v < N; ++v) {
    for (int pass = 0; pass < 2; ++pass) {
      u32 total = 0;
      std::vector<std::vector<Entry>> lists(np);
      for (int i = 0; i < np; ++i) {
        u32 c;
        if (fread(&c, 4, 1, in[i]) != 1) fail("truncated part");
        lists[i].resize(c);
        if (c && fread(lists[i].data(), sizeof(Entry), c, in[i]) != c) fail("truncated part");
        total += c;
      }
      if (fwrite(&total, 4, 1, out) != 1) fail("write failed");
      for (int i = 0; i < np; ++i)
        if (!lists[i].empty() &&
            fwrite(lists[i].data(), sizeof(Entry), lists[i].size(), out) != lists[i].size())
          fail("write failed");
    }
  }
  for (int i = 0; i < np; ++i) fclose(in[i]);
  if (fclose(out) != 0 || rename(tmp.c_str(), argv[1]) != 0) fail("publish failed");
  printf("[mergeidx] wrote %s from %d parts, |V|=%u K=%u\n", argv[1], np, N, K);
  return 0;
}
