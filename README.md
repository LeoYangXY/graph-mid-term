# GPU index construction for k-hop reachability on temporal bipartite graphs

Paper: **Junfeng Zhou, Zuyong Wang, Yuting Tan, Ming Du, Ziyang Chen, Xian Tang**,
*Efficient processing of k-hop reachability queries on temporal bipartite graphs*,
Information Processing and Management 62 (2025) 104178.

Machine: h20-230, 1x H20 96GB (sm_90). CPU numbers are single threaded, matching
the paper.

## Files

- `src/paper.cpp` - CPU side: graphConvert (Alg.1), onlineQuery (Alg.2), the
  two index builders, indexQuery (Alg.3), plus `cmp` which diffs an index against
  exact BFS.
- `src/iex.cu` - GPU index builder.
- `src/q.cu` - GPU batched **online** query (not an index query).
- `scripts/prep.py` - paper preprocessing: power-law (alpha=-2.5) durations in
  1-48 h, plus query generation.
- `scripts/test_correctness.py` - independent exhaustive oracle (see below).

## Build

```bash
export TMPDIR=/data1/xuanyiyang/tmp      # / is full on h20-230
export PATH=/data1/xuanyiyang/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/data1/xuanyiyang/cuda-13.0/lib64
make            # builds paper, qgpu and iex-k5-p128 (KHOP=5, SPH=128)
make test       # exhaustive correctness suite
```

## The index

TB2Hop: every vertex has `L_in` and `L_out`, each entry a quadruple
`(w, t1, t2, k)` meaning "w reaches v inside [t1,t2] in k hops". A query is
answered by C1/C2/C3 only, with no graph traversal.

**The paper builds ONE index with no hop limit** - Algorithms 4 and 5 take only the
graph as input, so k is a query parameter and every path length is stored.

Our GPU builder needs a hop bound `KHOP` (compile time) and refuses a runtime k larger
than it. `KHOP=5` was chosen because the paper's query experiments use k in [1,5].
The bound is recorded in the index file header and a query with `k > KHOP` exits with
an error instead of returning a wrong answer.

## GPU builder design

- Batches of B sources are relaxed together; state is one u64 (start offset, end)
  per (vertex, hop, slot, query), so consecutive queries of a warp are contiguous.
- One Pareto frontier **per hop** (`SPH` slots). Fixing the hop reduces domination to
  two dimensions, which keeps each frontier small.
- Only nondominated intervals survive, so no answer is lost.
- **SPH is a capacity, not an accuracy knob.** A batch that overflows doubles its
  slots and recomputes; if the frontier does not fit in memory the batch is halved;
  if it still does not fit the program fails and publishes nothing.
- Output is compacted on the device (count, allocate exactly, emit) - the earlier
  fixed output pool silently truncated.
- The index is written to `.partial` and renamed only after every write succeeds.

## Correctness

Verified by `scripts/test_correctness.py`, an independent oracle that enumerates
real walks (no share code with either implementation). It compares **every vertex
pair x every time window x every k** on 16 tiny graphs, at batch 1/3/8:

```
PASS: 338010 exhaustive query comparisons + 300-interval overflow + invalid input/horizon tests
```

It also checks that labels are real paths, that a 300-interval frontier triggers the
retry path, and that invalid input / truncated files / k > KHOP fail without touching
an existing index. `compute-sanitizer memcheck/initcheck`: 0 errors.

End-to-end on real data, `paper cmp` compares the GPU index against exact BFS per
query. **0 false positives, 0 false negatives** on elwiki (K=1..5, 100k queries each)
and linux-kernel (K=3,5, 100k queries).

Limitation of this evidence: it is verification on these graphs and these queries,
not a proof for arbitrary inputs.

## Results: elwiki (N=11906, M=765013, B=8)

| KHOP | build | index size | entries | query (100k) | cmp |
|---|---|---|---|---|---|
| 1 | 0.37 s | 1.5 MB | 91,117 | 0.03 us/q | 0/0 |
| 2 | 2.29 s | 8.2 MB | 527,122 | 0.08 us/q | 0/0 |
| 3 | 4.32 s | 21 MB | 1,353,276 | 0.22 us/q | 0/0 |
| 4 | 6.41 s | 36 MB | 2,338,607 | 0.51 us/q | 0/0 |
| 5 | 8.18 s | 52 MB | 3,356,768 | 0.92 us/q | 0/0 |

One K=5 index serves k=3 (0.60 us), k=4 (0.76 us) and k=5 (0.92 us); the horizon
check makes that explicit rather than silently truncating.

## Results: linux-kernel (N=42045, M=205611, B=16)

| KHOP | build | index size | entries | indexQuery | onlineQuery | cmp |
|---|---|---|---|---|---|---|
| 3 | 12.15 s | 139 MB | 9,045,628 | 19.98 us/q | 437.1 us/q | 0/0 |
| 5 | 26.15 s | 440 MB | 28,768,418 | 174.81 us/q | - | 0/0 |

indexQuery is 22x faster than exact BFS at K=3, far short of the paper's "three
orders of magnitude". That is our `indexQuery` implementation, not the paper's: it
scans `L_out(s) x L_in(t)` as a nested loop instead of grouping entries by shared
vertex as Algorithm 3 does. At K=5 the index is 3x larger and the naive scan costs
9x more, which is exactly the quadratic term. Optimising it is the obvious next step.

## Results: GPU online query (`src/q.cu`)

This is a batched re-implementation of Algorithm 2, not an index query. Earlier
measurements (before the correctness hardening) were twitter k=3 15.5x and k=4
26.3x, epinions k=3 69.9x against a single-threaded CPU. **Those numbers are stale**:
the hardened kernel uses a wider arrival type, so it must be re-benchmarked before
being quoted.

## What was withdrawn

An earlier README claimed 5787x on linux-kernel. That compared the GPU against
`paper.cpp`'s `idump2`, which turns out not to be the paper's Algorithm 5:

- it never runs the combined-dominance test (Alg.4 line 30 / Alg.5 line 12), so its
  index is a superset and its build time is not the paper's;
- its visited stamp is reused across different hubs, which can make a later hub skip
  an edge an earlier hub had marked. An 8-vertex graph reproduces this: the CPU index
  misses one reachable query where the GPU index answers it.

So the CPU baseline is an approximation of the paper, and no speedup should be
quoted against it. A faithful implementation is required before any such number is
published.

## Known limits

- `k > KHOP` is refused, not approximated. The paper has no such limit.
- Index size grows quickly with KHOP (elwiki: 1.5 MB at K=1 to 52 MB at K=5), so does
  build time; a paper-equivalent unbounded build is not implemented.
- indexQuery is the naive nested loop described above.
- Index update (Algorithms 6/7) is not implemented.
- Datasets are KONECT/SNAP with paper-style durations. Our edge counts differ from
  the paper's Table 6 (e.g. zhwiki 1,576,187 vs their 947,552), so absolute times are
  not comparable with their Table 7 - only the trend is.
- Build state is N x KHOP x SPH x B x 8 bytes; on a shared GPU the batch shrinks
  automatically, which costs speed.
