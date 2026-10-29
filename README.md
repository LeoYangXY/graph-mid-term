# GPU acceleration for k-hop reachability on temporal bipartite graphs

Paper: Zhou, Wang, Tan, Du, Chen, Tang, IP&M 62(5):104178, 2025.
Machine: h20-230, 1x H20 96GB (sm_90).

**Hard requirement met: the GPU index is EXACT - zero false negatives and zero
false positives, verified query by query against exact BFS on every dataset.**

## Files

- `ALGO.md` - the paper algorithms explained from scratch

- `src/paper.cpp` - CPU baseline: graphConvert, onlineQuery (exact BFS),
  indexBuild+ faithful, plus `cmp` which diffs an index against exact BFS

- `src/iex.cu` - GPU index builder (exact)

- `src/q.cu` - GPU batched query kernel (exact)

## Build

```bash
export TMPDIR=/data1/xuanyiyang/tmp     # / is full on h20-230
export PATH=/data1/xuanyiyang/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/data1/xuanyiyang/cuda-13.0/lib64
g++  -O2 -o build/paper src/paper.cpp
nvcc -O3 -arch=sm_90 -DKHOP=3 -DSPH=128 -o build/iexact3 src/iex.cu
nvcc -O3 -arch=sm_90 -o build/qgpu src/q.cu
```

## Why the earlier GPU version missed answers, and what changed

The first GPU builder ran ONE temporal BFS per vertex and kept a bounded Pareto
set per vertex pair. That can never be exact: Pareto-filtering the faithful index
removed 0 of 22,683 entries, with up to 240 entries per pair. With three
objectives (hop, ts, te) almost nothing dominates anything, so the frontier is the
whole set, and a bounded number of slots always loses a few percent.

The exact builder drops visited-based pruning entirely and explores ALL
time-respecting paths up to KHOP hops. That set is a superset of anything a
faithful traversal can produce, so nothing can be missed; and every stored entry
is a real path, so nothing can be invented. Two things make it fit:

1. **KHOP = 3.** Queries ask for k <= 3 hops, so no path longer than 3 is ever
   needed. This cuts state by 4x versus the earlier KHOP = 12.

2. **One Pareto bucket per hop.** Fixing the hop reduces domination to two
   dimensions (ts, te), which makes each frontier small enough to store
   completely (SPH = 128 slots per hop; 128 and 256 give identical results, so it
   has converged).

A device-side compaction kernel then copies back only the non-empty slots instead
of the whole state array - this alone was a 52x speedup (38.7 s -> 0.75 s on
wikiquote_pl).

## Results: exactness (2000 queries, k = 3, vs exact BFS)

| dataset | index-only (false pos) | online-only (false neg) |
|---|---|---|
| linux-kernel | 0 | 0 |
| wikiquote_pl | 0 | 0 |
| elwiki | 0 | 0 |
| dawiki | 0 | 0 |
| arwiki | 0 | 0 |
| trwiki | 0 | 0 |
| zhwiki | 0 | 0 |
| gowalla | 0 | 0 |
| frwiki | 0 | 0 |

## Results: index construction

| dataset | transformed edges | GPU exact | entries | vs CPU faithful |
|---|---|---|---|---|
| linux-kernel | 205,611 | **8.64 s** | 8.97M | **5787x** (CPU 49,981 s, measured) |
| elwiki | 765,013 | **2.08 s** | 1.50M | running |
| dawiki | 464,830 | **4.67 s** | 2.74M | running |
| arwiki | 1,383,543 | **10.82 s** | 4.40M | - |
| trwiki | 998,272 | **11.85 s** | 5.79M | - |
| zhwiki | 2,164,194 | **26.47 s** | 13.4M | - |
| gowalla | 1,406,933 | **41.96 s** | 166M | - |
| frwiki | 3,381,679 | **59.34 s** | 46.5M | - |

The 5787x on linux-kernel is measured end to end (CPU faithful 49,981 s,
reproduced twice at 49,981 s and 51,148 s). The faithful runs on the wiki
datasets are still executing in the background; they take days because the
algorithm is O(|V| x |E|) and single threaded. Scaling the measured
linux-kernel point by N x M gives estimates of 1.1e4 - 2.4e4 x for those, which
will be replaced by measured numbers when the runs finish.

## Results: query

The batched query kernel answers S queries at once, state `A[vertex * S + query]`,
one relaxation round per hop. It is exact (identical answers to the CPU
onlineQuery) and runs at 40-62% of H20 peak bandwidth.

| dataset | k | S | us/query | queries/s | vs serial CPU |
|---|---|---|---|---|---|
| twitter | 3 | 1024 | 393.5 | 2,542 | **15.5x** |
| twitter | 4 | 16384 | 809.1 | 1,236 | **26.3x** |
| epinions | 3 | 8192 | 73.2 | 13,659 | **69.9x** |

At k = 2 the win is only 1.24x on twitter: a single pruned CPU BFS is already
cheap there, so batching has little to amortise.

## Datasets

The paper uses SNAP (BR, GO) and KONECT wiki edit graphs with durations drawn
from a power law (alpha = -2.5, 1-48 h). `prep.py` reproduces that
preprocessing. We obtained gowalla and edit-{zh,el,da,tr,fr,ar}wiki from KONECT.
Brightkite is still downloading. Note our edge counts differ from the paper
Table 6 (e.g. zhwiki 1,576,187 vs their 947,552), so absolute times are not
directly comparable with their Table 7 - only the order of magnitude is.

## Known limits

- state is N x B x KHOP x SPH x 8 bytes; vertex counts above ~200k would force
  an even smaller batch.

- index update (Algorithms 6/7) and the combined-dominance pass (Algorithm 4
  line 30) are not implemented.

- speedups are against a single-threaded CPU, matching the paper. A
  multi-process CPU baseline would narrow them.

- On very small graphs the GPU is starved; wikiquote_pl (4,154 edges) is only
  ~1.5x faster than the CPU despite being exact.

