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

## What hop length is actually guaranteed? (measured)

A TB2Hop index built with hop bound K answers a query of length L correctly iff some
hub x on the path satisfies dist(s,x) <= K and dist(x,t) <= K. Every path has a hub
with minimum rank, and both halves stay in the rank-restricted subgraph, so

> **L <= K is always covered. L up to 2K is covered only when the minimum-rank
> vertex happens to sit near the middle of the path. That is luck, not a guarantee.**

Measured on elwiki (20k queries, guard disabled to expose the cliff):

| index KHOP | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|
| 2 | 0 | 0 | 1 | 11 | 16 | 17 |
| 3 | 0 | 0 | 0 | 0 | 0 | 1 |
| 4 | 0 | 0 | 0 | 0 | 0 | 1 |

(numbers are false negatives; false positives stay 0 everywhere)

Two things follow. First, k <= KHOP is exact by construction, which is why the
reader refuses k > KHOP instead of answering. Second, practical coverage can reach
beyond KHOP on a friendly graph (KHOP=3 covered k=5 here), but that depends on the
data and must not be sold as a guarantee.

The paper never faces this: its build has no hop bound at all, so it covers every
path length and never truncates.

## Cost model: where the time actually goes

State is one array of N x K x P x B u64, cleared and compacted once per batch and
direction. Total traffic is therefore

    memset + scan  ~  2 x (N/B) x N*K*P*B*8 bytes  =  16 * N^2 * K * P  bytes

Independent of B, quadratic in the vertex count. Predicted vs measured:

| dataset | N | K | P | predicted traffic | time | implied bandwidth |
|---|---|---|---|---|---|---|
| elwiki | 11,906 | 5 | 1024 | 11.6 TB | 8.2 s | 1.4 TB/s |
| FR | 72,517 | 3 | 1024 | 258 TB | 119.7 s | 2.2 TB/s |

So the builder is not compute bound: on FR at B=16 the GPU was busy in only 12% of
samples (95% when busy), peak DRAM 82%. It is bound by touching an array that is
almost entirely empty.

Larger batches do **not** help - they make it worse, because the per-batch array
grows with B while the number of batches shrinks in step, so the traffic is fixed
while frontier pressure rises:

| FR | B=16 | B=128 |
|---|---|---|
| K=3 | 119.7 s | 246.8 s |
| K=5 | 250.9 s | 513.1 s |

The fix is to stop clearing and scanning the whole N x K x P x B array and instead
only touch the (vertex, hop, source) cells that a batch actually wrote.

B=32 is slower than B=8 on elwiki too (21.9 s vs 8.2 s), and B=128 is slower
than B=16 on FR. So the guidance is: small batches, and the real win is sparsity.

Same runs, verified against exact BFS (batch size does not change the result):

| run | build | entries | cmp |
|---|---|---|---|
| lk K=5, B=128 (auto 16) | 35.9 s | 28,545,476 | 0 / 0 |
| elwiki K=5, B=32 | 21.9 s | 3,293,402 | 0 / 0 |



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

## How large does KHOP need to be? (measured)

The paper has no hop bound, so a natural question is what the bound would be.

- Upper bound: a shortest wedge path never repeats a vertex, so k <= |V| - 1.
- Actual: it is set by the temporal structure (how long a chain of time-ordered,
  non-overlapping intervals exists), not by the vertex count alone.

Measured on K=12 builds, entries added per hop:

| k | elwiki added | linux-kernel added |
|---|---|---|
| 5 | 1,019,523 | 10,745,085 |
| 8 | 1,016,544 | 11,971,996 |
| 12 | 1,022,193 | 12,126,328 |

There is **no saturation through k=12** - every extra hop adds about as many entries
as the previous one. So the index grows roughly linearly in K, and so does build
cost:

| dataset | K=5 | K=12 |
|---|---|---|
| elwiki | 8.18 s / 52 MB | 24.01 s / 168 MB |
| linux-kernel | 26.15 s / 440 MB | 50.80 s / 1.8 GB |

An unbounded build therefore has no natural small K; it runs until the temporal
structure can no longer extend a path. That is the mechanism behind the paper's
Table 7 (BR 21 min, FR 13.4 h). Capping K at 5 is a deliberate trade: it covers the
paper's query range k in [1,5] while keeping the index and the build bounded.

## Results: FR (N=72517, M=3381679, B=16, 20k queries)

| KHOP | build | index size | entries | indexQuery | onlineQuery | cmp |
|---|---|---|---|---|---|---|
| 3 | 117.5 s | 681 MB | 44,565,794 | 2.60 us/q | 179.6 us/q | 0/0 |
| 5 | 250.9 s | 2.5 GB | 167,697,429 | 34.46 us/q | 1108.2 us/q | 0/0 |

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
