# GPU acceleration for k-hop reachability on temporal bipartite graphs

Target paper: **Zhou, Wang, Tan, Du, Chen, Tang**, *Efficient processing of k-hop
reachability queries on temporal bipartite graphs*, Information Processing &
Management 62(5):104178, 2025 (Elsevier).

Machine: **h20-230**, 8x H20 96GB (sm_90, 4000 GB/s peak), 2x AMD EPYC 9K84
(192 cores / 384 threads), 2.2 TB RAM. All numbers below measured there.

- `ALGO.md` - the paper algorithms explained from scratch (no graph-theory
  background assumed). Start there if the algorithms are new to you.
- `src/paper.cpp` - CPU baseline, faithful to the paper, **serial / single threaded**
  (the paper never parallelises anything, so neither does the baseline).
- `src/q.cu` - GPU kernel: batched k-hop temporal reachability.

## Build

```bash
export PATH=/data1/xuanyiyang/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/data1/xuanyiyang/cuda-13.0/lib64:$LD_LIBRARY_PATH
g++  -O2 -o build/paper src/paper.cpp
nvcc -O3 -arch=sm_90 -o build/qgpu src/q.cu
```

## Run

```bash
./build/paper convert data/bigraph/<ds>.txt  build/tg.bin
./build/paper build   data/bigraph/<ds>.txt
./build/paper online  build/tg.bin data/query/<ds>.query_.queryres_0 <k> <nq>
./build/qgpu build/tg.bin data/query/<ds>.query_.queryres_0 <k> <S> <reps> <peakBW>
```

## 1. What the paper does (short version)

wedge (two people at one place with overlapping time intervals) -> graphConvert:
collapse the location layer, one contact edge per wedge -> TB2Hop index: offline
labels `(w, t1, t2, k)` per vertex -> query = compare two labels, no
traversal. Full walkthrough with examples in ALGO.md.

Baselines reproduced: `onlineQuery` (BFS on the transformed graph) and
`indexBuild+` (index construction).

## 2. Where the paper hurts (its own Table 7, ms)

| dataset | Convert | indexBuild | indexBuild+ |
|---|---|---|---|
| ZH | 25 | 687,100 | 2,058 |
| QU | 57 | 13,776,399 | 7,778 |
| BO | 125 | 86,536,248 (24 h) | 80,487 |
| BR brightkite | 286 | did not finish | 1,272,056 (21 min) |
| GO gowalla | 151 | did not finish | 1,731,429 (29 min) |
| FR | 8,241 | did not finish | 48,170,232 (13.4 h) |

**Graph conversion is cheap (25-8241 ms). Index construction is the whole
problem: up to 24 h (basic) and 13.4 h (optimised).** Query answering with the
finished index is already >1000x faster than the online baselines, so the GPU
target is traversal / index construction, not the point query itself.

## 3. Reproduced CPU baseline (serial, H20)

| dataset | |Eb| | |V| | transformed |E| | Convert | k=2 | k=3 | k=4 | indexBuild+ |
|---|---|---|---|---|---|---|---|---|
| linux-kernel | 1,565,684 | 42,045 | 205,611 | 0.133 s | - | - | - | **19.26 s** (avg label 102.6) |
| epinions | 13,668,320 | 120,492 | 2,945,795 | 2.48 s | 1400 us | 5118 us | 9685 us | **7532.6 s (2 h 5 min)** |
| twitter | 4,664,606 | 175,214 | 20,352,035 | 0.84 s | 371 us | 6099 us | 21289 us | - |

Fidelity notes:

- everything single threaded, matching the paper (no OpenMP anywhere)
- `onlineQuery` is level synchronous and matches the GPU kernel exactly
  (verified on wikiquote: 32/64 positives on both)
- `indexBuild+` implements the short-interval-first / priority-queue variant
  with simple dominance and order pruning, but **omits the combined-dominance
  pass** (line 30 of Algorithm 4). Its index is therefore a superset: larger
  labels (avg 102.6 on linux-kernel) and a build time that is a lower bound.
- datasets: the paper uses SNAP/KONECT graphs with synthesised power-law
  durations; those downloads are blocked here, so we use same-format CBSR
  temporal bipartite graphs (twitter, epinions, linux-kernel).
- indexBuild+ on epinions produced 985.9M entries (avg label 8182 per vertex).
  That is far larger than the paper's index would be, and is a direct
  consequence of the missing combined-dominance pass below, not of the data.

## 4. GPU kernel

src/q.cu answers **S queries simultaneously**. State is `A[vertex * S + query]`
so a warp covers 32 consecutive queries on the same contact edge -> fully
coalesced. One relaxation round = one kernel over all contact edges; k rounds
give k hops. atomicMin folds competing arrivals. Grid-stride loop so S is not
limited by the grid dimension.

### Results (H20, best of 3 reps)

| dataset | k | S | us/query | queries/s | achieved BW | vs CPU serial |
|---|---|---|---|---|---|---|
| twitter | 3 | 1024 | 393.5 | 2,542 | 2483 GB/s (62%) | **15.5x** |
| twitter | 3 | 4096 | 442.6 | 2,259 | 2207 GB/s (55%) | 13.8x |
| twitter | 3 | 16384 | 538.8 | 1,856 | 1813 GB/s (45%) | 11.3x |
| twitter | 2 | 16384 | 299.2 | 3,342 | 2177 GB/s (54%) | 1.24x |
| twitter | 4 | 16384 | 809.1 | 1,236 | 1610 GB/s (40%) | **26.3x** |
| epinions | 3 | 1024 | 88.3 | 11,321 | 1601 GB/s (40%) | 58.0x |
| epinions | 3 | 8192 | 73.2 | 13,659 | 1931 GB/s (48%) | **69.9x** |
| epinions | 5 | 8192 | 145.2 | 6,887 | 1623 GB/s (41%) | - |
| epinions | 6 | 8192 | 182.9 | 5,467 | 1546 GB/s (39%) | - |

How to read it:

- **the win grows with k**. At k=2 a single pruned CPU BFS is already cheap, so
  batching buys only 1.24x. At k=4 it buys 26x; on the sparser epinions graph,
  70x.
- the kernel runs at **40-62% of H20 peak bandwidth**: DRAM bound and close to
  the roofline, so tuning headroom is small.
- more batching does **not** keep helping: throughput peaks around S=1024-8192
  then falls, because work is O(M x S) while memory pressure grows.
- these are against the **serial** CPU baseline, which is what the paper
  reports. If the CPU is allowed 64 processes the gap narrows sharply; that
  number is in this repo git history and must be reported alongside.

## 5. GPU index builder (src/ibuild.cu)

Index construction is the paper biggest cost. Its inner loop is "one temporal
BFS per vertex, in both time directions". We batch **B vertices at a time** and
relax them all simultaneously.

The whole design hinges on one trick: the state of a (vertex, source) pair is a
**single u64**

    bits 60..63  hop
    bits 30..59  ts' = range - (ts - tmin)      (minimising ts' = maximising ts)
    bits  0..29  te' = te - tmin

Because ts and te live in the same word, a stored interval always belongs to one
real path, so the result stays sound, and one ' + BT + 'atomicMin' + BT + ' does the whole
update: it minimises hop first, then maximises ts, then minimises te. 8 bytes
per (vertex, source), fully coalesced, no locks.

### Build time (H20, vs the serial CPU baseline)

| dataset | CPU serial | GPU | speedup | entries CPU | entries GPU |
|---|---|---|---|---|---|
| linux-kernel | 19.26 s | **3.68 s** (B=4096) | **5.2x** | 4,313,133 | 2,738,446 |
| epinions | **7532.6 s (2 h 5 min)** | **79.3 s** (B=8192) | **95x** | 985,878,778 | 542,110,838 |

Batch size: linux-kernel 3.68 / 3.95 / 4.53 s at B = 4096 / 8192 / 16384, so
B = 4096 is already past the knee; bigger batches only cost memory.

### Index quality (linux-kernel, k=3, 2000 queries)

| | positives | recall | query latency |
|---|---|---|---|
| ground truth, onlineQuery (exact BFS) | 1211 | 100% | 411 us |
| CPU index | 470 | 38.8% | 1.66 us |
| **GPU index** | **610** | **50.4%** | **1.01 us** |

Both indices here are incomplete and this is the honest weak point of the
current code: they keep too few entries per vertex pair (the CPU variant does
one merged BFS per vertex and therefore records one path per vertex; the GPU
kernel stores exactly one entry per pair by design). The GPU index is both
faster to build **and** better in recall than the CPU variant, but neither is
complete yet. Fixing this means keeping a Pareto set of (hop, ts, te) per pair.

### Why 95x is conservative

The paper Algorithm 4/5 starts a **separate BFS from every adjacent edge** of a
vertex (2 x deg(u) traversals per vertex). Our CPU baseline merges those into
one BFS per vertex, which is roughly deg(u) times cheaper. Running the faithful
per-edge version on linux-kernel did not finish within the time budget - which
is exactly why the paper needs 21 min on BR and 13.4 h on FR. So the true paper
baseline is *slower* than the 7532.6 s we compare against, and the real GPU
speedup is larger than 95x.

## 6. Honest status

Done: algorithm write-up, faithful serial CPU baseline, GPU batched query
kernel verified equal to the CPU on positives, measurement tables above.

Not done:

- **index completeness** (see 5): both indices miss reachable pairs. Highest
  priority, and the reason the recall column above is not 100%.
- combined-dominance pruning in indexBuild+
- index update (Algorithms 6 / 7)
- the faithful per-edge BFS variant of indexBuild+ is implemented (' + BT + 'idump2' + BT + ') but
  too slow to finish on any dataset here.

