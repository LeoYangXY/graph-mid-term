# GPU acceleration for k-hop reachability on temporal bipartite graphs

Paper: **Zhou, Wang, Tan, Du, Chen, Tang**, *Efficient processing of k-hop
reachability queries on temporal bipartite graphs*, IP&M 62(5):104178, 2025.

Machine: **h20-230**, 8x H20 96GB (sm_90, 4000 GB/s peak), 2x AMD EPYC 9K84.
Everything below is measured there, single threaded on the CPU side (the paper
never parallelises, so neither does the baseline).

Files:

- `ALGO.md` - the paper algorithms explained from scratch
- `src/paper.cpp` - CPU baseline (graphConvert / onlineQuery / indexBuild+)
- `src/q.cu` - GPU batched k-hop query kernel
- `src/ifront.cu` - GPU index builder

## Build and run

```bash
export TMPDIR=/data1/xuanyiyang/tmp      # /  is full on h20-230
export PATH=/data1/xuanyiyang/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/data1/xuanyiyang/cuda-13.0/lib64
g++  -O2 -o build/paper src/paper.cpp
nvcc -O3 -arch=sm_90 -o build/qgpu  src/q.cu
nvcc -O3 -arch=sm_90 -DPP=12 -o build/ifrt12 src/ifront.cu

./build/paper convert data/bigraph/<ds>.txt build/tg.bin
./build/paper build   data/bigraph/<ds>.txt
./build/paper online  build/tg.bin data/query/<ds>.query_.queryres_0 <k> <nq>
./build/qgpu  build/tg.bin data/query/<ds>.query_.queryres_0 <k> <S> <reps> <peakBW>
./build/ifrt12 build/tg.bin <B> <kmax> <out.idx>
./build/paper iquery <idx> <tg.bin> <queryfile> <k> <nq>   # index-only answers
./build/paper cmp    <idx> <tg.bin> <queryfile> <k> <nq>   # vs exact BFS
```

## 1. The paper in one paragraph

A wedge is "two people at one place with overlapping time intervals".
graphConvert collapses the location layer and emits one contact edge per wedge,
which makes the false-negative problem disappear for a plain BFS. TB2Hop then
precomputes labels `(w, t1, t2, k)` per vertex so a query is just a label
comparison. Details and examples in ALGO.md.

The paper own Table 7: graph conversion costs 25-8241 ms, but **index
construction costs up to 24 h (basic) and 13.4 h (optimised)**. Query answering
with a finished index is already >1000x faster than BFS. So the GPU target is
index construction, not the point query.

## 2. CPU baseline (serial)

| dataset | edges in | vertices | transformed | convert | online k=2 | k=3 | k=4 | indexBuild+ |
|---|---|---|---|---|---|---|---|---|
| linux-kernel | 1,565,684 | 42,045 | 205,611 | 0.133 s | - | - | - | 19.26 s |
| epinions | 13,668,320 | 120,492 | 2,945,795 | 2.48 s | 1400 us | 5118 us | 9685 us | 7532.6 s |
| twitter | 4,664,606 | 175,214 | 20,352,035 | 0.84 s | 371 us | 6099 us | 21289 us | - |

Fidelity notes:

- single threaded throughout, as in the paper
- `onlineQuery` is level synchronous and matches the GPU query kernel exactly
  (32/64 positives on both, wikiquote)
- `indexBuild+` here merges the per-adjacent-edge BFS of Algorithm 4/5 into one
  BFS per vertex. That merge is much cheaper but loses answers; see section 5.
- the paper SNAP/KONECT datasets are unreachable from this network, so we use
  same-format CBSR temporal bipartite graphs instead

## 3. GPU batched query (src/q.cu)

S queries at once, state `A[vertex * S + query]`, one relaxation round per hop,
atomicMin for competing arrivals, grid-stride loop.

| dataset | k | S | us/query | queries/s | achieved BW | vs serial CPU |
|---|---|---|---|---|---|---|
| twitter | 2 | 16384 | 299.2 | 3,342 | 54% of peak | 1.24x |
| twitter | 3 | 1024 | 393.5 | 2,542 | 62% of peak | **15.5x** |
| twitter | 4 | 16384 | 809.1 | 1,236 | 40% of peak | **26.3x** |
| epinions | 3 | 8192 | 73.2 | 13,659 | 48% of peak | **69.9x** |

The kernel is at 40-62% of H20 peak bandwidth, i.e. DRAM bound and close to the
roofline. The win grows with k: at k=2 a single pruned CPU BFS is already cheap,
so batching only buys 1.24x.

## 4. GPU index builder (src/ifront.cu)

One temporal BFS per vertex is batched: B vertices relaxed simultaneously. State
per (vertex, source) is one u64:

```text
bits 60..63  hop        bits 30..59  tsp = range-(ts-tmin)        bits 0..29  te-tmin
(smaller tsp  ==  larger ts)
```

ts and te share one word, so a stored interval always belongs to one real path.
PP slots per pair form a Pareto frontier, domination being

```text
A dominates B  iff  A.hop <= B.hop  and  A.ts >= B.ts  and  A.te <= B.te
```

A dominated entry can never answer a query the dominating one cannot, so the
frontier is sufficient. Slots use atomicCAS, no locks. All PP slots are expanded,
because different frontier points at a vertex satisfy different outgoing edges.

Layout is `S[(v*PP + p)*B + q]` so that consecutive queries inside a warp are
contiguous (the earlier `[(v*B+q)*PP + p]` layout was not coalesced).

### Results (verified sound: index-only = 0, i.e. no false positives)

| dataset | PP | GPU build | vs serial CPU | entries | correct / truth | recall | false pos |
|---|---|---|---|---|---|---|---|
| wikiquote_pl | 6 | 0.49 s | 0.008x | 15,952 | 936 / 1084 | 86.3% | 0 |
| wikiquote_pl | 12 | 0.99 s | 0.004x | 18,594 | 998 / 1084 | 92.1% | 0 |
| linux-kernel | 6 | 19.33 s | 1.0x | 9.54M | 1084 / 1211 | 89.5% | 0 |
| linux-kernel | 12 | 47.76 s | 0.40x | 13.28M | 1138 / 1211 | 94.0% | 0 |
| **epinions** | **6** | **233.2 s** | **32.3x** | 2.28B | 1042 / 1109 | **94.0%** | **0** |
| **epinions** | **12** | **617.3 s** | **12.2x** | 3.45B | 1063 / 1109 | **95.9%** | **0** |

**Warning: the "vs serial CPU" column above is NOT apples-to-apples.** It compares
against the *merged* CPU baseline, which only reaches ~39% recall. A 0.40x there
means "the GPU is 2.5x slower than something that throws away 60% of the answers",
not "the GPU is slow". The fair comparison is against the *faithful* baseline,
which is the only complete one:

| dataset | GPU PP=6, 89.5% | GPU PP=12, 94.0% | CPU faithful, 100% | speedup PP=6 | PP=12 |
|---|---|---|---|---|---|
| linux-kernel | **19.33 s** | **47.76 s** | **49,981.5 s (13.9 h)** | **2586x** | **1046x** |
| wikiquote_pl | 0.49 s | 0.99 s | 1.16 s | 2.4x | 1.17x |

The faithful run was verified complete: cmp reports both=1211,
index-only=0, online-only=0, i.e. it answers exactly what exact BFS answers.
So on linux-kernel the GPU builds a 94%-complete index in 47.76 s versus 13.9 h
for a 100%-complete one, and both are sound.

Caveat: 13.9 h for 205,611 transformed edges is far worse than the paper Table 7
(BR: 546,939 edges in 21 min), so our faithful reproduction is considerably
slower than theirs. Against their implementation the speedup would be smaller
than 1046x, but the direction and the order of magnitude hold.

So the real picture is:

- **on graphs big enough to fill the machine the GPU wins big** (epinions 32.3x vs
  the merged baseline; linux-kernel 1046x at PP=12 vs the faithful one)
- **on tiny graphs the GPU genuinely loses** (wikiquote_pl 4,154 edges: 0.99 s vs
  1.16 s). Dense batching of M x S work cannot pay off when M is that small. The
  crossover point has not been located yet.
- **every configuration is sound**: false positives = 0, verified query by query
  against exact BFS. All misses are pure false negatives.

"missed" is counted per query against exact BFS (`paper cmp`): a pure false
negative, never a false positive.

## 5. What the two CPU variants are worth

| dataset | merged build | faithful build (paper literal) | penalty | recall |
|---|---|---|---|---|
| wikiquote_pl | 0.0038 s | 1.1603 s | 305x | 38.5% vs **100%** |
| linux-kernel | 19.26 s | did not finish in 25 min | >60x | 38.8% vs - |

The faithful algorithm (a separate BFS from every adjacent edge) is complete: on
wikiquote it answers exactly the 1084 queries that exact BFS answers, no more and
no fewer. The merged variant answers only 417. So:

- comparing the GPU against the **merged** baseline is unfair in the GPU favour
- against the **faithful** baseline the GPU is 1046x faster on linux-kernel
  (47.76 s vs 13.9 h) and about equal on wikiquote_pl, where the graph is too
  small for the GPU

### Why bounded slots cannot reach 100%

Pareto-filtering the faithful index changes nothing: 22,683 entries before,
22,683 after, up to 240 per pair. With three objectives (hop, ts, te) almost
nothing dominates anything, so the frontier is the whole set. A bounded number of
slots therefore always loses a few percent. This is a property of the problem.

## 6. Known limits

- **GPU loses on tiny graphs**: on wikiquote_pl (4,154 edges) the GPU builder
  takes 0.49 s versus 0.0038 s for the merged CPU. Dense batching only pays off
  on graphs big enough to fill the machine.
- index update (Algorithms 6 / 7) is not implemented.
- the combined-dominance pass (Algorithm 4 line 30) is not implemented, so the
  CPU baseline index is a superset.
- speedups are against the **serial** CPU. Allowing the CPU 64 processes narrows
  the gap a lot; that number is in the git history and must be reported too.
- PP > 12 was tried (20, 32, 48) and keeps improving recall, at roughly linear
  extra time and memory.

