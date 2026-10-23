# graph-mid-term

GPU acceleration study for **k-hop reachability on temporal bipartite graphs**.

Target paper: Zhou et al., *Efficient processing of k-hop reachability queries on
temporal bipartite graphs*, IP&M 62(5):104178, 2025 (Elsevier, paywalled).
Problem-equivalent open reference used instead: **CBSR** (github.com/SunboTax/CBSR),
same definition, same datasets, compares against TBP = Chen et al. VLDB 2021.

## What is in here

| file | role |
|---|---|
| `src/tb.cpp` | CPU baseline: stats / transform (wedge enumeration) / online query / batch |
| `src/batch.cu` | GPU multi-source k-hop relaxation kernel |
| `scripts/download_data.sh` | fetch the 11 CBSR temporal bipartite graphs + 100k queries each |
| `scripts/run_all.sh` | full pipeline on one dataset |

Pipeline:

```text
temporal bigraph (u,v,ts,te)
   -- wedge enumeration (interval-overlap self-join per location) -->
transformed graph: one contact edge (x,y,[a,b]) per co-presence
   -- k rounds of relaxation -->
earliest-arrival state  A[vertex][source]   ->  k-hop reachability
```

## Build

```bash
make ARCH=sm_90            # H20 / A100 / H100
make ARCH=sm_120           # RTX 5050 laptop
```

On h20-230 the system nvcc is 11.5, use the 13.0 toolkit:

```bash
export PATH=/data1/xuanyiyang/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/data1/xuanyiyang/cuda-13.0/lib64:$LD_LIBRARY_PATH
make ARCH=sm_90
```

## Data (7GB, not committed)

```bash
bash scripts/download_data.sh
```

## Run

```bash
bash scripts/run_all.sh twitter 3 1024       # dataset, k, batch size
bash scripts/run_all.sh edit-dewiki 3 4096   # the big one
```

or manually:

```bash
./build/tb stats    data/bigraph/twitter.txt
./build/tb transform data/bigraph/twitter.txt build/tg.bin      # wedge enumeration
./build/tb online   build/tg.bin data/query/twitter.query_.queryres_0 3 2000
./build/tb batch    build/tg.bin 1024 3 0 2147483647 16         # CPU, 16 threads
./build/batch_gpu   build/tg.bin 1024 3 0 3 4000                # GPU; last arg = peak BW GB/s
```

`batch_gpu` args: `<tg.bin> <S> <k> <ta> <reps> [peakBW_GBps]`; S is rounded up
to a power of two. It prints achieved bandwidth and percentage of peak -- use it for roofline.

## Measured on RTX 5050 Laptop (20 SM, 369 GB/s), twitter: |U|=175K, |E|=4.66M, 21.0M contacts

### graph transform, CPU 16 threads

| dataset | |U| | |E| | contacts | ratio | sort | join(count) |
|---|---|---|---|---|---|---|
| wikiquote_pl | 4,312 | 378,979 | 46,893 | 0.12x | 0.03s | 0.01s |
| bibsonomy | 5,794 | 2,555,081 | 289,839 | 0.11x | 0.29s | 0.02s |
| linux-kernel | 42,045 | 1,565,684 | 287,943 | 0.18x | 0.11s | 0.01s |
| stackexchange | 545,195 | 1,301,943 | 1,185,783 | 0.91x | 0.15s | 0.00s |
| lastfm_song | 992 | 19,150,869 | 556,871 | 0.03x | 4.55s | 0.08s |
| amazon | 2,146,057 | 5,838,042 | 1,671,255 | 0.29x | 1.28s | 0.06s |
| epinions-rating | 120,492 | 13,668,321 | 2,960,013 | 0.22x | 2.71s | 0.08s |
| edit-jawiki | 444,563 | 41,998,341 | 5,701,669 | 0.14x | 11.2s | 0.30s |
| **twitter** | 175,214 | 4,664,606 | 21,039,776 | **4.51x** | 0.84s | 0.09s |
| edit-dewiki | 1,025,084 | 129,885,940 | 40,249,292 | 0.31x | 40.0s | 1.08s |

The wedge blow-up is mild (avg 0.3x, twitter 4.5x). The join itself is nearly free;
**the sort dominates** (40s on edit-dewiki) -- that is the part to move to GPU.

### online single query, CPU 1 thread (pruned BFS, real query windows)

| k | per query | 100k queries |
|---|---|---|
| 2 | 537 us | 54 s |
| 3 | 9.78 ms | 978 s |
| 4 | 29.7 ms | 2970 s |

### CPU vs GPU, same dense multi-source relaxation, twitter, k=3

| S | CPU 16 thr | GPU 5050 | speedup | GPU G edge*src/s | achieved BW |
|---|---|---|---|---|---|
| 64 | 0.319 s | 0.0998 s | 3.2x | 40.5 | ~88% of peak |
| 256 | 0.709 s | 0.405 s | 1.75x | 39.9 | ~87% of peak |
| 1024 | - | 1.447 s | - | 44.7 | ~97% of peak |

## Key findings (and negative results)

1. **Query answering does not scale with cores.** 1 thread = 9.78 ms/query;
   16 concurrent processes = 10.6 ms/query (1.08x). The workload is purely
   memory-bandwidth bound, which is both why an index is mandatory and why a
   wider memory pipe (GPU) is the only lever.
2. **The dense batched kernel is already at ~90% of the bandwidth roofline**
   (8 bytes per edge*source: 4B load + 4B atomicMin). No tuning headroom left;
   further gains must come from moving fewer bytes (frontier pruning, bitset state).
3. **The GPU/CPU ratio here is basically the bandwidth ratio.** On this laptop
   (369 vs ~182 GB/s) that is only 1.75-3.2x. The same kernel on H20 (~4 TB/s)
   should be an order of magnitude faster than a 16-core CPU. Report any speedup
   as bandwidth ratio x traffic ratio, never as a bare multiple.
4. **Single indexed query must not go to GPU**: ~10 us on CPU vs 5-10 us kernel launch.

## Measured on H20 (8x H20 96GB, sm_90, 4000 GB/s; 2x AMD EPYC 9K84 = 192 cores / 384 threads)

### CPU baseline, twitter, k=3

| configuration | per query | throughput |
|---|---|---|
| online pruned BFS, 1 thread | 11.0 ms | 91 q/s |
| online pruned BFS, 64 concurrent procs | 0.664 ms | 1507 q/s |
| dense batch, OpenMP 16 thr, S=1024 | 3.56 ms | - |
| dense batch, OpenMP 64 thr, S=1024 | 0.933 ms | 1071 q/s |
| dense batch, OpenMP 128 thr, S=1024 | 0.964 ms | - |

Contrast with the laptop: there 16 concurrent processes gave only 1.08x over one
thread (memory saturated immediately). On H20 64 processes give **16.6x**, because
the EPYC memory subsystem is far wider. CPU scaling is entirely a bandwidth story.

### GPU dense batch, twitter, k=3

| S | time | G edge*src/s | achieved BW | pct of peak | per query | vs CPU 64-proc |
|---|---|---|---|---|---|---|
| 1024 | 0.3403 s | 189.9 | 1519 GB/s | 38% | 0.332 ms | 2.0x |
| 4096 | 1.2067 s | 214.2 | 1714 GB/s | 43% | 0.295 ms | 2.3x |
| 8192 | 2.1757 s | 237.7 | 1901 GB/s | 48% | 0.266 ms | 2.5x |

(achieved BW is computed as 8 B per edge*source; counting the extra load inside
the guarded store the real figure is ~12 B/op, i.e. ~57-70% of peak.)

### is it the atomics?  no

Rebuilding with -DNOATOM (plain store instead of atomicMin; result incorrect,
bandwidth probe only) gives **exactly the same time**: 0.0944 s vs 0.0944 s at
S=1024 k=1, 0.6120 s vs 0.6119 s at S=8192 k=1. The kernel is not atomic-bound and
not latency-bound -- it is purely DRAM-bound. No kernel-level headroom left.

### GPU 5050 vs H20, same kernel, twitter S=1024 k=3

| GPU | time | achieved BW |
|---|---|---|
| RTX 5050 Laptop (369 GB/s) | 1.447 s | 325 GB/s (88% of peak) |
| H20 (4000 GB/s) | 0.3403 s | 1519 GB/s (38% of peak) |

Speedup 4.25x for a 10.8x peak-bandwidth ratio -- the laptop was already at 88% of
its roofline, the H20 only reaches ~50-70%. H20 leaves more absolute headroom.


### how much does pruning actually save? (instrumented, H20, 1 thread)

| k | edge-visits per query | pct of M (21.0M) | per query |
|---|---|---|---|
| 2 | 250,898 | 1.19% | 0.600 ms |
| 3 | 4,592,497 | 21.83% | 10.76 ms |
| 4 | 11,890,220 | 56.51% | 30.27 ms |

This kills the frontier idea: pruning is only dramatic at k=2 (84x less work).
The transformed graph has average out-degree ~120, so by k=3 a single source
already touches 22% of all contacts and by k=4 57%.

Worse, **batching destroys the pruning advantage**. A single source touches 22% of
M at k=3, but the union frontier of S=1024 sources covers ~70% of the graph already
at hop 1 (1024 x 120 = 123k vertices out of 175k). So union-frontier ~= dense.
Pruning helps only for few sources + small k; feeding a GPU needs many sources.
The two requirements are directly opposed.

## Is the reference work parallel? (checked)

- IP&M 2025 (the target paper): no mention of parallelism / threads / concurrency
  anywhere in abstract, highlights or contributions. Serial algorithm + serial timing.
- CBSR (open reference, full text + code available): the word parallel appears once
  in the whole paper, inside the machine spec (Xeon Platinum 8358, 32 cores / 64
  threads) -- they report the core count but never use it. Its CMake sets
  -fopenmp, but the only `#pragma omp parallel for` are 4 sites in
  `src/partition.cpp` (graph partitioning); index construction and query
  are serial.

So the field convention is **serial CPU, single-threaded timing**. That matters for
how the speedup is reported:

| comparison | GPU per query | CPU per query | speedup |
|---|---|---|---|
| vs. the papers convention (1 thread) | 0.266 ms | 11.0 ms | **41x** |
| vs. our own 64-process CPU baseline | 0.266 ms | 0.664 ms | **2.5x** |

Both numbers are real; the paper must state which baseline is being compared.

## Verdict

The dense batched relaxation buys only **2-2.5x on H20** (vs the best CPU baseline)
and **1.75-3.2x on a laptop**. It is already at the DRAM roofline, so no amount of
kernel tuning helps. The remaining lever is algorithmic: the CPU pruned BFS touches
only ~1.6% of the edges per query while the dense kernel touches 100%. A
frontier-based GPU multi-source BFS is the change that converts 2.5x into 10x+.

## Next step

The dense kernel does M * S work per hop with no pruning, while the CPU pruned BFS
touches ~1.6% of edges per query. A **frontier-based GPU multi-source BFS** (scan only
the union frontier, amortise edge reads across sources) is the change that can turn
1.75x into 10-30x. That is `src/batch.cu` v2 and is not written yet.

