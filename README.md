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

## Next step

The dense kernel does M * S work per hop with no pruning, while the CPU pruned BFS
touches ~1.6% of edges per query. A **frontier-based GPU multi-source BFS** (scan only
the union frontier, amortise edge reads across sources) is the change that can turn
1.75x into 10-30x. That is `src/batch.cu` v2 and is not written yet.

