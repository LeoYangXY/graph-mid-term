#!/usr/bin/env bash
# Build a bounded-hop TB2Hop-style index for a stay table on GPUs.
# Input line format: "<object> <location> <start> <end>" (same as the stay data).
# Usage: run_stay_gpu.sh stays.txt K [shards] [out.idx]
set -euo pipefail
SRC=${1:?usage: run_stay_gpu.sh stays.txt K [shards] [out.idx]}
K=${2:-5}
SH=${3:-8}
OUT=${4:-${SRC%.txt}.k${K}.idx}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=$ROOT/build
mkdir -p $B
[ -x $B/paper ] || g++ -O2 -o $B/paper $ROOT/src/paper.cpp
[ -x $B/iex ]   || nvcc -O3 -arch=sm_90 -DKHOP=32 -DSPH=128 -o $B/iex $ROOT/src/iex.cu
[ -x $B/mergeidx ] || g++ -O2 -o $B/mergeidx $ROOT/src/mergeidx.cpp

echo "== convert $(basename $SRC)"
$B/paper convert "$SRC" $B/tg_$(basename $SRC .txt).bin

T=$B/tg_$(basename $SRC .txt).bin
echo "== build K=$K on $SH shard(s)"
start=$(date +%s)
pids=()
for i in $(seq 0 $((SH-1))); do
  CUDA_VISIBLE_DEVICES=$i $B/iex "$T" 16 $K "$OUT.part$i" $i $SH >"$OUT.part$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p || { echo "a shard failed; see $OUT.part*.log"; exit 1; }; done
echo "== merge"
$B/mergeidx "$OUT" $(for i in $(seq 0 $((SH-1))); do echo "$OUT.part$i"; done)
rm -f $OUT.part*
echo "elapsed $(( $(date +%s) - start )) s -> $OUT"
