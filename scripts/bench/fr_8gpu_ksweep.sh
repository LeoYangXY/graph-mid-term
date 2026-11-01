#!/usr/bin/env bash
set -u
export TMPDIR=/data1/xuanyiyang/tmp
export PATH=/data1/xuanyiyang/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/data1/xuanyiyang/cuda-13.0/lib64
cd /data1/xuanyiyang/workspace/graph-mid-term
R=build/k-sweep
nvcc -O3 -arch=sm_90 -DKHOP=32 -DSPH=128 -o $R/iex-lpt build/correctness-review/iex.cu 2>&1 | grep -iE " error" | head -3
df -h /data1 | tail -1
for K in 5 10 20; do
  OUT=$R/fr8-k$K.idx
  rm -f $OUT $OUT.part* $R/fr8k-$K-*.log
  echo "===== FR K=$K, 8 shards ====="
  t0=$(date +%s.%N)
  for i in 0 1 2 3 4 5 6 7; do
    ( export CUDA_VISIBLE_DEVICES=$i
      $R/iex-lpt build/pt_frwiki.bin 16 $K $OUT.part$i $i 8 > $R/fr8k-$K-$i.log 2>&1 ) &
  done
  wait
  t1=$(date +%s.%N)
  echo "shards wall: $(echo "$t1 - $t0" | bc) s"
  grep -h "gpu-index" $R/fr8k-$K-*.log | sed 's/.*entries/entries/'
  t2=$(date +%s.%N)
  $R/mergeidx $OUT $OUT.part0 $OUT.part1 $OUT.part2 $OUT.part3 $OUT.part4 $OUT.part5 $OUT.part6 $OUT.part7 | tail -1
  t3=$(date +%s.%N)
  echo "merge wall: $(echo "$t3 - $t2" | bc) s   total: $(echo "$t3 - $t0" | bc) s"
  ls -lh $OUT | awk '{print "SIZE " $5}'
  $R/paper5k cmp $OUT build/pt_frwiki.bin data/paper/frwiki.txt.query $K 5000 2>&1 | head -1
  rm -f $OUT.part*
done
echo FR8K_DONE
