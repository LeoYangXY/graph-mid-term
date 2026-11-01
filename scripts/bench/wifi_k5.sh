#!/usr/bin/env bash
set -u
export TMPDIR=/data1/xuanyiyang/tmp
export LD_LIBRARY_PATH=/data1/xuanyiyang/cuda-13.0/lib64
cd /data1/xuanyiyang/workspace/graph-mid-term
R=build/k-sweep
for w in 10min 1h 3h; do
  echo "===== wifi_$w ====="
  $R/paper5k convert data/paper/wifi_$w.txt $R/wf_$w.bin 2>&1 | tail -1
  echo "-- 1 GPU, K=5 --"
  CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f "WALL %e s" $R/iex-lpt $R/wf_$w.bin 16 5 $R/wf_$w-k5.idx 2>&1 | grep -E "gpu-index|WALL"
  ls -lh $R/wf_$w-k5.idx | awk '{print "SIZE " $5}'
  echo "-- 8 GPU, K=5 --"
  OUT=$R/wf8_$w-k5.idx
  rm -f $OUT $OUT.part*
  t0=$(date +%s)
  for i in 0 1 2 3 4 5 6 7; do
    ( export CUDA_VISIBLE_DEVICES=$i; $R/iex-lpt $R/wf_$w.bin 16 5 $OUT.part$i $i 8 > $R/wf8_$w-$i.log 2>&1 ) &
  done
  wait
  t1=$(date +%s)
  $R/mergeidx $OUT $OUT.part0 $OUT.part1 $OUT.part2 $OUT.part3 $OUT.part4 $OUT.part5 $OUT.part6 $OUT.part7 | tail -1
  echo "8GPU total: $(( $(date +%s) - t0 )) s  (shards $((t1-t0)) s)"
  ls -lh $OUT | awk '{print "SIZE " $5}'
  rm -f $OUT.part*
done
echo WIFI_DONE
