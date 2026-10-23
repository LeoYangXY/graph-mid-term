#!/usr/bin/env bash
# full pipeline: transform -> online query -> cpu batch -> gpu batch
# usage: bash scripts/run_all.sh [dataset] [k] [S]
set -e
DS=$1
if [ -z "$DS" ]; then DS=twitter; fi
K=$2
if [ -z "$K" ]; then K=3; fi
S=$3
if [ -z "$S" ]; then S=1024; fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
G=data/bigraph/$DS.txt
Q=data/query/$DS.query_.queryres_0
T=build/tg_$DS.bin
mkdir -p build results
echo === 1 stats ===;     ./build/tb stats "$G"
echo === 2 transform ===; ./build/tb transform "$G" "$T"
echo === 3 online ===;    ./build/tb online "$T" "$Q" "$K" 2000
echo === 4 cpu batch ===; ./build/tb batch "$T" "$S" "$K" 0 2147483647 16
echo === 5 gpu batch ===; ./build/batch_gpu "$T" "$S" "$K" 0 3
