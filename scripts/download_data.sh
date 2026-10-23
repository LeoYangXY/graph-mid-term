#!/usr/bin/env bash
# CBSR temporal-bipartite datasets (11 graphs, 100k queries each)
set -e
mkdir -p data
cd data
python3 -m pip install gdown --quiet
python3 -c "import gdown; gdown.download(id='1CP_jigGHExmbNYWpxTBNR9-3JnBJraob', output='datasets.zip')"
unzip -q datasets.zip
mv CBSR_dataset/* . && rm -rf CBSR_dataset datasets.zip
ls bigraph
