#!/usr/bin/env bash

source activate pytorch0.4py3.6

dir=exp/1e-2
rm $dir/train.log

python main.py --lr 1e-2 --out $dir