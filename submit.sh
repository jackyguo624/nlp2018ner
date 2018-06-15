#!/usr/bin/env bash
#SBATCH --job-name=ner
#SBATCH --partition=3gpuq
##SBATCH --nodelist=zhangjiagang
##SBATCH -n 5
#SBATCH -o slurm.out
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

source activate pytorch0.4py3.6
lr=1
loss=--loss-viterbi
dir=exp/${lr}${loss}
rm $dir/train.log

python main.py --lr ${lr} --out $dir --cuda ${loss}