#!/usr/bin/env bash
#SBATCH --job-name=ner
#SBATCH --partition=3gpuq
##SBATCH --nodelist=zhangjiagang
##SBATCH -n 5
#SBATCH -o slurm.out
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

source activate pytorch0.4py3.6

dir=exp/1e-2
rm $dir/train.log

python main.py --lr 1e-2 --out $dir 