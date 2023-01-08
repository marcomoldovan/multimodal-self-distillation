#!/bin/bash
#SBATCH -N 1
#SBATCH -p dgx-1-v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o multimodal-self-distillation.out
#SBATCH -e multimodal-self-distillation.err

conda activate multimodal-ssl
python train.py experiment=multimodal/experiment_name.yaml