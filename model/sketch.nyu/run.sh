#!/usr/bin/env bash
export NGPUS=2
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py -p 10097
python eval.py -e 200-250 -d 0-1 --save_path results