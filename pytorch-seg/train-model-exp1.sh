#!/usr/bin/env bash
python train.py \
--dataroot "/media/shrek/work/datasets/public/shuran-mini-pytorch-examples/" \
--file_list "./data/datalist" \
--batchSize 4 \
--shuffle True \
--phase train \
--num_epochs 1 \
--imsize 224 \
--logs_path "logs/exp3"

