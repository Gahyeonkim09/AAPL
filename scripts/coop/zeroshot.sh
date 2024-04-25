#!/bin/bash

# custom config
DATA=/SSD2/embedding_run_1/CoOp/datasets/
TRAINER=ZeroshotCLIP
DATASET=$1      # caltech101 
CFG=$2  # rn50, rn101, vit_b32 or vit_b16

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only