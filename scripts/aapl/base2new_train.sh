#!/bin/bash

# cd ../..

# custom config
# DATA=/path/to/datasets
DATA=/SSD2/data/
TRAINER=AAPL

DATASET=$1
SEED=$2
GPU=$3

CFG=vit_b16_c4_ep10_batch1
# CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=16

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=${GPU} \
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi
