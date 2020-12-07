#!/usr/bin/env bash

ATTACK="free"
OUTPUT="weights/${ATTACK}"
GPU="7"

mkdir -p ${OUTPUT}

for epoch in {8..96..8}
do
    echo "start ${ATTACK} adv train with epoch ${epoch}..."

    fname="${OUTPUT}/checkpoint_${epoch}"
    CUDA_VISIBLE_DEVICES=${GPU} python train_cifar.py --attack=${ATTACK} \
                                                      --fname ${fname} \
                                                      --epochs ${epoch} \
                                                      --attack-iters 8 \
                                                      --lr-max 0.04
done
