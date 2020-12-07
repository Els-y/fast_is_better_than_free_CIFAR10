#!/usr/bin/env bash

ATTACK="pgd"
OUTPUT="weights/${ATTACK}"
GPU="6"

mkdir -p ${OUTPUT}

for epoch in {5..50..5}
do
    echo "start ${ATTACK} adv train with epoch ${epoch}..."

    fname="${OUTPUT}/checkpoint_${epoch}"
    CUDA_VISIBLE_DEVICES=${GPU} python train_cifar.py --attack=${ATTACK} \
                                                      --fname ${fname} \
                                                      --epochs ${epoch} \
                                                      --attack-iters 7
done
