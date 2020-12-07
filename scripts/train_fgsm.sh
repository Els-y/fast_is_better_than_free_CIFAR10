#!/usr/bin/env bash

OUTPUT="weights/fgsm"
GPU="5"

mkdir -p ${OUTPUT}

for epoch in {5..50..5}
do
    echo "start fgsm adv train with epoch ${epoch}..."

    fname="${OUTPUT}/checkpoint_${epoch}"
    CUDA_VISIBLE_DEVICES=${GPU} python train_cifar.py --attack=fgsm \
                                                      --fname ${fname} \
                                                      --epochs ${epoch}
done
