#!/bin/bash
dataset="imagenet_32x32"
algorithms=("dualprop-lagr-ff")
alphas=("0.0")
beta=("0.1")
nudged_passes=("16")
seed=("8697")

for N in "${nudged_passes[@]}"; do
    for alg in "${algorithms[@]}"; do
        for alpha in "${alphas[@]}"; do
            python train.py --model VGG16 --dataset ${dataset} --num-epochs 130 --batch-size 250 \
                            --alpha ${alpha} \
                            --beta ${beta} --loss sce \
                            --inference-sequence fwK --inference-passes-nudged ${N} \
                            --learning-rate 0.02\
                            --learning-rate-final 0.000002  --warmup-learning-rate 0.001 \
                            --learning-algorithm ${alg} --activation relu \
                            --decay-epochs 120 --warmup-epochs 10 --momentum 0.9 --weight-decay 5e-4 \
                            --dtype float32 --param-dtype float32 \
                            --percent-train 95 --percent-val 5 \
                            --seeds ${seed} \
                            --experiment-name ${dataset}-seed-${seed}-alg-${alg}-alpha-${alpha}-nudged-passes-${N}
        done
    done
done