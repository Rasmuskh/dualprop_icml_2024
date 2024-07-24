#!/bin/bash
dataset="cifar10"
algorithms=("dualprop-lagr-ff")
alphas=("0.5")
beta=("0.1")
nudged_passes=("16")

for N in "${nudged_passes[@]}"; do
    for alg in "${algorithms[@]}"; do
        for alpha in "${alphas[@]}"; do
            python train.py --model VGG16 --dataset ${dataset} --num-epochs 130 --batch-size 100 \
                            --alpha ${alpha} \
                            --beta ${beta} --loss sce \
                            --inference-sequence fwK --inference-passes-nudged ${N} \
                            --learning-rate 0.025\
                            --learning-rate-final 0.000002  --warmup-learning-rate 0.001 \
                            --learning-algorithm ${alg} --activation relu \
                            --decay-epochs 120 --warmup-epochs 10 --momentum 0.9 --weight-decay 5e-4 \
                            --dtype float32 --param-dtype float32 \
                            --percent-train 90 --percent-val 10 \
                            --seeds 1988 1542 8697\
                            --experiment-name ${dataset}-alg-${alg}-alpha-${alpha}-beta-${beta}-nudged-passes-${N}
        done
    done
done