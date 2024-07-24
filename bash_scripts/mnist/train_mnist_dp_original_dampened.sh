#!/bin/bash
dataset="mnist"
algorithms=("dualprop-lagr-ff")
alphas=("1.0") # "0.5" "0.0")
nudged_passes=("8") # "40")

for N in "${nudged_passes[@]}"; do
    for alg in "${algorithms[@]}"; do
        for alpha in "${alphas[@]}"; do
            python train.py --model CNN --dataset ${dataset} --num-epochs 100 --batch-size 100 \
                            --alpha ${alpha} \
                            --beta 0.5 --loss sce \
                            --inference-sequence fwK --inference-passes-nudged ${N} \
                            --learning-rate 0.025\
                            --learning-rate-final 0.000002  --warmup-learning-rate 0.001 \
                            --learning-algorithm ${alg} --activation relu \
                            --decay-epochs 90 --warmup-epochs 10 --momentum 0.9 --weight-decay 2.5e-4 \
                            --dtype float32 --param-dtype float32 \
                            --percent-train 90 --percent-val 10 \
                            --seeds 9800 4486 17515\
                            --experiment-name test-${dataset}-alg-${alg}-alpha-${alpha}-nudged-passes-${N}
        done
    done
done