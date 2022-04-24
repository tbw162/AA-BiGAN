#!/bin/bash
for i in $(seq 0 9)
do
    for j in $(seq 0 9)
    do
            if [ $i -eq $j ]; then
            continue
        fi

        CUDA_VISIBLE_DEVICE=0 python main.py --normal_digit $i --gpu 0 --n_epochs 200  --batch_size 200 --auxiliary_digit $j --latent_dim 100  --name mnist --gamma_p 0 --gamma_l 0.2 --k 1 --dataset MNIST --dir /MNIST0.2/summary//
    done
done
exit 0
