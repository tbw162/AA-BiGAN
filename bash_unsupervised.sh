#!/bin/bash
for i in $(seq 0 9)
do
    CUDA_VISIBLE_DEVICE=0 python main_unsupervised.py --normal_digit $i --gpu 0 --n_epochs 500  --batch_size 100  --latent_dim 128  --name cifar --gamma_p 0 --dataset CIFAR --dir /CIFAR0/summary//    
done
exit 0
