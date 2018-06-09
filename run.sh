#!/bin/bash

###############################################
# Training on primitive dataset using LBA1 
###############################################
if [ 1 -eq 0 ]; then  
    CUDA_VISIBLE_DEVICES='6,7' python main_primitive_lba1.py --cuda \
        --ngpu 1 \
        --text_encoder \
        --learning_rate 2e-4 \
        --decay_steps 5000 \
        --LBA_test_mode shape \
        --LBA_model_type MM \
        --LBA_cosin_dist \
        --LBA_unnormalize \
        --batch_size 100 
fi 

##############################################
## Evaluation on primitive dataset (text mode) 
############################################## 
if [ 1 -eq 1 ]; then 
    CUDA_VISIBLE_DEVICES='6,7' python eval_primitive_lba1.py --cuda \
        --ngpu 1 \
        --text_encoder \
        --val_split 'test' \
        --pretrained_model ./models_checkpoint/model_best.pth \
        --LBA_test_mode text \
        --LBA_unnormalize 
fi 

