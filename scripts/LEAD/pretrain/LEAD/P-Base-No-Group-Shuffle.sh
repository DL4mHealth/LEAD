#!/usr/bin/env bash

# Pretraining ablation: dataset-homogeneous batches only.
# --group_shuffle is intentionally omitted, so --group_size is inactive.
python -u run.py --method LEAD --task_name pretrain --is_training 1 --root_path ./dataset/ --model_id P-11-b1024-p50-no-group-learnable-3D --model LEAD --data MultiDatasets \
--pretraining_datasets TUEP,CAUEEG,BACA-RS,TDBrain,AD-Auditory,FEPCR,MCEF-RS,P-ADIC,PD-RS,PEARL-Neuro,SRM-RS \
--training_dataset ADFTD \
--e_layers 12 --batch_size 1024 --n_heads 8 --d_model 128 --d_ff 256 --augmentations patch0.2,mask0.2,channel0.2 --patch_len 50 --stride 50 \
--temporal_pos_type learnable --channel_pos_type 3D --use_sampling_embedding --sampling_rate_list 200,100,50 --contrastive_token_ratio 1.0 --ratio_a 0.8 --ratio_b 0.9 \
--des 'Exp' --itr 1 --learning_rate 0.0002 --train_epochs 30
