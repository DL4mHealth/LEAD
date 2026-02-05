export CUDA_VISIBLE_DEVICES=0,1,2,3

# Finetuning


# Multi-Class Classification
# ADFTD
# No subject cross entropy loss
python -u run.py --method LEADv2 --checkpoints_path ./checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/L400/ --model_id P-Base-F-ADFTD-Multi-No-Subject --model LEADv2 --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--augmentations flip,frequency,jitter,mask,channel,drop --patch_len 50 --stride 50 --sampling_rate_list 200,100,50 \
--ratio_a 0.8 --ratio_b 0.9 --montage_name standard_1005 --channel_names Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2 --use_subject_vote --swa \
--classify_choice multi_class --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# No shuffling
python -u run.py --method LEADv2 --checkpoints_path ./checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/L400/ --model_id P-Base-F-ADFTD-Multi-No-Shuffling --model LEADv2 --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--augmentations flip,frequency,jitter,mask,channel,drop --patch_len 50 --stride 50 --use_subject_loss --sampling_rate_list 200,100,50 \
--ratio_a 0.8 --ratio_b 0.9 --montage_name standard_1005 --channel_names Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2 --use_subject_vote --swa \
--classify_choice multi_class --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# Single sampling rate
python -u run.py --method LEADv2 --checkpoints_path ./checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/L400/ --model_id P-Base-F-ADFTD-Multi-Sampling-200 --model LEADv2 --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--augmentations flip,frequency,jitter,mask,channel,drop --patch_len 50 --stride 50 --group_shuffle --group_size 8 --use_subject_loss --sampling_rate_list 200 \
--ratio_a 0.8 --ratio_b 0.9 --montage_name standard_1005 --channel_names Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2 --use_subject_vote --swa \
--classify_choice multi_class --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# Group Size 4
python -u run.py --method LEADv2 --checkpoints_path ./checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/L400/ --model_id P-Base-F-ADFTD-Multi-Group-4 --model LEADv2 --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--augmentations flip,frequency,jitter,mask,channel,drop --patch_len 50 --stride 50 --group_shuffle --group_size 4 --use_subject_loss --sampling_rate_list 200,100,50 \
--ratio_a 0.8 --ratio_b 0.9 --montage_name standard_1005 --channel_names Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2 --use_subject_vote --swa \
--classify_choice multi_class --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15

# Group Size 16
python -u run.py --method LEADv2 --checkpoints_path ./checkpoints/LEADv2/pretrain_lead/LEADv2/P-Base/nh8_el12_dm128_df256_seed41/checkpoint.pth \
--task_name finetune --is_training 1 --root_path ./dataset/L400/ --model_id P-Base-F-ADFTD-Multi-Group-16 --model LEADv2 --data MultiDatasets \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 512 --n_heads 8 --d_model 128 --d_ff 256 \
--augmentations flip,frequency,jitter,mask,channel,drop --patch_len 50 --stride 50 --group_shuffle --group_size 16 --use_subject_loss --sampling_rate_list 200,100,50 \
--ratio_a 0.8 --ratio_b 0.9 --montage_name standard_1005 --channel_names Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2 --use_subject_vote --swa \
--classify_choice multi_class --des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 200 --patience 15


