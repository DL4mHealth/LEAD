export CUDA_VISIBLE_DEVICES=0,1,2,3

# Pretraining
python -u run.py --method LEADv2 --task_name pretrain_lead --is_training 1 --root_path ./dataset/L400/ --model_id P-Base --model LEADv2 --data MultiDatasets \
--pretraining_datasets TUEP,CAUEEG,BrainLat,BACA-RS,Depression,TDBrain,AD-Auditory,FEPCR,MCEF-RS,P-ADIC,PD-RS,PEARL-Neuro,SRM-RS \
--training_datasets ADFTD \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 2048 --n_heads 8 --d_model 128 --d_ff 256 \
--augmentations patch0.2,mask0.2,channel0.2 --patch_len 50 --stride 50 --group_shuffle --group_size 32 \
--sampling_rate_list 200,100,50 --montage_name standard_1005 --channel_names Fp1,Fp2,F7,F3,Fz,F4,F8,T7,C3,Cz,C4,T8,P7,P3,Pz,P4,P8,O1,O2 --swa \
--des 'Exp' --itr 1 --learning_rate 0.0002 --train_epochs 30