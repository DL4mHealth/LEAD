export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training
python -u run.py --method LEAD --task_name supervised --is_training 1 --root_path ./dataset/ --model_id S-5-Sup --model LEAD --data MultiDatasets \
--training_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--testing_datasets ADFTD,CNBPM,Cognision-rsEEG-19,Cognision-ERP-19,BrainLat-19 \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15



# Testing
# ADFTD
python -u run.py --method LEAD --task_name supervised --is_training 0 --root_path ./dataset/ --model_id S-5-Sup --model LEAD --data MultiDatasets \
--testing_datasets ADFTD \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# CNBPM
python -u run.py --method LEAD --task_name supervised --is_training 0 --root_path ./dataset/ --model_id S-5-Sup --model LEAD --data MultiDatasets \
--testing_datasets CNBPM \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# Cognision-rsEEG-19
python -u run.py --method LEAD --task_name supervised --is_training 0 --root_path ./dataset/ --model_id S-5-Sup --model LEAD --data MultiDatasets \
--testing_datasets Cognision-rsEEG-19 \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# Cognision-ERP-19
python -u run.py --method LEAD --task_name supervised --is_training 0 --root_path ./dataset/ --model_id S-5-Sup --model LEAD --data MultiDatasets \
--testing_datasets Cognision-ERP-19 \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15

# BrainLat-19
python -u run.py --method LEAD --task_name supervised --is_training 0 --root_path ./dataset/ --model_id S-5-Sup --model LEAD --data MultiDatasets \
--testing_datasets BrainLat-19 \
--e_layers 12 --batch_size 128 --n_heads 8 --d_model 128 --d_ff 256 --swa \
--des 'Exp' --itr 5 --learning_rate 0.0001 --train_epochs 100 --patience 15