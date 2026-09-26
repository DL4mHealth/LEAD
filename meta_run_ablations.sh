#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# These two ablations require their own pretraining checkpoints.
# No group shuffle pretraining
bash ./scripts/LEAD/pretrain/LEAD/P-Base-No-Group-Shuffle.sh
# No Group Shuffle
bash ./scripts/LEAD/finetune/LEAD/P-Base-F-1-Multi-No-Group-Shuffle.sh
# No Multi Segmentation
bash ./scripts/LEAD/finetune/LEAD/P-Base-F-1-Multi-No-Multi-Seg.sh
# No Subject Loss
bash ./scripts/LEAD/finetune/LEAD/P-Base-F-1-Multi-No-Subject-Loss.sh
# No sampling-rate embedding pretraining
bash ./scripts/LEAD/pretrain/LEAD/P-Base-No-Sampling-Embed.sh
# No Sampling-rate Embedding
bash ./scripts/LEAD/finetune/LEAD/P-Base-F-1-Multi-No-Sampling-Embed.sh
