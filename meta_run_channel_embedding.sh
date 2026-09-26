#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Pretrain fixed/learnable channel-position variants
bash ./scripts/LEAD/pretrain/LEAD/P-Base-Channel-Embedding.sh
# Fixed channel embedding
bash ./scripts/LEAD/finetune/LEAD/P-Base-F-1-Multi-Channel-Fixed.sh
# Learnable channel embedding
bash ./scripts/LEAD/finetune/LEAD/P-Base-F-1-Multi-Channel-Learnable.sh
