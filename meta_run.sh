#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
# Run scripts sequentially


# pretrain
bash ./scripts/LEAD/pretrain/LEAD/P-Base.sh
# finetune
bash ./scripts/LEAD/finetune/LEAD/P-Base-F-1-Multi.sh

