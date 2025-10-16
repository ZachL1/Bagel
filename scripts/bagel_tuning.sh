# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

pip install -U wandb
wandb login --relogin 2943b45498fccaa5f1941a06bcc942bf3bea49fb

num_nodes=1
node_rank=${ARNOLD_ID}
master_addr=${ARNOLD_WORKER_0_HOST}
master_port=(${ARNOLD_WORKER_0_PORT//,/ })

exp_name=from_bagel_7b_edit0.9_lang
output_path=./results/$exp_name
ckpt_path=$output_path/checkpoints

model_path=./models/BAGEL-7B-MoT
# Fine-tuning
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/aes_vlm.yaml \
  --model_path $model_path \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $model_path \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 10 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens 16384 \
  --max_num_tokens 36864 \
  --max_num_tokens_per_sample 16384 \
  --wandb_runid 121 \
  --save_every 2000 \
  --wandb_name $exp_name \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --visual_und True \
  --freeze_vit True \
  --freeze_und True \
  --aes_moe True \
  --train_moe True

  # --expected_num_tokens 10240 \
  # --max_num_tokens 11520 \
  # --max_num_tokens_per_sample 10240 \