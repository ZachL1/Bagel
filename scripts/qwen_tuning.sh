# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

num_nodes=1
node_rank=${ARNOLD_ID}
master_addr=${ARNOLD_WORKER_0_HOST}
master_port=(${ARNOLD_WORKER_0_PORT//,/ })

exp_name=from_qwen25_7b_edit0.8_fix
output_path=./results/$exp_name
ckpt_path=$output_path/checkpoints

llm_path=./models/Qwen2.5-7B-Instruct
vae_path=./models/BAGEL-7B-MoT/ae.safetensors
vit_path=./models/siglip-so400m-14-980-flash-attn2-navit
# Pre-training
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/aes.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --vae_path $vae_path \
  --vit_path $vit_path \
  --llm_path $llm_path \
  --use_flex True \
  --max_latent_size 64 \
  --expected_num_tokens 16384 \
  --max_num_tokens 36864 \
  --max_num_tokens_per_sample 16384 \
  --num_workers 1 \
  --num_shard 8 \
  --wandb_runid 6 \
  --wandb_name $exp_name \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
