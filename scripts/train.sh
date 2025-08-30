# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

num_nodes=1
node_rank=0
master_addr=127.0.0.1
master_port=12345

llm_path=./models/Qwen2.5-0.5B-Instruct
vae_path=./models/BAGEL-7B-MoT/ae.safetensors
vit_path=./models/siglip-so400m-14-980-flash-attn2-navit
# Pre-training
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=1 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/aes.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --vae_path $vae_path \
  --vit_path $vit_path \
  --llm_path $llm_path \
  --use_flex True \
  --max_latent_size 64 \
  --max_num_tokens 18432 \
  --num_workers 1 \
  --num_shard 1 \
  --wandb_runid 0 \


# model_path=models/BAGEL-7B-MoT
# # Fine-tuning
# torchrun \
#   --nnodes=$num_nodes \
#   --node_rank=$node_rank \
#   --nproc_per_node=1 \
#   --master_addr=$master_addr \
#   --master_port=$master_port \
#   train/pretrain_unified_navit.py \
#   --dataset_config_file ./data/configs/example.yaml \
#   --model_path $model_path \
#   --layer_module Qwen2MoTDecoderLayer \
#   --max_latent_size 64 \
#   --resume-from $model_path \
#   --finetune_from_hf True \
#   --auto_resume True \
#   --resume-model-only True \
#   --finetune-from-ema True \
#   --log_every 1 \
#   --lr 2e-5 \
#   --num_worker 1 \
#   --expected_num_tokens 10240 \
#   --max_num_tokens 11520 \
#   --max_num_tokens_per_sample 10240