#!/bin/bash

# Example usage scripts for BAGEL inference

# python eval/aes/vlm_infer.py \
#     --tag aesbench_bagel \
#     --model_path ./models/BAGEL-7B-MoT \
#     --llm_path ./models/BAGEL-7B-MoT \
#     --eval_data_path data/sft_data/EAPD_release/AesBench_evaluation.json \
#     --image_dir data/sft_data/EAPD_release/images \
#     --output_dir results/aes_eval \
#     --max_mem_per_gpu 40GiB \
#     --seed 42

EXP_TAG=aes_edit_qwen_60
MODEL_PATH=./models/BAGEL-7B-MoT
LLM_PATH=./results/from_qwen25_7b_edit0.8_fix/checkpoints/0060000
mkdir -p results/aes_eval/$EXP_TAG
CUDA_BASE=0
SPLIT_CNT=8
for ((i=0; i<SPLIT_CNT; ++i)); do
    CUDA_VISIBLE_DEVICES=$((CUDA_BASE + i)) /bin/python -u eval/aes/edit_infer.py \
        --tag $EXP_TAG \
        --model_path $MODEL_PATH \
        --llm_path $LLM_PATH \
        --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
        --base_image_dir data/sft_data/AesEditor \
        --output_dir results/aes_eval \
        --max_mem_per_gpu 80GiB \
        --data_split ${SPLIT_CNT}-${i} \
        --max_samples 10000 \
        --seed 42 > results/aes_eval/$EXP_TAG/log${i}.txt &
done

# EXP_TAG=aes_edit_qwen_40
# MODEL_PATH=./models/BAGEL-7B-MoT
# LLM_PATH=./results/from_qwen25_7b_edit0.8_fix/checkpoints/0040000
# mkdir -p results/aes_eval/$EXP_TAG
# CUDA_BASE=4
# SPLIT_CNT=4
# for ((i=0; i<SPLIT_CNT; ++i)); do
#     CUDA_VISIBLE_DEVICES=$((CUDA_BASE + i)) /bin/python -u eval/aes/edit_infer.py \
#         --tag $EXP_TAG \
#         --model_path $MODEL_PATH \
#         --llm_path $LLM_PATH \
#         --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
#         --base_image_dir data/sft_data/AesEditor \
#         --output_dir results/aes_eval \
#         --max_mem_per_gpu 80GiB \
#         --data_split ${SPLIT_CNT}-${i} \
#         --max_samples 100 \
#         --seed 42 > results/aes_eval/$EXP_TAG/log${i}.txt &
# done

# EXP_TAG=aes_edit_bagel
# MODEL_PATH=./models/BAGEL-7B-MoT
# LLM_PATH=./models/BAGEL-7B-MoT
# mkdir -p results/aes_eval/$EXP_TAG
# CUDA_BASE=0
# SPLIT_CNT=4
# for ((i=0; i<SPLIT_CNT; ++i)); do
#     CUDA_VISIBLE_DEVICES=$((CUDA_BASE + i)) /bin/python -u eval/aes/edit_infer.py \
#         --tag $EXP_TAG \
#         --model_path $MODEL_PATH \
#         --llm_path $LLM_PATH \
#         --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
#         --base_image_dir data/sft_data/AesEditor \
#         --output_dir results/aes_eval \
#         --max_mem_per_gpu 80GiB \
#         --data_split ${SPLIT_CNT}-${i} \
#         --max_samples 10000 \
#         --seed 42 > results/aes_eval/$EXP_TAG/log${i}.txt &
# done


# EXP_TAG=aes_edit_bagel_14
# MODEL_PATH=./models/BAGEL-7B-MoT
# LLM_PATH=./results/from_bagel_7b_edit0.8_fix/checkpoints/0014000
# mkdir -p results/aes_eval/$EXP_TAG
# CUDA_BASE=4
# SPLIT_CNT=4
# for ((i=0; i<SPLIT_CNT; ++i)); do
#     CUDA_VISIBLE_DEVICES=$((CUDA_BASE + i)) /bin/python -u eval/aes/edit_infer.py \
#         --tag $EXP_TAG \
#         --model_path $MODEL_PATH \
#         --llm_path $LLM_PATH \
#         --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
#         --base_image_dir data/sft_data/AesEditor \
#         --output_dir results/aes_eval \
#         --max_mem_per_gpu 80GiB \
#         --data_split ${SPLIT_CNT}-${i} \
#         --seed 42 > results/aes_eval/$EXP_TAG/log${i}.txt &
# done