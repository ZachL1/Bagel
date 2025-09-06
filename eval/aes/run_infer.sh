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


mkdir -p results/aes_eval/aes_edit_bagel
CUDA_VISIBLE_DEVICES=0 /bin/python -u eval/aes/edit_infer.py \
    --tag aes_edit_bagel \
    --model_path ./models/BAGEL-7B-MoT \
    --llm_path ./models/BAGEL-7B-MoT \
    --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
    --base_image_dir data/sft_data/AesEditor \
    --output_dir results/aes_eval \
    --max_mem_per_gpu 80GiB \
    --seed 42 > results/aes_eval/aes_edit_bagel/log.txt &


mkdir -p results/aes_eval/aes_edit_bagel_02
CUDA_VISIBLE_DEVICES=1 /bin/python -u eval/aes/edit_infer.py \
    --tag aes_edit_bagel_02 \
    --model_path ./models/BAGEL-7B-MoT \
    --llm_path ./results/from_bagel_7b_edit0.8_fix/checkpoints/0002000 \
    --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
    --base_image_dir data/sft_data/AesEditor \
    --output_dir results/aes_eval \
    --max_mem_per_gpu 80GiB \
    --seed 42 > results/aes_eval/aes_edit_bagel_02/log.txt &


mkdir -p results/aes_eval/aes_edit_bagel_06
CUDA_VISIBLE_DEVICES=2 /bin/python -u eval/aes/edit_infer.py \
    --tag aes_edit_bagel_06 \
    --model_path ./models/BAGEL-7B-MoT \
    --llm_path ./results/from_bagel_7b_edit0.8_fix/checkpoints/0006000 \
    --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
    --base_image_dir data/sft_data/AesEditor \
    --output_dir results/aes_eval \
    --max_mem_per_gpu 80GiB \
    --seed 42 > results/aes_eval/aes_edit_bagel_06/log.txt &


mkdir -p results/aes_eval/aes_edit_bagel_10
CUDA_VISIBLE_DEVICES=3 /bin/python -u eval/aes/edit_infer.py \
    --tag aes_edit_bagel_10 \
    --model_path ./models/BAGEL-7B-MoT \
    --llm_path ./results/from_bagel_7b_edit0.8_fix/checkpoints/0010000 \
    --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
    --base_image_dir data/sft_data/AesEditor \
    --output_dir results/aes_eval \
    --max_mem_per_gpu 80GiB \
    --seed 42 > results/aes_eval/aes_edit_bagel_10/log.txt &


mkdir -p results/aes_eval/aes_edit_bagel_20
CUDA_VISIBLE_DEVICES=4 /bin/python -u eval/aes/edit_infer.py \
    --tag aes_edit_bagel_20 \
    --model_path ./models/BAGEL-7B-MoT \
    --llm_path ./results/from_bagel_7b_edit0.8_fix/checkpoints/0020000 \
    --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
    --base_image_dir data/sft_data/AesEditor \
    --output_dir results/aes_eval \
    --max_mem_per_gpu 80GiB \
    --seed 42 > results/aes_eval/aes_edit_bagel_20/log.txt &

mkdir -p results/aes_eval/aes_edit_bagel_28
CUDA_VISIBLE_DEVICES=5 /bin/python -u eval/aes/edit_infer.py \
    --tag aes_edit_bagel_28 \
    --model_path ./models/BAGEL-7B-MoT \
    --llm_path ./results/from_bagel_7b_edit0.8_fix/checkpoints/0028000 \
    --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
    --base_image_dir data/sft_data/AesEditor \
    --output_dir results/aes_eval \
    --max_mem_per_gpu 80GiB \
    --seed 42 > results/aes_eval/aes_edit_bagel_28/log.txt &

mkdir -p results/aes_eval/aes_edit_bagel_14
CUDA_VISIBLE_DEVICES=6 /bin/python -u eval/aes/edit_infer.py \
    --tag aes_edit_bagel_14 \
    --model_path ./models/BAGEL-7B-MoT \
    --llm_path ./results/from_bagel_7b_edit0.8_fix/checkpoints/0014000 \
    --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
    --base_image_dir data/sft_data/AesEditor \
    --output_dir results/aes_eval \
    --max_mem_per_gpu 80GiB \
    --seed 42 > results/aes_eval/aes_edit_bagel_14/log.txt &

mkdir -p results/aes_eval/aes_edit_bagel_04
CUDA_VISIBLE_DEVICES=7 /bin/python -u eval/aes/edit_infer.py \
    --tag aes_edit_bagel_04 \
    --model_path ./models/BAGEL-7B-MoT \
    --llm_path ./results/from_bagel_7b_edit0.8_fix/checkpoints/0004000 \
    --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
    --base_image_dir data/sft_data/AesEditor \
    --output_dir results/aes_eval \
    --max_mem_per_gpu 80GiB \
    --seed 42 > results/aes_eval/aes_edit_bagel_04/log.txt &