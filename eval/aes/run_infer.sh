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


python eval/aes/edit_infer.py \
    --tag aes_edit_bagel \
    --model_path ./models/BAGEL-7B-MoT \
    --llm_path ./models/BAGEL-7B-MoT \
    --edit_data_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
    --base_image_dir data/sft_data/AesEditor \
    --output_dir results/aes_eval \
    --max_mem_per_gpu 40GiB \
    --seed 42
