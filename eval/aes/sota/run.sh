# pip install diffusers protobuf==3.20.0 huggingface-hub==0.34.0

# CUDA_BASE=0
# SPLIT_CNT=1
# for ((i=0; i<SPLIT_CNT; ++i)); do
#     CUDA_VISIBLE_DEVICES=$((CUDA_BASE + i)) python eval/aes/sota/flux_gen.py \
#         --json_path data/sft_data/Pexels/all_test_c.jsonl \
#         --data_path data/sft_data/Pexels \
#         --output_path results/pexels_eval/aes_edit_flux/edited_images \
#         --max_samples 1000 \
#         --data_split ${SPLIT_CNT}-${i} &
# done

# wait

# CUDA_BASE=1
# SPLIT_CNT=7
# for ((i=0; i<SPLIT_CNT; ++i)); do
#     CUDA_VISIBLE_DEVICES=$((CUDA_BASE + i)) python eval/aes/sota/qwen_gen.py \
#         --json_path data/sft_data/Pexels/all_test_c.jsonl \
#         --data_path data/sft_data/Pexels \
#         --output_path results/pexels_eval/aes_edit_qwen/edited_images \
#         --max_samples 1000 \
#         --data_split ${SPLIT_CNT}-${i} &
# done


# git clone -b step1xedit_v1p2 https://github.com/Peyton-Chen/diffusers.git
cd diffusers
pip install -e .
cd ..
pip install megfile transformers==4.55.0 qwen_vl_utils

CUDA_BASE=0
SPLIT_CNT=8
for ((i=0; i<SPLIT_CNT; ++i)); do
    CUDA_VISIBLE_DEVICES=$((CUDA_BASE + i)) python eval/aes/sota/step1x_gen.py \
        --json_path data/sft_data/Pexels/all_test_c.jsonl \
        --data_path data/sft_data/Pexels \
        --output_path results/pexels_eval/aes_edit_step1x/edited_images \
        --max_samples 1000 \
        --data_split ${SPLIT_CNT}-${i} &
done

wait

pip install -U git+https://github.com/huggingface/diffusers.git

CUDA_BASE=0
SPLIT_CNT=8
for ((i=0; i<SPLIT_CNT; ++i)); do
    CUDA_VISIBLE_DEVICES=$((CUDA_BASE + i)) python eval/aes/sota/hidream_gen.py \
        --json_path data/sft_data/Pexels/all_test_c.jsonl \
        --data_path data/sft_data/Pexels \
        --output_path results/pexels_eval/aes_edit_hidream/edited_images \
        --max_samples 1000 \
        --data_split ${SPLIT_CNT}-${i} &
done