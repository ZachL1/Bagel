

CUDA_BASE=0
SPLIT_CNT=2
for ((i=0; i<SPLIT_CNT; ++i)); do
    CUDA_VISIBLE_DEVICES=$((CUDA_BASE + i)) python eval/aes/sota/flux_gen.py \
        --json_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
        --data_path data/sft_data/AesEditor \
        --output_path results/aes_eval/aes_edit_flux/edited_images \
        --max_samples 10 \
        --data_split ${SPLIT_CNT}-${i} &
done

# CUDA_BASE=0
# SPLIT_CNT=2
# for ((i=0; i<SPLIT_CNT; ++i)); do
#     CUDA_VISIBLE_DEVICES=$((CUDA_BASE + i)) python eval/aes/sota/qwen_gen.py \
#         --json_path data/sft_data/AesEditor/data_json/aes_edit_test.jsonl \
#         --data_path data/sft_data/AesEditor \
#         --output_path results/aes_eval/aes_edit_qwen/edited_images \
#         --max_samples 10 \
#         --data_split ${SPLIT_CNT}-${i}
# done