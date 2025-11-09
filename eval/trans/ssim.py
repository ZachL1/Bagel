import os
from tqdm import tqdm
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import multiprocessing as mp
import argparse
import json

def split_image_into_patches(img, patch_size):
    patches = []
    height, width, channels = img.shape
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

def calculate_ssim_for_files(g_file, r_file, patch_size):
    g_img = cv2.imread(g_file)
    r_img = cv2.imread(r_file)
    
    if g_img is None or r_img is None:
        print(f"Warning: Could not read images: {g_file} or {r_file}")
        return 0.0

    g_patches = split_image_into_patches(g_img, patch_size)
    r_patches = split_image_into_patches(r_img, patch_size)

    patch_ssims = []
    for g_patch, r_patch in zip(g_patches, r_patches):
        patch_ssim = compare_ssim(g_patch, r_patch, channel_axis=-1, multichannel=True)
        patch_ssims.append(patch_ssim)

    return np.mean(patch_ssims)

def process_files(file_pair):
    g_file, r_file = file_pair
    ssims = {}
    for patch_size in patch_sizes:
        ssims[patch_size] = calculate_ssim_for_files(g_file, r_file, patch_size)
    return ssims

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, default="data/trans_data/annotations_with_paths.jsonl", help="Path to jsonl file containing image paths.")
    parser.add_argument("--ref_key", type=str, default="tgt_img_path", help="Key for reference image path in jsonl.")
    parser.add_argument("--eval_key", type=str, default="tgt_img_path", help="Key for evaluation image path in jsonl.")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of workers for multiprocessing.")
    args = parser.parse_args()

    # Read jsonl file
    items = []
    with open(args.jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line.strip()))
    
    # Extract file pairs
    test_result_files = []
    ref_result_files = []
    for item in items:
        eval_file = item[args.eval_key]
        ref_file = item[args.ref_key]
        
        # Check if files exist
        if not os.path.exists(eval_file):
            print(f"Warning: Eval file not found: {eval_file}")
        if not os.path.exists(ref_file):
            print(f"Warning: Reference file not found: {ref_file}")
        
        test_result_files.append(eval_file)
        ref_result_files.append(ref_file)
    
    psnrs = []
    ssims = []
    mses = []
    patch_sizes = [512]
    ssims = {patch_size: [] for patch_size in patch_sizes}
    print(f"CPU count: {mp.cpu_count()}")
    print(f"Using {args.num_workers} workers")
    print(f"Processing {len(test_result_files)} image pairs")
    
    with mp.Pool(args.num_workers) as pool:
        for file_ssims in tqdm(pool.imap(process_files, zip(test_result_files, ref_result_files)), total=len(test_result_files)):
            for patch_size in patch_sizes:
                ssims[patch_size].append(file_ssims[patch_size])
    average_ssims = {patch_size: np.mean(ssims[patch_size]) for patch_size in patch_sizes}

    for patch_size in patch_sizes:
        print(f"Average SSIM for patch {patch_size}: {average_ssims[patch_size]}")
    