import json
import numpy as np
import math
import cv2
import lpips
import os
import torch
from PIL import Image
from tqdm import tqdm

import concurrent.futures

bench_json = "data/sft_data/AesEditor/data_json/aes_edit_test.jsonl"
data_dir = "data/sft_data/AesEditor/"
result_dir = "results/aes_eval/aes_edit_bagel/edited_images"


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def load_image(image_path):
    """Load image and convert to numpy array in range [0, 255]"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def tensor_from_image(img):
    """Convert numpy image to tensor for LPIPS calculation"""
    # Convert from [0, 255] to [-1, 1] range as expected by LPIPS
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return img_tensor * 2 - 1

def main_task(lines, source_metrics, worker_id):
    for line in tqdm(lines, desc=f"Processing {worker_id}: "):
        item = json.loads(line.strip())
        image_name = item["target"]
        source = item.get("source", "unknown")  # Get source field, default to "unknown"
        
        # Initialize source group if not exists
        if source not in source_metrics:
            source_metrics[source] = {
                'psnr_scores': [],
                'ssim_scores': [],
                'lpips_scores': [],
                'processed_count': 0,
                'skipped_count': 0
            }
        
        # Construct image paths
        target_path = os.path.join(data_dir, image_name)
        result_path = os.path.join(result_dir, image_name.rsplit(".", 1)[0] + ".png")
        if not os.path.exists(result_path):
            source_metrics[source]['skipped_count'] += 1
            continue
        
        # Load images
        target_img = load_image(target_path)
        result_img = load_image(result_path)
        
        # Ensure images have the same dimensions
        if target_img.shape != result_img.shape:
            # resize target to result
            target_img = cv2.resize(target_img, (result_img.shape[1], result_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Calculate PSNR
        psnr = calculate_psnr(target_img, result_img)
        source_metrics[source]['psnr_scores'].append(psnr)
        
        # Calculate SSIM
        ssim_score = calculate_ssim(target_img, result_img)
        source_metrics[source]['ssim_scores'].append(ssim_score)
        
        # Calculate LPIPS
        target_tensor = tensor_from_image(target_img).to(device)
        result_tensor = tensor_from_image(result_img).to(device)
        
        with torch.no_grad():
            lpips_score = lpips_fn(target_tensor, result_tensor).squeeze().item()
        source_metrics[source]['lpips_scores'].append(lpips_score)
        
        source_metrics[source]['processed_count'] += 1
        
        # print(f"Processed {processed_count:3d}: [{source}] {image_name} - "
        #       f"PSNR: {psnr:.6f}, SSIM: {ssim_score:.6f}, LPIPS: {lpips_score:.6f}")


if __name__ == "__main__":
    # Initialize LPIPS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    # Storage for metrics grouped by source
    source_metrics = {}
    
    # Read and process each line in the JSONL file
    with open(bench_json, 'r') as f:
        bench_data = f.readlines()
    
    num_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(main_task, bench_data[i::num_workers], source_metrics, i) for i in range(num_workers)]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Working"):
            future.result()
    
    # Calculate and display results grouped by source
    print("\n" + "="*80)
    print("EVALUATION RESULTS BY SOURCE")
    print("="*80)
    
    overall_psnr = []
    overall_ssim = []
    overall_lpips = []
    
    for source, metrics in source_metrics.items():
        if metrics['psnr_scores']:
            avg_psnr = np.mean(metrics['psnr_scores'])
            avg_ssim = np.mean(metrics['ssim_scores'])
            avg_lpips = np.mean(metrics['lpips_scores'])
            
            overall_psnr.extend(metrics['psnr_scores'])
            overall_ssim.extend(metrics['ssim_scores'])
            overall_lpips.extend(metrics['lpips_scores'])
            
            print(f"\nSource: {source}")
            print("-" * 40)
            print(f"Processed images: {metrics['processed_count']}")
            print(f"Skipped images: {metrics['skipped_count']}")
            print(f"Average PSNR: {avg_psnr:.6f} dB")
            print(f"Average SSIM: {avg_ssim:.6f}")
            print(f"Average LPIPS: {avg_lpips:.6f}")
        else:
            print(f"\nSource: {source}")
            print("-" * 40)
            print("No images were successfully processed!")
    
    # Overall results
    if overall_psnr:
        print("\n" + "="*80)
        print("OVERALL EVALUATION RESULTS")
        print("="*80)
        print(f"Total processed images: {sum(metrics['processed_count'] for metrics in source_metrics.values())}")
        print(f"Total skipped images: {sum(metrics['skipped_count'] for metrics in source_metrics.values())}")
        print(f"Overall Average PSNR: {np.mean(overall_psnr):.6f} dB")
        print(f"Overall Average SSIM: {np.mean(overall_ssim):.6f}")
        print(f"Overall Average LPIPS: {np.mean(overall_lpips):.6f}")
        print("="*80)
    else:
        print("\nNo images were successfully processed!")

