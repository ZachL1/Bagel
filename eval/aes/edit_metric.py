import json
import numpy as np
import math
import cv2
import lpips
import os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from skimage import color
import clip
import sys
import concurrent.futures
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IQA-PyTorch'))
import pyiqa

# pip install git+https://github.com/openai/CLIP.git
# pip install lpips scikit-image==0.24
# pip install timm icecream transformers==4.37.2 # for pyiqa

bench_json = "data/sft_data/AesEditor/data_json/ae_test.jsonl"
data_dir = "data/sft_data/AesEditor/"
result_dir = "results/aes_eval_bak/aes_edit_bagel/edited_images"


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


def calculate_delta_e(img1, img2):
    """Calculate CIELAB color difference (Delta E)
    img1, img2: [0, 255] RGB images
    Returns: mean Delta E value
    """
    # Convert RGB to LAB color space
    # Input should be in [0, 1] range for skimage
    img1_lab = color.rgb2lab(img1 / 255.0)
    img2_lab = color.rgb2lab(img2 / 255.0)
    
    # Calculate Euclidean distance in LAB space
    delta_e = np.sqrt(np.sum((img1_lab - img2_lab) ** 2, axis=2))
    
    # Return mean Delta E
    return np.mean(delta_e)


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
        instruction = item.get("instruction", "")  # Get instruction for CLIP-T
        
        # Initialize source group if not exists
        if source not in source_metrics:
            source_metrics[source] = {
                'psnr_scores': [],
                'ssim_scores': [],
                'lpips_scores': [],
                'delta_e_scores': [],
                'clip_t_scores': [],
                'clip_i_scores': [],
                'aesclip_t_scores': [],
                'aesclip_i_scores': [],
                'niqe_scores': [],
                'nima_scores': [],
                'musiq_scores': [],
                'qalign_scores': [],
                'processed_count': 0,
                'skipped_count': 0
            }
        
        # Construct image paths
        target_path = os.path.join(data_dir, image_name)
        # result_path = os.path.join(result_dir, image_name)
        result_path = os.path.join(data_dir, item["raw"])
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
        
        # Calculate Delta E (CIELAB color difference)
        delta_e = calculate_delta_e(target_img, result_img)
        source_metrics[source]['delta_e_scores'].append(delta_e)
        
        # Calculate CLIP-T (text-image similarity) and CLIP-I (image-image similarity)
        with torch.no_grad():
            # Convert images to PIL for CLIP preprocessing
            target_pil = Image.fromarray(target_img.astype(np.uint8))
            result_pil = Image.fromarray(result_img.astype(np.uint8))
            
            # Preprocess images for CLIP
            target_clip = clip_preprocess(target_pil).unsqueeze(0).to(device)
            result_clip = clip_preprocess(result_pil).unsqueeze(0).to(device)
            # Encode images
            target_feature = clip_model.encode_image(target_clip)
            result_feature = clip_model.encode_image(result_clip)
            # Normalize features
            target_feature = F.normalize(target_feature, p=2, dim=1)
            result_feature = F.normalize(result_feature, p=2, dim=1)
            # Calculate CLIP-I (image-image similarity)
            clip_i_score = (target_feature * result_feature).sum(dim=1).cpu().item()
            source_metrics[source]['clip_i_scores'].append(clip_i_score)
            
            # Calculate AesCLIP-T (text-image similarity) and AesCLIP-I (image-image similarity)
            target_aesclip = aesclip_preprocess(target_pil).unsqueeze(0).to(device)
            result_aesclip = aesclip_preprocess(result_pil).unsqueeze(0).to(device)
            target_aesclip_feature = aesclip_model.encode_image(target_aesclip)
            result_aesclip_feature = aesclip_model.encode_image(result_aesclip)
            target_aesclip_feature = F.normalize(target_aesclip_feature, p=2, dim=1)
            result_aesclip_feature = F.normalize(result_aesclip_feature, p=2, dim=1)
            aesclip_i_score = (target_aesclip_feature * result_aesclip_feature).sum(dim=1).cpu().item()
            source_metrics[source]['aesclip_i_scores'].append(aesclip_i_score)

            # Calculate CLIP-T (text-image similarity) if instruction exists
            if instruction:
                text_token = clip.tokenize([instruction]).to(device)
                text_feature = clip_model.encode_text(text_token)
                text_feature = F.normalize(text_feature, p=2, dim=1)
                clip_t_score = (text_feature * result_feature).sum(dim=1).cpu().item()
                source_metrics[source]['clip_t_scores'].append(clip_t_score)
                
                text_aesclip_feature = aesclip_model.encode_text(text_token)
                text_aesclip_feature = F.normalize(text_aesclip_feature, p=2, dim=1)
                aesclip_t_score = (text_aesclip_feature * result_aesclip_feature).sum(dim=1).cpu().item()
                source_metrics[source]['aesclip_t_scores'].append(aesclip_t_score)
        
        # Calculate pyiqa metrics (NIQE, NIMA, MUSIQ, Q-Align)
        # Convert result image to tensor format for pyiqa [0, 1] range
        result_tensor_pyiqa = torch.from_numpy(result_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        result_tensor_pyiqa = result_tensor_pyiqa.to(device)
        
        # with torch.no_grad():
        #     # NIQE (No-Reference Image Quality Assessment)
        #     niqe_score = niqe_metric(result_tensor_pyiqa).cpu().item()
        #     source_metrics[source]['niqe_scores'].append(niqe_score)
            
        #     # NIMA (Neural Image Assessment)
        #     nima_score = nima_metric(result_tensor_pyiqa).cpu().item()
        #     source_metrics[source]['nima_scores'].append(nima_score)
            
        #     # MUSIQ (Multi-Scale Image Quality Transformer)
        #     musiq_score = musiq_metric(result_tensor_pyiqa).cpu().item()
        #     source_metrics[source]['musiq_scores'].append(musiq_score)
            
        #     # Q-Align (aesthetic score)
        #     qalign_score = qalign_metric(result_tensor_pyiqa, task_='aesthetic').cpu().item()
        #     # qalign_score = qalign_metric(result_tensor_pyiqa, task_='quality').cpu().item()
        #     source_metrics[source]['qalign_scores'].append(qalign_score)
        
        source_metrics[source]['processed_count'] += 1
        
        # print(f"Processed {processed_count:3d}: [{source}] {image_name} - "
        #       f"PSNR: {psnr:.6f}, SSIM: {ssim_score:.6f}, LPIPS: {lpips_score:.6f}")


if __name__ == "__main__":
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    # Initialize CLIP
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # Initialize AesCLIP
    print("Loading AesCLIP model...")
    aesclip_model, aesclip_preprocess = clip.load("ViT-B/16", device=device)
    aesclip_model.load_state_dict(torch.load("models/AesCLIP_weight/AesCLIP", map_location=device))
    
    # Initialize pyiqa metrics
    print("Loading pyiqa metrics...")
    # niqe_metric = pyiqa.create_metric('niqe', device=device)
    # nima_metric = pyiqa.create_metric('nima', device=device)
    # musiq_metric = pyiqa.create_metric('musiq-ava', device=device)
    # qalign_metric = pyiqa.create_metric('qalign', device=device)
    
    # Storage for metrics grouped by source
    source_metrics = {}
    
    # Read and process each line in the JSONL file
    with open(bench_json, 'r') as f:
        bench_data = f.readlines()
    
    num_workers = 16
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
    overall_delta_e = []
    overall_clip_t = []
    overall_clip_i = []
    overall_aesclip_t = []
    overall_aesclip_i = []
    overall_niqe = []
    overall_nima = []
    overall_musiq = []
    overall_qalign = []
    
    for source, metrics in source_metrics.items():
        if metrics['psnr_scores']:
            avg_psnr = np.mean(metrics['psnr_scores'])
            avg_ssim = np.mean(metrics['ssim_scores'])
            avg_lpips = np.mean(metrics['lpips_scores'])
            avg_delta_e = np.mean(metrics['delta_e_scores'])
            avg_clip_i = np.mean(metrics['clip_i_scores']) if metrics['clip_i_scores'] else 0.0
            avg_clip_t = np.mean(metrics['clip_t_scores']) if metrics['clip_t_scores'] else 0.0
            avg_aesclip_i = np.mean(metrics['aesclip_i_scores']) if metrics['aesclip_i_scores'] else 0.0
            avg_aesclip_t = np.mean(metrics['aesclip_t_scores']) if metrics['aesclip_t_scores'] else 0.0
            avg_niqe = np.mean(metrics['niqe_scores'])
            avg_nima = np.mean(metrics['nima_scores'])
            avg_musiq = np.mean(metrics['musiq_scores'])
            avg_qalign = np.mean(metrics['qalign_scores'])
            
            overall_psnr.extend(metrics['psnr_scores'])
            overall_ssim.extend(metrics['ssim_scores'])
            overall_lpips.extend(metrics['lpips_scores'])
            overall_delta_e.extend(metrics['delta_e_scores'])
            overall_clip_i.extend(metrics['clip_i_scores'])
            overall_clip_t.extend(metrics['clip_t_scores'])
            overall_aesclip_i.extend(metrics['aesclip_i_scores'])
            overall_aesclip_t.extend(metrics['aesclip_t_scores'])
            overall_niqe.extend(metrics['niqe_scores'])
            overall_nima.extend(metrics['nima_scores'])
            overall_musiq.extend(metrics['musiq_scores'])
            overall_qalign.extend(metrics['qalign_scores'])
            
            print(f"\nSource: {source}")
            print("-" * 40)
            print(f"Processed images: {metrics['processed_count']}")
            print(f"Skipped images: {metrics['skipped_count']}")
            print(f"Average PSNR: {avg_psnr:.6f} dB")
            print(f"Average SSIM: {avg_ssim:.6f}")
            print(f"Average LPIPS: {avg_lpips:.6f}")
            print(f"Average Delta E: {avg_delta_e:.6f}")
            print(f"Average CLIP-T: {avg_clip_t:.6f}")
            print(f"Average CLIP-I: {avg_clip_i:.6f}")
            print(f"Average AesCLIP-T: {avg_aesclip_t:.6f}")
            print(f"Average AesCLIP-I: {avg_aesclip_i:.6f}")
            print(f"Average NIQE: {avg_niqe:.6f}")
            print(f"Average NIMA: {avg_nima:.6f}")
            print(f"Average MUSIQ: {avg_musiq:.6f}")
            print(f"Average Q-Align: {avg_qalign:.6f}")
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
        print(f"Overall Average Delta E: {np.mean(overall_delta_e):.6f}")
        if overall_clip_t:
            print(f"Overall Average CLIP-T: {np.mean(overall_clip_t):.6f}")
        if overall_clip_i:
            print(f"Overall Average CLIP-I: {np.mean(overall_clip_i):.6f}")
        if overall_aesclip_t:
            print(f"Overall Average AesCLIP-T: {np.mean(overall_aesclip_t):.6f}")
        if overall_aesclip_i:
            print(f"Overall Average AesCLIP-I: {np.mean(overall_aesclip_i):.6f}")
        if overall_niqe:
            print(f"Overall Average NIQE: {np.mean(overall_niqe):.6f}")
        if overall_nima:
            print(f"Overall Average NIMA: {np.mean(overall_nima):.6f}")
        if overall_musiq:
            print(f"Overall Average MUSIQ: {np.mean(overall_musiq):.6f}")
        if overall_qalign:
            print(f"Overall Average Q-Align: {np.mean(overall_qalign):.6f}")
        print("="*80)
    else:
        print("\nNo images were successfully processed!")

