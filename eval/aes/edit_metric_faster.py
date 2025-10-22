import json
import numpy as np
import math
import cv2
import lpips
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from skimage import color
import clip
import sys
import concurrent.futures
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IQA-PyTorch'))
import pyiqa
import imagesize

# pip install git+https://github.com/openai/CLIP.git
# pip install lpips scikit-image==0.24 imagesize
# pip install timm icecream transformers==4.37.2 # for pyiqa
# https://drive.google.com/drive/folders/1kSjpyfBGL0k4bs2lkyL9HFVzcLZWGdKY to models/AesCLIP_weight/AesCLIP

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


def load_image(image_path, short_size=None):
    """Load image and convert to numpy array in range [0, 255]"""
    img = Image.open(image_path).convert('RGB')
    if short_size:
        img = transforms.Resize(short_size, interpolation=transforms.InterpolationMode.BICUBIC)(img)
    return img, np.array(img)


def tensor_from_image(img):
    """Convert numpy image to tensor for LPIPS calculation"""
    # Convert from [0, 255] to [-1, 1] range as expected by LPIPS
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    return img_tensor * 2 - 1

class EditDataset(Dataset):
    def __init__(self, json_path, data_dir, preprocess):
        self.json_path = json_path
        self.data_dir = data_dir
        self.data = self._load_data()
        self.preprocess = preprocess
    
    def _load_data(self):
        data = []
        with open(self.json_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                raw_path = os.path.join(self.data_dir, item["raw"])
                target_path = os.path.join(self.data_dir, item["target"])
                if not os.path.exists(raw_path) or not os.path.exists(target_path):
                    continue
                raw_size = imagesize.get(raw_path)
                target_size = imagesize.get(target_path)
                if abs(raw_size[0]/raw_size[1] - target_size[0]/target_size[1]) > 0.01:
                    continue
                data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        item = self.preprocess(item)
        return item

class EditMetric(object):
    def __init__(self, metric_name = []):
        self.metric_name = metric_name
        self.metric_dict = {} # key: source, value: dict of metrics
    
    def update_metric_bach(self, sources, scores, metric_name):
        for source, score in zip(sources, scores):
            if source not in self.metric_dict:
                self.metric_dict[source] = {}
            if metric_name not in self.metric_dict[source]:
                self.metric_dict[source][metric_name] = []
            self.metric_dict[source][metric_name].append(score)
    
    def print_metric(self):
        overall_metrics = {}
        # Average metric by source
        for source, metrics in self.metric_dict.items():
            print(f"Source: {source}")
            print("-" * 40)
            for metric_name, metric_values in metrics.items():
                if metric_name not in overall_metrics:
                    overall_metrics[metric_name] = []
                overall_metrics[metric_name].extend(metric_values)
                print(f"{metric_name}: {np.mean(metric_values):.6f} ({len(metric_values)} samples)")
            print("-" * 40)
        # Overall average metrics
        print("\nOverall average metrics:")
        for metric_name, metric_values in overall_metrics.items():
            print(f"{metric_name}: {np.mean(metric_values):.6f} ({len(metric_values)} samples)")
            print("-" * 40)

def collate_fn(batch):
    """Custom collate function to handle images of different sizes"""
    # Extract lists instead of stacking tensors
    sources = [item['source'] for item in batch]
    psnr = [item['psnr'] for item in batch]
    ssim = [item['ssim'] for item in batch]
    delta_e = [item['delta_e'] for item in batch]
    target_tensors = [item['target_tensor'] for item in batch]
    result_tensors = [item['result_tensor'] for item in batch]
    target_clips = [item['target_clip'] for item in batch]
    result_clips = [item['result_clip'] for item in batch]
    
    ret = {
        'source': sources,
        'psnr': psnr,
        'ssim': ssim,
        'delta_e': delta_e,
        'target_tensor': target_tensors,
        'result_tensor': result_tensors,
        'target_clip': target_clips,
        'result_clip': result_clips,
    }
    
    # Handle optional text_token field
    if 'text_token' in batch[0]:
        text_tokens = [item['text_token'] for item in batch]
        ret['text_token'] = text_tokens
    
    return ret

def main_task(batch, edit_metric):
    edit_metric.update_metric_bach(batch["source"], batch["psnr"], "psnr")
    edit_metric.update_metric_bach(batch["source"], batch["ssim"], "ssim")
    edit_metric.update_metric_bach(batch["source"], batch["delta_e"], "delta_e")
    
    # Process each sample individually due to different sizes
    batch_size = len(batch["source"])
    lpips_scores = []
    clip_i_scores = []
    aesclip_i_scores = []
    clip_t_scores = []
    aesclip_t_scores = []
    niqe_scores = []
    nima_scores = []
    musiq_scores = []
    qalign_scores = []
    
    with torch.no_grad():
        for i in range(batch_size):
            # Add batch dimension and move to device
            target_tensor = batch["target_tensor"][i].unsqueeze(0).to(device)
            result_tensor = batch["result_tensor"][i].unsqueeze(0).to(device)
            target_clip = batch["target_clip"][i].unsqueeze(0).to(device)
            result_clip = batch["result_clip"][i].unsqueeze(0).to(device)
            
            # LPIPS
            lpips_score = lpips_fn(target_tensor, result_tensor).item()
            lpips_scores.append(lpips_score)
            
            # CLIP-I
            target_feature = clip_model.encode_image(target_clip)
            result_feature = clip_model.encode_image(result_clip)
            target_feature = F.normalize(target_feature, p=2, dim=1)
            result_feature = F.normalize(result_feature, p=2, dim=1)
            clip_i_score = (target_feature * result_feature).sum(dim=1).item()
            clip_i_scores.append(clip_i_score)
            
            # AesCLIP-I
            target_aesclip_feature = aesclip_model.encode_image(target_clip)
            result_aesclip_feature = aesclip_model.encode_image(result_clip)
            target_aesclip_feature = F.normalize(target_aesclip_feature, p=2, dim=1)
            result_aesclip_feature = F.normalize(result_aesclip_feature, p=2, dim=1)
            aesclip_i_score = (target_aesclip_feature * result_aesclip_feature).sum(dim=1).item()
            aesclip_i_scores.append(aesclip_i_score)
            
            # CLIP-T and AesCLIP-T if instruction exists
            if 'text_token' in batch:
                text_token = batch['text_token'][i].unsqueeze(0).to(device)
                text_feature = clip_model.encode_text(text_token)
                text_feature = F.normalize(text_feature, p=2, dim=1)
                clip_t_score = (text_feature * result_feature).sum(dim=1).item()
                clip_t_scores.append(clip_t_score)
                
                text_aesclip_feature = aesclip_model.encode_text(text_token)
                text_aesclip_feature = F.normalize(text_aesclip_feature, p=2, dim=1)
                aesclip_t_score = (text_aesclip_feature * result_aesclip_feature).sum(dim=1).item()
                aesclip_t_scores.append(aesclip_t_score)
            
            # pyiqa metrics
            result_tensor_pyiqa = (result_tensor + 1) / 2
            niqe_score = niqe_metric(result_tensor_pyiqa).item()
            niqe_scores.append(niqe_score)
            nima_score = nima_metric(result_tensor_pyiqa).item()
            nima_scores.append(nima_score)
            musiq_score = musiq_metric(result_tensor_pyiqa).item()
            musiq_scores.append(musiq_score)
            qalign_score = qalign_metric(result_tensor_pyiqa, task_='aesthetic').item()
            qalign_scores.append(qalign_score)
        
        # Update metrics
        edit_metric.update_metric_bach(batch["source"], lpips_scores, "lpips")
        edit_metric.update_metric_bach(batch["source"], clip_i_scores, "clip_i")
        edit_metric.update_metric_bach(batch["source"], aesclip_i_scores, "aesclip_i")
        if clip_t_scores:
            edit_metric.update_metric_bach(batch["source"], clip_t_scores, "clip_t")
        if aesclip_t_scores:
            edit_metric.update_metric_bach(batch["source"], aesclip_t_scores, "aesclip_t")
        edit_metric.update_metric_bach(batch["source"], niqe_scores, "niqe")
        edit_metric.update_metric_bach(batch["source"], nima_scores, "nima")
        edit_metric.update_metric_bach(batch["source"], musiq_scores, "musiq")
        edit_metric.update_metric_bach(batch["source"], qalign_scores, "qalign") 


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
    aesclip_model, _ = clip.load("ViT-B/16", device=device)
    aesclip_model.load_state_dict(torch.load("models/AesCLIP_weight/AesCLIP", map_location=device))
    # Initialize pyiqa metrics
    print("Loading pyiqa metrics...")
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    nima_metric = pyiqa.create_metric('nima', device=device)
    musiq_metric = pyiqa.create_metric('musiq-ava', device=device)
    qalign_metric = pyiqa.create_metric('qalign', device=device)
    
    # Preprocess function
    def preprocess(item):
        image_name = item["target"]
        source = item.get("source", "unknown")  # Get source field, default to "unknown"
        instruction = item.get("instruction", "")  # Get instruction for CLIP-T
        target_path = os.path.join(data_dir, image_name)
        # result_path = os.path.join(result_dir, image_name)
        result_path = os.path.join(data_dir, item["raw"])
        target_pil, target_img = load_image(target_path, 512)
        result_pil, result_img = load_image(result_path, 512)
        if target_img.shape != result_img.shape:
            # resize target to result
            target_img = cv2.resize(target_img, (result_img.shape[1], result_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        
        # precompute PSNR, SSIM, Delta E
        psnr = calculate_psnr(target_img, result_img)
        if psnr is None or psnr == float('inf') or psnr == float('-inf'):
            psnr = 0
            print(f"Warning: psnr is None or inf for source: {source}, target_path: {target_path}, result_path: {result_path}")
        ssim = calculate_ssim(target_img, result_img)
        delta_e = calculate_delta_e(target_img, result_img)
        
        target_tensor = tensor_from_image(target_img)
        result_tensor = tensor_from_image(result_img)
        target_clip = clip_preprocess(target_pil)
        result_clip = clip_preprocess(result_pil)

        ret = dict(
            source=source,
            # target_img = target_img,
            # result_img = result_img,
            # instruction = instruction,
            target_tensor = target_tensor,
            result_tensor = result_tensor,
            target_clip = target_clip,
            result_clip = result_clip,
            psnr = psnr,
            ssim = ssim,
            delta_e = delta_e,
        )
        if instruction:
            text_token = clip.tokenize(instruction).squeeze(0)
            ret["text_token"] = text_token

        return ret
    
    edit_dataset = EditDataset(bench_json, data_dir, preprocess)
    edit_dataloader = DataLoader(edit_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn)

    # Storage for metrics grouped by source
    edit_metric = EditMetric()
    for batch in tqdm(edit_dataloader, desc="Processing"):
        main_task(batch, edit_metric)
    
    edit_metric.print_metric()