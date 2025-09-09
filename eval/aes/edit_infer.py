#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Inference script for AesEditor editing data
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
from tqdm import tqdm
import random
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

# Import shared utilities
from utils import set_seed, create_inferencer, DEFAULT_EDIT_INFERENCE_PARAMS


class DualThreadImageSaver:
    """
    双线程图片保存器，使用两个工作线程交替保存图片以提高性能
    """
    def __init__(self, max_workers=2):
        self.save_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ImageSaver")
        self.active_futures = []
        self.shutdown = False
        
    def save_image_async(self, image, save_path):
        """异步保存图片到指定路径"""
        if self.shutdown:
            return
            
        # 提交保存任务到线程池
        future = self.executor.submit(self._save_image_worker, image, save_path)
        self.active_futures.append(future)
        
        # 清理已完成的任务
        self.active_futures = [f for f in self.active_futures if not f.done()]
        
    def _save_image_worker(self, image, save_path):
        """工作线程中的图片保存函数"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 保存图片
            image.save(save_path)
            return True
        except Exception as e:
            print(f"Error saving image to {save_path}: {e}")
            return False
    
    def wait_for_completion(self):
        """等待所有保存任务完成"""
        for future in as_completed(self.active_futures):
            try:
                future.result()  # 获取结果，如果有异常会抛出
            except Exception as e:
                print(f"Image save task failed: {e}")
        self.active_futures.clear()
    
    def shutdown_saver(self):
        """关闭保存器并等待所有任务完成"""
        self.shutdown = True
        self.wait_for_completion()
        self.executor.shutdown(wait=True)


def load_edit_data(data_path: str, data_split: str) -> List[Dict[str, Any]]:
    """Load AesEditor editing data from JSONL file"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    random.seed(42)
    random.shuffle(data)

    data_count = {}
    used_data = []
    for item in data:
        if data_count.get(item["source"], 0) < 1000:
            data_count[item["source"]] = data_count.get(item["source"], 0) + 1
            used_data.append(item)
    
    # split the data into n parts and load the m-th part
    n, m = map(int, data_split.split("-"))
    begin = int(m * len(used_data) / n)
    end = int((m + 1) * len(used_data) / n)
    print(f"Loading {len(used_data)} editing requests from {begin} to {end}")
    used_data = used_data[begin:end]

    return used_data


class EditDataset(Dataset):
    def __init__(self, data_path: str, data_split: str, image_output_dir: str, base_image_dir: str):
        self.data = load_edit_data(data_path, data_split)
        self.image_output_dir = image_output_dir
        self.base_image_dir = base_image_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        item["output_image"] = os.path.join(self.image_output_dir, item["target"].rsplit(".", 1)[0] + ".png")
        os.makedirs(os.path.dirname(item["output_image"]), exist_ok=True)
        
        item["raw"] = os.path.join(self.base_image_dir, item["raw"])
        item["target"] = os.path.join(self.base_image_dir, item["target"])
        raw_image = Image.open(item["raw"]).convert('RGB')
        item["raw_image"] = raw_image

        return item

def process_edit_request(item: Dict[str, Any], inferencer, image_saver=None) -> Dict[str, Any]:
    """Process a single image editing request"""
    image_path = item.get("raw", "")
    instruction = item.get("instruction", "")
    instructions = item.get("instructions", "")
    image = item.pop("raw_image")
    
    # Use the instruction for editing
    edit_prompt = instruction if instruction else instructions
    
    # Inference hyperparameters for editing
    inference_hyper = DEFAULT_EDIT_INFERENCE_PARAMS.copy()
    
    # Perform editing
    output_dict = inferencer(image=image, text=edit_prompt, think=False, **inference_hyper)
    
    item.update({
        "generated_text": output_dict.get('text', ''),
        "output_image_generated": output_dict.get('image') is not None
    })
    
    # Save the output image if generated
    if output_dict.get('image') is not None:
        if image_saver is not None:
            # 使用异步保存器保存图片
            image_saver.save_image_async(output_dict['image'], item["output_image"])
        else:
            # 回退到同步保存
            output_dict['image'].save(item["output_image"])
    
    return item


def run_inference(args):
    set_seed(args.seed)
    
    print("Loading model...")
    inferencer = create_inferencer(args.model_path, args.llm_path, args.max_mem_per_gpu)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load editing data
    print("Loading editing data...")
    image_output_dir = output_dir / "edited_images"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    edit_dataset = EditDataset(args.edit_data_path, args.data_split, image_output_dir, args.base_image_dir)
    edit_data = DataLoader(edit_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x[0], pin_memory=True)
    
    # Process editing requests
    print("Processing editing requests...")
    results = []
    image_saver = DualThreadImageSaver(max_workers=2)
    
    for i, item in enumerate(tqdm(edit_data)):
        if args.max_samples > 0 and i >= args.max_samples:
            break
            
        try:
            if os.path.exists(item["output_image"]):
                item.pop("raw_image")
                result = item
            else:
                result = process_edit_request(item, inferencer, image_saver)
            print(result)
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            result.update({
                "error": str(e),
                "output_image": None
            })
        results.append(result)
    
    # Save results
    with open(os.path.join(output_dir, f"edit_results_{args.data_split}.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Processing completed. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="BAGEL Image Editing Inference for AesEditor data")
    parser.add_argument("--model_path", type=str, default="./models/BAGEL-7B-MoT",
                        help="Path to BAGEL model")
    parser.add_argument("--llm_path", type=str, default="./models/BAGEL-7B-MoT",
                        help="Path to LLM checkpoint")
    parser.add_argument("--edit_data_path", type=str, 
                        default="data/sft_data/AesEditor/data_json/aes_edit_test.jsonl",
                        help="Path to editing data JSONL file")
    parser.add_argument("--base_image_dir", type=str, 
                        default="data/sft_data/AesEditor/data_json",
                        help="Base directory for finding images")
    parser.add_argument("--output_dir", type=str, default="results/aes_eval",
                        help="Output directory for results")
    parser.add_argument("--tag", type=str, default="edit_inference",
                        help="Tag for this evaluation run")
    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB",
                        help="Maximum memory per GPU")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum number of samples to process (-1 for all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use_ema", type=bool, default=True,
                        help="Use EMA weights")
    parser.add_argument("--data_split", type=str, default="4-0", 
                        help="m-n means the data is split into m parts and inference the n-th part")
    
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()