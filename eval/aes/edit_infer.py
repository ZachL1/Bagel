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

# Import shared utilities
from utils import set_seed, create_inferencer, DEFAULT_EDIT_INFERENCE_PARAMS


def load_edit_data(data_path: str, data_split: str, max_samples: int, base_image_dir: str, image_output_dir: str) -> List[Dict[str, Any]]:
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
        if data_count.get(item["source"], 0) < max_samples:
            data_count[item["source"]] = data_count.get(item["source"], 0) + 1

            item["output_image"] = os.path.join(image_output_dir, item["target"])
            item["raw"] = os.path.join(base_image_dir, item["raw"])
            item["target"] = os.path.join(base_image_dir, item["target"])

            if not os.path.exists(item["output_image"]):
                used_data.append(item)
    
    # split the data into n parts and load the m-th part
    n, m = map(int, data_split.split("-"))
    begin = int(m * len(used_data) / n)
    end = int((m + 1) * len(used_data) / n)
    print(f"Loading {len(used_data)} editing requests from {begin} to {end}")
    used_data = used_data[begin:end]

    return used_data


def process_edit_request(item: Dict[str, Any], base_image_dir: str, inferencer, long_prompt: bool) -> Dict[str, Any]:
    """Process a single image editing request"""
    image_path = item.get("raw", "")
    instruction = item.get("instruction", "")
    instructions = item.get("instructions", "")
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        item.update({
            "error": f"Failed to load image: {e}",
            "output_image": None
        })
        return item
    
    # Use the instruction for editing
    if not long_prompt:
        edit_prompt = instruction if instruction else instructions
    else:
        edit_prompt = instructions if instructions else instruction
    
    # Inference hyperparameters for editing
    inference_hyper = DEFAULT_EDIT_INFERENCE_PARAMS.copy()
    
    # Perform editing
    # print(f"Performing editing for {image_path} with prompt: {edit_prompt}")
    output_dict = inferencer(image=image, text=edit_prompt, think=False, **inference_hyper)
    # print(f"Editing completed for {image_path}")
    
    item.update({
        "generated_text": output_dict.get('text', ''),
        "output_image_generated": output_dict.get('image') is not None
    })
    
    # Save the output image if generated
    if output_dict.get('image') is not None:
        # save the image to the output directory
        os.makedirs(os.path.dirname(item["output_image"]), exist_ok=True)
        output_dict['image'].save(item["output_image"])
    
    return item


def run_inference(args):
    set_seed(args.seed)
    
    print("Loading model...")
    inferencer = create_inferencer(args.model_path, args.llm_path, args.max_mem_per_gpu, visual_und=args.visual_und)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir = output_dir / "edited_images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # Load editing data
    print("Loading editing data...")
    edit_data = load_edit_data(args.edit_data_path, args.data_split, args.max_samples, args.base_image_dir, image_output_dir)
    
    # Process editing requests
    print("Processing editing requests...")
    results = []
    
    for i, item in enumerate(tqdm(edit_data)):
        try:
            if os.path.exists(item["output_image"]):
                result = item
            else:
                result = process_edit_request(item, args.base_image_dir, inferencer, args.long_prompt)
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
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples to process (-1 for all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no_ema", action="store_true",
                        help="Use EMA weights")
    parser.add_argument("--data_split", type=str, default="4-0", 
                        help="m-n means the data is split into m parts and inference the n-th part")
    parser.add_argument("--no_visual_und", action="store_true",
                        help="Use visual understanding")
    parser.add_argument("--long_prompt", action="store_true",
                        help="Use long prompt")
    args = parser.parse_args()
    args.use_ema = not args.no_ema
    args.visual_und = not args.no_visual_und
    run_inference(args)


if __name__ == "__main__":
    main()