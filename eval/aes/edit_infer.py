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

# Import shared utilities
from utils import set_seed, create_inferencer, DEFAULT_EDIT_INFERENCE_PARAMS


def load_edit_data(data_path: str) -> List[Dict[str, Any]]:
    """Load AesEditor editing data from JSONL file"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def process_edit_request(item: Dict[str, Any], base_image_dir: str, inferencer) -> Dict[str, Any]:
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
        # save the image to the output directory
        output_dict['image'].save(item["output_image"])
    
    return item


def run_inference(args):
    """Main inference loop"""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load model and create inferencer
    print("Loading model...")
    inferencer = create_inferencer(args.model_path, args.llm_path, args.max_mem_per_gpu)
    
    # Load editing data
    print("Loading editing data...")
    edit_data = load_edit_data(args.edit_data_path)
    print(f"Loaded {len(edit_data)} editing requests")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir = output_dir / "edited_images"
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process editing requests
    print("Processing editing requests...")
    results = []
    
    for i, item in enumerate(tqdm(edit_data)):
        if args.max_samples > 0 and i >= args.max_samples:
            break
            
        try:
            item["raw"] = os.path.join(args.base_image_dir, item["raw"])
            item["target"] = os.path.join(args.base_image_dir, item["target"])
            item["output_image"] = os.path.join(image_output_dir, f"edit_{i:06d}.png")
            result = process_edit_request(item, args.base_image_dir, inferencer)
            print(result)
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            result.update({
                "error": str(e),
                "output_image": None
            })
        results.append(result)
    
    # Save results
    with open(os.path.join(output_dir, f"edit_results.json"), 'w', encoding='utf-8') as f:
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
    
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()