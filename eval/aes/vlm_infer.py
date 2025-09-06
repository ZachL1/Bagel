#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Inference script for EAPD_release benchmark
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
from tqdm import tqdm

# Import shared utilities
from utils import set_seed, create_inferencer, DEFAULT_VLM_INFERENCE_PARAMS


def load_evaluation_data(data_path: str) -> Dict[str, Any]:
    """Load EAPD evaluation data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    return eval_data


def process_aes_question(image_path: str, question_data: Dict[str, Any], task_type: str, inferencer) -> str:
    """Process a single aesthetic question"""
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return ""
    
    question = question_data.get("Question", "")
    options = question_data.get("Options", "")
    
    if task_type in ["AesP", "AesE", "AesA1"]:
        # Multiple choice questions
        if options:
            prompt = f"{question} Choose one from the following options:\n{options}\n\nAnswer with the option's letter from the given choices directly."
        else:
            prompt = question
    else:
        # AesI - open-ended explanation
        prompt = question
    
    # Inference hyperparameters for understanding tasks
    inference_hyper = DEFAULT_VLM_INFERENCE_PARAMS.copy()
    
    output_dict = inferencer(image=image, text=prompt, understanding_output=True, **inference_hyper)
    response = output_dict.get('text', '')
    
    return response


def run_inference(args):
    """Main inference loop"""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load model and create inferencer
    print("Loading model...")
    inferencer = create_inferencer(args.model_path, args.llm_path, args.max_mem_per_gpu, args.use_ema)
    
    # Load evaluation data
    print("Loading evaluation data...")
    eval_data = load_evaluation_data(args.eval_data_path)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each task type
    task_types = ["AesP", "AesE", "AesA1", "AesI"]
    
    for task_type in task_types:
        print(f"Processing {task_type} task...")
        results = {}
        
        for image_filename, image_data in tqdm(eval_data.items()):
            task_data_key = f"{task_type}_data"
            if task_data_key not in image_data:
                continue
                
            image_path = os.path.join(args.image_dir, image_filename)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            question_data = image_data[task_data_key]
            response = process_aes_question(image_path, question_data, task_type, inferencer)
            
            results[image_filename] = {
                "question": question_data.get("Question", ""),
                "options": question_data.get("Options", ""),
                f"{task_type}_response": response
            }
            print(results[image_filename])
        
        # Save results
        output_file = output_dir / f"{task_type}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(results)} results to {output_file}")
    
    print(f"All results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="BAGEL VLM Inference for EAPD_release benchmark")
    parser.add_argument("--model_path", type=str, default="./models/BAGEL-7B-MoT",
                        help="Path to BAGEL model")
    parser.add_argument("--llm_path", type=str, default="./models/BAGEL-7B-MoT",
                        help="Path to LLM checkpoint")
    parser.add_argument("--eval_data_path", type=str, 
                        default="data/sft_data/EAPD_release/AesBench_evaluation.json",
                        help="Path to evaluation data JSON file")
    parser.add_argument("--image_dir", type=str, 
                        default="data/sft_data/EAPD_release/images",
                        help="Directory containing evaluation images")
    parser.add_argument("--output_dir", type=str, default="results/aes_eval",
                        help="Output directory for results")
    parser.add_argument("--tag", type=str, default="vlm_inference",
                        help="Tag for this evaluation run")
    parser.add_argument("--max_mem_per_gpu", type=str, default="40GiB",
                        help="Maximum memory per GPU")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--use_ema", type=bool, default=True,
                        help="Use EMA weights")
    
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()