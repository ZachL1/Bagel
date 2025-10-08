import torch
import os
import sys
import argparse
from tqdm import tqdm
from diffusers import FluxKontextPipeline
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from eval.aes.sota.custom_dataset import ImageEditDataset, collate_fn

#set_path
# json_path = "data/sft_data/AesEditor/data_json/aes_edit_test.jsonl"
# data_path = "data/sft_data/AesEditor"
# output_path = "results/aes_eval/aes_edit_flux/edited_images"
# max_samples = 10

# get from args
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default="data/sft_data/AesEditor/data_json/aes_edit_test.jsonl")
parser.add_argument("--data_path", type=str, default="data/sft_data/AesEditor")
parser.add_argument("--output_path", type=str, default="results/aes_eval/aes_edit_flux/edited_images")
parser.add_argument("--max_samples", type=int, default=10)
parser.add_argument("--data_split", type=str, default="1-0")
args = parser.parse_args()

#load_data
max_samples_per_source = None
dataset = ImageEditDataset(args.json_path, args.data_path, args.output_path, max_samples_per_source=args.max_samples, data_split=args.data_split)
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn
)

#load pipeline
pipe = FluxKontextPipeline.from_pretrained("./models/FLUX.1-Kontext-dev", 
                                            torch_dtype=torch.bfloat16,
                                            revision='bf16',
                                            device_map="cuda")

#run_image_edit_inference
with tqdm(total=len(dataset), desc="Processing images", unit="img") as pbar:
    for batch in dataloader:
        prompts = batch['prompt']
        images = batch['image']
        target_paths = batch['target_paths']
        widths = batch['widths']
        heights = batch['heights']
        
        for i in range(len(images)):

            target_path = target_paths[i]
            output_file_path = os.path.join(args.output_path, target_path)
            
            if os.path.exists(output_file_path):
                pbar.update(1)
                pbar.set_postfix({"Skipped": target_path})
                continue
            
            result = pipe(
                image=images[i],
                width=widths[i],
                height=heights[i],
                prompt=prompts[i],
                guidance_scale=2.5,
                generator=torch.Generator("cuda").manual_seed(42)
            ).images[0]
            
            output_dir = os.path.dirname(output_file_path)
            os.makedirs(output_dir, exist_ok=True)
            result.save(output_file_path)
            
            pbar.update(1)
            pbar.set_postfix({"Saved": target_path})