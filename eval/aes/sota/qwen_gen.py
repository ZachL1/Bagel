import os
import sys
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from diffusers import QwenImageEditPipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from eval.aes.sota.custom_dataset import ImageEditDataset, collate_fn

# pip install diffusers
# pip install protobuf==3.20.0
# /opt/conda/lib/python3.11/site-packages/transformers/generation/configuration_utils.py:1288
# decoder_config_dict = decoder_config.to_dict() # change to :
# decoder_config_dict = decoder_config.to_dict() if isinstance(decoder_config, PretrainedConfig) else decoder_config

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

def inference(pipe, batch, output_path):
    images = batch['image']
    prompts = batch['prompt']
    target_paths = batch['target_paths']
    widths = batch['widths']
    heights = batch['heights']
    
    results = []
    valid_indices = []
    
    # Check which images need to be processed (those that don't exist)
    for i, target_path in enumerate(target_paths):
        output_file_path = os.path.join(output_path, target_path)
        if not os.path.exists(output_file_path):
            valid_indices.append(i)
        else:
            print(f"Output file already exists, skipping: {output_file_path}")
    
    # Only process the images that need to be generated
    if valid_indices:
        for i in valid_indices:
            image = images[i]
            prompt = prompts[i]
            width = widths[i]
            height = heights[i]
            
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=" ",
                generator=torch.manual_seed(0),
                true_cfg_scale=4.0,
                num_inference_steps=50
            ).images[0]
            results.append((result, target_paths[i]))
        
        # Save the results
        for image, target_path in results:
            output_file_path = os.path.join(output_path, target_path)
            output_dir = os.path.dirname(output_file_path)
            os.makedirs(output_dir, exist_ok=True)
            image.save(output_file_path)

dataset = ImageEditDataset(args.json_path, args.data_path, args.output_path, max_samples_per_source=args.max_samples, data_split=args.data_split)
dataloader = DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=False, 
    collate_fn=collate_fn,
    num_workers=8  
)


pipe = QwenImageEditPipeline.from_pretrained("./models/Qwen-Image-Edit", 
                                            torch_dtype=torch.bfloat16,
                                            revision='bf16',
                                            device_map="cuda")

with tqdm(total=len(dataset), desc="Processing images", unit="img") as pbar:
    for batch_idx, batch in enumerate(dataloader):
        pbar.set_description(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
        inference(pipe, batch, args.output_path)
        pbar.update(len(batch['image']))
        pbar.set_description(f"Completed batch {batch_idx + 1}/{len(dataloader)}")

print("All images processed!")