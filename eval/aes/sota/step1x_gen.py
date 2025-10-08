import torch
import os
import sys
from tqdm import tqdm
from diffusers import Step1XEditPipelineV1P2
from diffusers.utils import load_image
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from custom_dataset import ImageEditDataset, collate_fn

#set_path
json_path = "data/sft_data/AesEditor/data_json/aes_edit_test.jsonl"
data_path = "data/sft_data/AesEditor"
output_path = "results/aes_eval/aes_edit_step/edited_images"
max_samples_per_source = 10  # 每个source最多生成10个样本


#load_data
dataset=ImageEditDataset(json_path,data_path,output_path)
dataloader=DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn
)
#load pipeline
pipe = Step1XEditPipelineV1P2.from_pretrained("./models/Step1X-Edit-v1p2-preview", 
                                                torch_dtype=torch.bfloat16,
                                                device_map='auto')

#run_image_edit_inference
with tqdm(total=len(dataset),desc="Processing images",unit="img") as pbar:
    for batch in dataloader:
        prompts = batch['prompt']
        images = batch['image']
        target_paths = batch['target_paths']
        
        for i in range(len(images)):
                target_path = target_paths[i]
                output_file_path = os.path.join(output_path, target_path)
                
                if os.path.exists(output_file_path):
                    print(f"图像已存在，跳过: {target_path}")
                    pbar.update(1)
                    pbar.set_postfix({"Skipped": target_path})
                    continue
                
                enable_thinking_mode=True
                enable_reflection_mode=True
                pipe_output = pipe(
                    image=images[i],
                    prompt=prompts[i],
                    num_inference_steps=28,
                    true_cfg_scale=4,
                    generator=torch.Generator().manual_seed(42),
                    enable_thinking_mode=enable_thinking_mode,
                    enable_reflection_mode=enable_reflection_mode,
                    )

                output_dir = os.path.dirname(output_file_path)
                os.makedirs(output_dir, exist_ok=True)
                pipe_output.images[0].save(output_file_path)
                
                pbar.update(1)
                pbar.set_postfix({"Saved": target_path})

'''
image = load_image("examples/0000.jpg").convert("RGB")
prompt = "add a ruby ​​pendant on the girl's neck."
enable_thinking_mode=True
enable_reflection_mode=True
pipe_output = pipe(
    image=image,
    prompt=prompt,
    num_inference_steps=28,
    true_cfg_scale=4,
    generator=torch.Generator().manual_seed(42),
    enable_thinking_mode=enable_thinking_mode,
    enable_reflection_mode=enable_reflection_mode,
)
'''