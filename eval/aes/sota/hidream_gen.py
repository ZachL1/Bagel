import torch
import os
import sys
import argparse
import math
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

from pipeline_hidream_image_editing import HiDreamImageEditingPipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from eval.aes.sota.custom_dataset import ImageEditDataset, collate_fn

def resize_image(pil_image, image_size=1024):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    m = 16
    width, height = pil_image.width, pil_image.height
    S_max = image_size * image_size
    scale = S_max / (width * height)
    scale = math.sqrt(scale)

    new_sizes = [
        (round(width * scale) // m * m, round(height * scale) // m * m),
        (round(width * scale) // m * m, math.floor(height * scale) // m * m),
        (math.floor(width * scale) // m * m, round(height * scale) // m * m),
        (math.floor(width * scale) // m * m, math.floor(height * scale) // m * m),
    ]
    new_sizes = sorted(new_sizes, key=lambda x: x[0] * x[1], reverse=True)

    for new_size in new_sizes:
        if new_size[0] * new_size[1] <= S_max:
            break

    s1 = width / new_size[0]
    s2 = height / new_size[1]
    if s1 < s2:
        pil_image = pil_image.resize([new_size[0], round(height / s1)], resample=Image.BICUBIC)
        top = (round(height / s1) - new_size[1]) // 2
        pil_image = pil_image.crop((0, top, new_size[0], top + new_size[1]))
    else:
        pil_image = pil_image.resize([round(width / s2), new_size[1]], resample=Image.BICUBIC)
        left = (round(width / s2) - new_size[0]) // 2
        pil_image = pil_image.crop((left, 0, left + new_size[0], new_size[1]))

    return pil_image

# #set_path
# json_path = "/root/autodl-tmp/test_unit_json_subset/aes_edit_test.subset.jsonl"
# data_path = "/root/autodl-tmp/test_unit_json_subset"
# output_path = "/root/autodl-tmp/output"

# get from args
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default="data/sft_data/AesEditor/data_json/aes_edit_test.jsonl")
parser.add_argument("--data_path", type=str, default="data/sft_data/AesEditor")
parser.add_argument("--output_path", type=str, default="results/aes_eval/aes_edit_hidream/edited_images")
parser.add_argument("--max_samples", type=int, default=10)
parser.add_argument("--data_split", type=str, default="1-0")
args = parser.parse_args()

#load_data
dataset=ImageEditDataset(args.json_path,args.data_path,args.output_path,max_samples_per_source=args.max_samples,data_split=args.data_split)
dataloader=DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=8,
    collate_fn=collate_fn
)


#load pipeline
tokenizer_4 = PreTrainedTokenizerFast.from_pretrained("./models/Llama-3.1-8B-Instruct")
text_encoder_4 = LlamaForCausalLM.from_pretrained(
            "./models/Llama-3.1-8B-Instruct",
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.bfloat16)
text_encoder_4.to('cuda')
pipe = HiDreamImageEditingPipeline.from_pretrained(
             "./models/HiDream-E1-1",
             tokenizer_4=tokenizer_4,
             text_encoder_4=text_encoder_4,
             torch_dtype=torch.bfloat16
             ).to('cuda')
pipe.transformer.max_seq = 8192
pipe.enable_model_cpu_offload()
#run_image_edit_inference
with tqdm(total=len(dataset),desc="Processing images",unit="img") as pbar:
    for batch in dataloader:
        prompts = batch['prompt']
        images = batch['image']
        target_paths = batch['target_paths']
        
        for i in range(len(images)):
                img = images[i].convert("RGB")
                img = resize_image(img)
                img_w, img_h = img.size

                pipe_output=pipe(
                                prompt=prompts[i],
                                image=img,
                                width=img_w,
                                height=img_h,
                                guidance_scale=3.0,
                                image_guidance_scale=1.5,
                                num_inference_steps=28,
                                refine_strength=0.3,
                                clip_cfg_norm=True,
                                generator=torch.Generator("cuda").manual_seed(42))

                target_path = target_paths[i]
                output_file_path = os.path.join(args.output_path, target_path)
                output_dir = os.path.dirname(output_file_path)
                os.makedirs(output_dir, exist_ok=True)
                pipe_output.images[0].save(output_file_path)
                
                pbar.update(1)
                pbar.set_postfix({"Saved": target_path})