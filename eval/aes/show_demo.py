import json
import os
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

bench_json = "data/sft_data/AesEditor/data_json/aes_edit_test.jsonl"
data_dir = "data/sft_data/AesEditor/"
result_dirs = {
    "Bagel": "results/aes_eval/aes_edit_bagel/edited_images",
    "flux": "results/aes_eval/flux_subset/data",
    "qwen_image": "results/aes_eval/qwen_subset/aes_edit_data",
    "Bagel_14k": "results/aes_eval/aes_edit_bagel_14/edited_images",
    # "Bagel_28k": "results/aes_eval/aes_edit_bagel_28/edited_images",
    # "Qwen_60k": "results/aes_eval/aes_edit_qwen_60/edited_images",
    "Bagel_r": "results/aes_eval/aes_edit_bagel_random_60k/edited_images",
    "Bagel_r_nov": "results/aes_eval/aes_edit_bagel_random_60k_nov/edited_images",
    "Bagel_r_long": "results/aes_eval/aes_edit_bagel_random_60k_long/edited_images",
}

show_count = 500
save_dir = "results/aes_eval/show_demo"
os.makedirs(save_dir, exist_ok=True)

if __name__ == "__main__":
    with open(bench_json, 'r') as f:
        bench_data = f.readlines()

    random.shuffle(bench_data)
    for line in bench_data:
        data = json.loads(line)
        image_path = os.path.join(data_dir, data['raw'])
        raw_image = Image.open(image_path)
        image_path = os.path.join(data_dir, data['target'])
        target_image = Image.open(image_path)
        instruction = data['instruction'] + f" (source: {data['source']})"
        sample_type = data['type'].replace(" ", "_")
        
        result_images = {}
        for tag, result_dir in result_dirs.items():
            result_path = os.path.join(result_dir, data['target'])
            if not os.path.exists(result_path):
                continue
            result_image = Image.open(result_path)
            result_images[tag] = result_image
        if len(result_images) != len(result_dirs):
            continue
        
        # show the raw image, instruction, and target image in one raw
        # and the result images in one raw
        cols = 3
        rows = 1 + (len(result_images.keys()) + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows*(raw_image.height/raw_image.width)))
        axes[0, 0].imshow(raw_image)
        axes[0, 0].set_title("Raw Image")
        axes[0, 1].imshow(target_image)
        axes[0, 1].set_title("Target Image")
        # axes[0, 1].set_title("Instruction")
        # axes[0, 1].text(0.5, 0.5, instruction, ha='center', va='center')
        for idx, tag in enumerate(result_images.keys()):
            i = idx + 2
            axes[i//cols, i%cols].imshow(result_images[tag])
            axes[i//cols, i%cols].set_title(f"{tag}")
        
        # Hide axes for all subplots
        for i in range(rows):
            for j in range(cols):
                axes[i, j].axis('off')

        fig.suptitle(instruction, wrap=True)
        fig.tight_layout()
        save_path = os.path.join(save_dir, f"{sample_type}_{show_count:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        show_count -= 1
        print(f"Saved {save_path}")
        
        if show_count == 0:
            break