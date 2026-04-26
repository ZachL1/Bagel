import json
import os
import random
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

bench_json = "data/sft_data/Pexels_test/all_test_c.jsonl"
data_dir = "data/sft_data/Pexels_test/"
result_dirs = {
    "instructp2p": "results/pexels_eval/instructp2p_new",
    "ultraedit": "results/pexels_eval/ultraedit_new",
    "step1x": "results/pexels_eval/aes_edit_step1x/edited_images",
    "hidream": "results/pexels_eval/aes_edit_hidream/edited_images",
    "icedit": "results/pexels_eval/icedit_new",
    "flux": "results/pexels_eval/aes_edit_flux/edited_images",
    "qwen": "results/pexels_eval/aes_edit_qwen/edited_images",
    "Bagel": "results/pexels_eval/aes_edit_bagel/edited_images",
    "Our": "results/pexels_eval/sft_all_14/edited_images",
}

show_count = 500
save_dir = "results/pexels_eval/show_demo"
os.makedirs(save_dir, exist_ok=True)

if __name__ == "__main__":
    with open(bench_json, 'r') as f:
        bench_data = f.readlines()

    random.shuffle(bench_data)
    for line in bench_data:
        data = json.loads(line)

        use = True
        for tag, result_dir in result_dirs.items():
            result_path = os.path.join(result_dir, data['target'])
            raw_path = os.path.join(result_dir, data['raw'])
            if not os.path.exists(result_path) and not os.path.exists(raw_path):
                use = False
                break
        if not use:
            continue
        
        image_path = os.path.join(data_dir, data['raw'])
        raw_image = Image.open(image_path)
        image_path = os.path.join(data_dir, data['target'])
        target_image = Image.open(image_path)
        instruction = data['instruction'] + f" (source: {data['source']})"
        sample_type = data['type'].replace(" ", "_") + '_' + data['source'].replace(" ", "_")
        
        result_images = {}
        for tag, result_dir in result_dirs.items():
            result_path = os.path.join(result_dir, data['target'])
            if data['source'] == "Pexels.com" or result_dir.endswith("_new"):
                result_path = os.path.join(result_dir, data['raw'])
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
        # save_path = os.path.join(save_dir, f"{sample_type}_{show_count:03d}.png")
        save_path = os.path.join(save_dir, data['target'] if data['source'] != "Pexels.com" else data['raw'])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        show_count -= 1
        print(f"Saved {save_path}")
        
        if show_count == 0:
            break