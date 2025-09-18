# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import traceback
from PIL import Image, ImageFile, PngImagePlugin

from .data_utils import pil_img2rgb
from .interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class AesEditorIterableDataset(InterleavedBaseIterableDataset):
    def __init__(
        self, dataset_name, transform, vit_transform, tokenizer, 
        jsonl_path_list, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        AesEditor dataset for image editing tasks.
        
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of base directories containing the images
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            data_dir_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(
        self, 
        jsonl_path_list, 
        data_dir_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, base_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, base_dir) for json_data in raw_data])
        return data_paths

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0
        transform_stride = self.transform.stride
        vit_transform_stride = self.vit_transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, base_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data_item = json.loads(data)
                    
                    # Load raw (input) image
                    raw_image_path = os.path.join(base_dir, data_item['raw'])
                    raw_image = pil_img2rgb(Image.open(raw_image_path))
                    
                    # Load target (edited) image
                    target_image_path = os.path.join(base_dir, data_item['target'])
                    target_image = pil_img2rgb(Image.open(target_image_path))
                    
                    # Get instruction
                    instruction = data_item['instruction']
                    if data_item['type'] == 'enhancement' and data_item['instructions'] and random.random() < 0.5:
                        instruction = data_item['instructions']
                    elif data_item['type'] != 'enhancement':
                        # Randomly select instruction to avoid bias
                        p = random.random()
                        if p > 0.2 and p < 0.8:
                            instruction = random.choice(instructions_set[data_item['type']])
                        elif p <= 0.2:
                            instruction = random.choice(instructions_set["general"])
                    
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error processing item: {e}")
                    continue

                # Add input image, VAE token and VIT token
                data = self._init_data()
                data = self._add_image(
                    data,
                    raw_image,
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                )

                # Add instruction text
                data = self._add_text(data, instruction, need_loss=False)

                # Add target image (VAE transform for generation)
                data = self._add_image(
                    data,
                    target_image,
                    need_loss=True,
                    need_vae=False,
                    need_vit=False,
                )

                # Verify we have loss
                has_loss = [item['loss'] for item in data['sequence_plan']]
                if sum(has_loss) == 0:
                    print(f'No loss defined, skipped.')
                    continue

                data['data_indexes'] = {
                    "data_indexes": row_idx,
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                }

                yield data

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


# {'denoising', 'raindrop removal', 'inpainting', 'deblurring', 'low-light enhancement', 'compression artifacts removal', 'dehazing', 'shadow removal', 'deraining', 'desnowing', 'enhancement'}
instructions_set = {
    "general": [
        "Restore the image to a clean, high‑quality, natural photo by removing visible degradations. Preserve scene content, geometry, and colors without stylization.",
    ],
    "denoising": [
        "Remove image noise while preserving fine textures and edges. Keep colors and exposure natural; avoid over‑smoothing.",
        "Remove image noise while preserving fine textures and edges. Keep colors natural.",
        "Denoise the photo and maintain detail and sharpness. Avoid a waxy look.",
        "Reduce strong noise and retain natural textures and tones.",
        "Suppress sensor noise and keep edges crisp. Do not blur content.",
        "Clean mild noise while preserving structure and color fidelity.",
        "Diminish noise artifacts and keep a realistic, detailed appearance.",
    ],
    "deblurring": [
        "Remove motion/defocus blur to recover sharp, crisp details. Preserve original structures without hallucinating new content.",
        "Deblur the image to recover sharp, crisp details while preserving structures and colors.",
        "Remove motion/defocus blur and restore clear edges. Avoid hallucinating new content.",
        "Reduce blur across the frame and enhance clarity. Keep a natural look without over‑sharpening.",
        "Correct heavy blur to a clean, sharp photo. Maintain geometry and textures.",
        "Undo camera shake and restore fine detail. Keep tones and composition unchanged.",
        "Make the photo sharp and clear. Avoid artifacts and preserve original content.",
    ],
    "dehazing": [
        "Remove haze/fog and restore clear visibility and contrast. Neutralize color cast and keep the scene natural.",
        "Remove haze/fog and restore contrast and visibility. Keep colors natural.",
        "Dehaze the scene globally and recover distant details. Neutralize any color cast.",
        "Clear heavy haze and bring back natural sky and terrain tones. Avoid over‑saturation.",
        "Reduce atmospheric veil and enhance clarity. Preserve textures and geometry.",
        "Remove milder haze for a clean, natural photo. Keep white balance accurate.",
        "Restore a haze‑free image with balanced contrast. Do not introduce artifacts.",
    ],
    "deraining": [
        "Remove rain streaks and recover the scene behind them. Maintain sharpness and natural colors without artifacts.",
        "Remove rain streaks and restore the scene behind them. Maintain sharpness and natural colors.",
        "Derain the image by clearing streaks across the frame. Keep non‑rain content unchanged.",
        "Reduce heavy rain streaks and enhance visibility and contrast. Avoid over‑smoothing.",
        "Clean mild rain streaks while preserving textures and edges.",
        "Eliminate oblique streaks and recover clarity. Maintain accurate tones.",
        "Clear rain lines without affecting background details or geometry.",
    ],
    "raindrop removal": [
        "Remove raindrop occlusions on the lens/glass and reconstruct occluded content using context. Leave non‑occluded regions unchanged.",
        "Remove raindrop occlusions on the lens and reconstruct occluded content using context. Leave non‑occluded regions unchanged.",
        "Clear raindrops from glass and inpaint what’s behind them. Preserve the rest of the scene.",
        "Eliminate lens raindrops and recover hidden details. Do not alter clean areas.",
        "Remove heavy raindrop blobs and restore continuity of textures. Keep colors consistent.",
        "Erase small and large raindrops and reconstruct plausibly. Avoid artifacts outside drops.",
        "Clean water droplet distortions while leaving unaffected regions intact.",
    ],
    "desnowing": [
        "Remove snow flakes/streaks and restore visibility and contrast. Preserve textures and natural colors.",
        "Remove snowflakes and streaks and restore visibility and contrast. Keep a natural look.",
        "Desnow the image by clearing falling snow and veiling effects. Preserve textures and colors.",
        "Reduce heavy snowfall artifacts and recover scene details. Avoid over‑sharpening.",
        "Clean mild snow specks while keeping gradients smooth and edges crisp.",
        "Eliminate snow occlusions and restore clarity. Maintain accurate tones.",
        "Suppress snow veil and flake artifacts without affecting background content.",
    ],
    "compression artifacts removal": [
        "Remove JPEG blocking, ringing, and mosquito noise. Reconstruct smooth gradients and fine details without blurring.",
        "Remove JPEG blocking, ringing, and mosquito noise. Preserve fine details and smooth gradients.",
        "Deartifact and deblock compression artifacts while keeping edges clean and colors faithful.",
        "Reduce JPEG artifacts and recover natural textures. Avoid over‑smoothing.",
        "Clean compression noise and restore a crisp, natural look. Keep geometry unchanged.",
        "Suppress blocking and ringing; reconstruct subtle shading. Do not blur edges.",
        "Correct strong JPEG artifacts and retain true color and detail.",
    ],
    "low-light enhancement": [
        "Increase brightness and dynamic range for a low‑light image; correct white balance and suppress noise. Keep a natural appearance without clipping highlights.",
        "Brighten the image and expand dynamic range; suppress noise. Keep a natural appearance.",
        "Enhance a low‑light photo by raising exposure and correcting white balance. Avoid clipping highlights.",
        "Improve visibility in dark regions while keeping colors realistic and noise low.",
        "Increase brightness and contrast in a natural way. Preserve textures and avoid halos.",
        "Correct color cast and boost exposure for low light. Keep details intact, no over‑smoothing.",
        "Lift shadows and balance tones without introducing artifacts or glare.",
    ],
    "shadow removal": [
        "Remove cast shadows on subjects and surfaces while leaving non‑shadow regions untouched. Keep geometry, textures, and colors consistent.",
        "Remove cast shadows on subjects and surfaces while leaving non‑shadow regions untouched. Keep colors consistent.",
        "Correct heavy shadows and equalize illumination. Preserve textures and structure.",
        "Lift shadows on the scene to natural brightness. Avoid altering lit areas.",
        "Reduce soft and hard shadows while keeping geometry and tones realistic.",
        "Remove face/body shadows and match surrounding illumination. Do not change non‑shadow regions.",
        "Clean shadow bands and restore uniform lighting without halos.",
    ],
    "inpainting": [
        "Fill missing or masked regions realistically using surrounding context. Leave the rest of the image unchanged.",
        "Fill missing or masked regions realistically using surrounding context. Leave other areas unchanged.",
        "Inpaint the masked area with plausible content matching texture, color, and lighting.",
        "Reconstruct the absent region seamlessly; do not modify unmasked pixels.",
        "Complete holes with context‑aware content and preserve global consistency.",
        "Synthesize natural content for the mask while keeping boundaries seamless.",
        "Restore missing parts to look coherent with neighbors; avoid artifacts elsewhere.",
    ],
}