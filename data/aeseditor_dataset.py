# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
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
