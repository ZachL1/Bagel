import os
import json
from torch.utils.data import Dataset
from diffusers.utils import load_image
import random

class ImageEditDataset(Dataset):
    def __init__(self, json_path, data_path, output_path, max_samples_per_source=None):
        self.json_path=json_path
        self.data_path = data_path
        self.output_path = output_path
        self.max_samples_per_source = max_samples_per_source
        
        data=[]
        with open(self.json_path,'r',encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        random.seed(42)
        random.shuffle(data)
        
        if max_samples_per_source is not None:
            data = self._sample_data_by_source(data, max_samples_per_source)
        
        self.data=data
    
    def _sample_data_by_source(self, data, max_samples):
        data_count = {}
        used_data = []
        
        for item in data:
            source = item.get("source", "unknown") 
            if data_count.get(source, 0) < max_samples:
                data_count[source] = data_count.get(source, 0) + 1

                target_path = item.get("target", "")
                output_file_path = os.path.join(self.output_path, target_path)
                
                if not os.path.exists(output_file_path):
                    used_data.append(item)
                else:
                    print(f"Output file already exists, skipping: {output_file_path}")     
        return used_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        single_data = self.data[idx]
        raw_image_path = single_data["raw"]
        image_path = os.path.join(self.data_path, raw_image_path)
        raw_image = load_image(image_path)
        prompt = single_data["instruction"]
        target_image_path = single_data["target"]
        
        return {
            'image': raw_image,
            'prompt': prompt,
            'target_path': target_image_path,
            'width': raw_image.width,
            'height': raw_image.height
        }



def collate_fn(batch):
    images = [item['image'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    target_paths = [item['target_path'] for item in batch]
    widths = [item['width'] for item in batch]
    heights = [item['height'] for item in batch]
    return {
        'image': images,  # 保持 PIL.Image.Image 对象列表
        'prompt': prompts,
        'target_paths': target_paths,  # 修正字段名
        'widths': widths,
        'heights': heights
    }