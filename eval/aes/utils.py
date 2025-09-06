#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for BAGEL inference scripts
"""

import os
import yaml
import random
import numpy as np
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

# Import model components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_configs(model_path: str, llm_path: str):
    """Load and prepare model configurations"""
    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(llm_path, "config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    return llm_config, vit_config, vae_model, vae_config, config


def create_empty_model(llm_config, vit_config, config):
    """Create empty model structure"""
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    return model


def prepare_tokenizer_and_transforms(model_path: str):
    """Prepare tokenizer and image transforms"""
    # Tokenizer Preparing
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Image Transform Preparing
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    return tokenizer, new_token_ids, vae_transform, vit_transform


def setup_device_mapping(model, max_mem_per_gpu: str = "40GiB"):
    """Setup device mapping for multi-GPU inference"""
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    return device_map


def load_model_weights(model, llm_path: str, device_map, use_ema: bool = True):
    """Load model weights and dispatch to devices"""
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(llm_path, "ema.safetensors" if use_ema else "model.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )
    model = model.eval()
    return model


def load_bagel_model(model_path: str, llm_path: str, max_mem_per_gpu: str = "40GiB", use_ema: bool = True):
    """
    Complete BAGEL model loading pipeline
    
    Args:
        model_path: Path to BAGEL model directory
        llm_path: Path to LLM checkpoint directory
        max_mem_per_gpu: Maximum memory per GPU
        
    Returns:
        Tuple of (model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    """
    print("Loading model configurations...")
    llm_config, vit_config, vae_model, vae_config, config = load_model_configs(model_path, llm_path)
    
    print("Creating empty model structure...")
    model = create_empty_model(llm_config, vit_config, config)
    
    print("Preparing tokenizer and transforms...")
    tokenizer, new_token_ids, vae_transform, vit_transform = prepare_tokenizer_and_transforms(model_path)
    
    print("Setting up device mapping...")
    device_map = setup_device_mapping(model, max_mem_per_gpu)
    print(f"Device map: {device_map}")
    
    print("Loading model weights...")
    model = load_model_weights(model, llm_path, device_map, use_ema)
    
    print("Model loaded successfully!")
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def create_inferencer(model_path: str, llm_path: str, max_mem_per_gpu: str = "40GiB", use_ema: bool = True):
    """
    Create InterleaveInferencer with loaded BAGEL model
    
    Args:
        model_path: Path to BAGEL model directory
        llm_path: Path to LLM checkpoint directory
        max_mem_per_gpu: Maximum memory per GPU
        
    Returns:
        InterleaveInferencer instance
    """
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_bagel_model(
        model_path, llm_path, max_mem_per_gpu, use_ema
    )
    
    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )
    
    return inferencer


# Default inference hyperparameters
DEFAULT_VLM_INFERENCE_PARAMS = {
    "max_think_token_n": 1000,
    "do_sample": False,
}

DEFAULT_EDIT_INFERENCE_PARAMS = {
    "max_think_token_n": 1000,
    "do_sample": False,
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 2.0,
    "cfg_interval": [0.0, 1.0],
    "timestep_shift": 3.0,
    "num_timesteps": 50,
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "text_channel",
}

DEFAULT_GENERATION_INFERENCE_PARAMS = {
    "max_think_token_n": 1000,
    "do_sample": False,
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 1.0,
    "cfg_interval": [0.4, 1.0],
    "timestep_shift": 3.0,
    "num_timesteps": 50,
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "global",
}


########################################################

def load_model_bf16(model, model_path):
    # Model Loading and Multi GPU Infernece Preparing
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "40GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
    
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        offload_folder="/tmp/offload",
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()
    return model

def load_model_and_tokenizer(args):
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module ="Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    config = BagelConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
    # model_state_dict = load_file(model_state_dict_path, device="cpu")
    # msg = model.load_state_dict(model_state_dict, strict=False)
    # print(msg)
    # del model_state_dict
    # model = model.cuda().eval()

    model = load_model_bf16(model, args.model_path)

    return model, tokenizer, new_token_ids


def build_transform(data_type="vlm_sft"):
    with open("./data/configs/aes.yaml", "r") as f:
        data_config = yaml.safe_load(f)

    max_image_size = data_config[data_type]['image_transform_args']['max_image_size']
    min_image_size = data_config[data_type]['image_transform_args']['min_image_size']
    image_stride = data_config[data_type]['image_transform_args']['image_stride']
    max_pixels = data_config[data_type]['image_transform_args']['max_pixels']

    image_transform = ImageTransform(
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        image_stride=image_stride,
        max_pixels=max_pixels,
    )

    return image_transform


def process_conversation(images, conversation):
    images = [pil_img2rgb(image) for image in images]
    return images, conversation
