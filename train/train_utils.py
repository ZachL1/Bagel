# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import os


def create_logger(logging_dir, rank, filename="log"):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0 and logging_dir is not None:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(), 
                logging.FileHandler(f"{logging_dir}/{filename}.txt")
            ]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def get_latest_ckpt(checkpoint_dir):
    step_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if len(step_dirs) == 0:
        return None
    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    latest_step_dir = os.path.join(checkpoint_dir, step_dirs[-1])
    return latest_step_dir


import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional, Tuple, Dict, Any
import wandb


class TrainingVisualizer:
    def __init__(self, vae_model, latent_patch_size, latent_channel, save_dir: str = "training_visualizations", log_to_wandb: bool = True):
        """
        Visualizer for monitoring training predictions.
        
        Args:
            vae_model: VAE model for decoding latents to images
            latent_patch_size: size of the latent patch
            latent_channel: number of channels in the latent
            save_dir: directory to save visualizations
            log_to_wandb: whether to log visualizations to wandb
        """
        self.vae_model = vae_model
        self.latent_patch_size = latent_patch_size
        self.latent_channel = latent_channel
        self.save_dir = save_dir
        self.log_to_wandb = log_to_wandb
        
        os.makedirs(save_dir, exist_ok=True)
    
    def decode_latents_to_images(
        self, 
        predicted_latents: torch.Tensor,
        patchified_vae_latent_shapes: List[Tuple[int, int]],
        patchified: bool = True
    ) -> List[Image.Image]:
        """
        Args:
            predicted_latents: predicted latents [N, patch_latent_dim]
            patchified_vae_latent_shapes: patchified latent shapes for each image [(h, w), ...]
            patchified: whether the latents are patchified
        """
        images = []
        start_idx = 0
        
        for h, w in patchified_vae_latent_shapes:
            p = self.latent_patch_size
            if patchified:
                num_tokens = h * w
                current_latents = predicted_latents[start_idx:start_idx + num_tokens]
                # unpatchify
                latent = current_latents.reshape(h, w, p, p, self.latent_channel)
                latent = torch.einsum("hwpqc->chpwq", latent)
                latent = latent.reshape(1, self.latent_channel, h * p, w * p)
                start_idx += num_tokens
            else:
                latent = predicted_latents[start_idx:start_idx + 1, :, :h * p, :w * p]
                start_idx += 1
            
            # decode with VAE
            with torch.no_grad():
                decoded_image = self.vae_model.decode(latent.float())
                decoded_image = (decoded_image * 0.5 + 0.5).clamp(0, 1)
                decoded_image = (decoded_image[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                
            pil_image = Image.fromarray(decoded_image)
            images.append(pil_image)
            
        return images
    
    def visualize_prediction_comparison(
        self,
        target_images: List[Image.Image],
        predicted_images: List[Image.Image],
        prompts: Optional[List[str]] = None,
        timesteps: Optional[List[int]] = None,
        step: int = 0,
        save_name: str = "prediction_comparison"
    ) -> str:
        num_images = min(len(target_images), len(predicted_images), 4)  # maximum 4 samples
        if num_images == 0:
            return None
        
        fig = plt.figure(figsize=(16, 4 * num_images))
        gs = gridspec.GridSpec(num_images, 2, hspace=0.3, wspace=0.1)
        
        for i in range(num_images):
            # show target image
            ax1 = plt.subplot(gs[i, 0])
            ax1.imshow(target_images[i])
            ax1.set_title(f"Target Image {i+1}")
            ax1.axis('off')
            
            # show predicted image
            ax2 = plt.subplot(gs[i, 1])
            ax2.imshow(predicted_images[i])
            ax2.set_title(f"Predicted Image {i+1} (T={timesteps[i]})")
            ax2.axis('off')
            
            # add prompt to title
            if prompts and i < len(prompts):
                fig.text(0.5, 0.95 - i * (0.8 / num_images), f"Prompt: {prompts[i][:100]}...", 
                        ha='center', fontsize=10, wrap=True)
        
        plt.suptitle(f"Training Step {step}: Target vs Predicted Images", fontsize=16, y=0.98)
        
        # save image
        save_path = os.path.join(self.save_dir, f"step_{step}_{save_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_training_predictions(
        self,
        padded_latent: torch.Tensor,
        pred_dict: Dict[str, Any],
        step: int = 0,
        max_samples: int = 4
    ) -> Optional[str]:
        """
        Args:
            padded_latent: all padded latents (inluding raw and target latents)
            pred_dict: prediction dictionary
            step: current training step
            max_samples: maximum number of samples to visualize
            
        Returns:
            save path of the visualization
        """
    
        # get predicted latents
        predicted_latents = pred_dict.get('predicted_latents') # predicted target latents
        all_vae_latent_shapes = pred_dict.get('all_vae_latent_shapes')
        all_packed_timesteps = pred_dict.get('all_packed_timesteps') # >0 timesteps are target latents, <=0 timesteps are raw latents
        if predicted_latents is None or not all_vae_latent_shapes:
            return None
        
        # get timesteps
        timesteps = []
        start_idx = 0
        for h, w in all_vae_latent_shapes:
            num_tokens = h * w
            timesteps.append(all_packed_timesteps[start_idx].item())
            start_idx += num_tokens
        target = torch.Tensor(timesteps) > 0
        
        # decode predicted latents to images
        predicted_images = self.decode_latents_to_images(
            predicted_latents,
            [all_vae_latent_shapes[i] for i in range(len(all_vae_latent_shapes)) if target[i]],
        )[:max_samples]
        
        # get target images
        target_images = []
        if padded_latent is not None:
            target_images = self.decode_latents_to_images(
                padded_latent[target],
                [all_vae_latent_shapes[i] for i in range(len(all_vae_latent_shapes)) if target[i]],
                patchified=False,
            )[:max_samples]

        # get raw images
        raw_images = []
        if padded_latent is not None:
            raw_images = self.decode_latents_to_images(
                padded_latent[~target],
                [all_vae_latent_shapes[i] for i in range(len(all_vae_latent_shapes)) if ~target[i]],
                patchified=False,
            )
        
        # create visualization
        if target_images and len(target_images) == len(predicted_images):
            # create comparison visualization
            save_path = self.visualize_prediction_comparison(
                target_images, predicted_images, 
                step=step, save_name="target_pred",
                timesteps=[timesteps[i] for i in range(len(timesteps)) if target[i]],
            )
            raw_save_path = self.create_image_grid(
                raw_images,
                step=step,
                save_name="raw",
            )
        
            # log to wandb
            if self.log_to_wandb:
                try:
                    wandb.log({
                        f"training_vis/target_pred": wandb.Image(save_path),
                        f"training_vis/raw": wandb.Image(raw_save_path),
                        }, step=step)
                except Exception as e:
                    print(f"Failed to log to wandb: {e}")
        else:
            pass
            # # only show predicted images
            # titles = [f"Predicted {i+1}" for i in range(len(predicted_images))]
            # save_path = self.create_image_grid(
            #     predicted_images, titles, 
            #     step=step, save_name="training_predictions"
            # )
        
        return save_path
    
    def create_image_grid(
        self, 
        images: List[Image.Image], 
        titles: List[str] = None,
        step: int = 0,
        save_name: str = "training_grid"
    ) -> str:
        if not images:
            return None
            
        num_images = len(images)
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        for i, img in enumerate(images):
            if i < len(axes):
                axes[i].imshow(img)
                if titles and i < len(titles):
                    axes[i].set_title(titles[i])
                axes[i].axis('off')
        
        # 隐藏未使用的子图
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
            
        plt.suptitle(f"Training Step {step}", fontsize=16)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.save_dir, f"step_{step}_{save_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
                
        return save_path


# def get_model_predictions_during_training(
#     model, 
#     data_batch: Dict[str, Any], 
#     vae_model,
#     sample_noise: bool = True
# ) -> Tuple[Optional[torch.Tensor], List[Tuple[int, int]]]:
#     """
#     在训练过程中获取模型的预测latents
    
#     Args:
#         model: 训练中的模型（可能被FSDP包装）
#         data_batch: 当前批次的数据
#         vae_model: VAE模型
#         sample_noise: 是否使用采样噪声
        
#     Returns:
#         predicted_latents: 预测的latents
#         patchified_shapes: patchified VAE latent shapes
#     """
#     # 检查是否有视觉生成相关数据
#     mse_loss_indexes = data_batch.get('mse_loss_indexes')
#     patchified_shapes = data_batch.get('patchified_vae_latent_shapes', [])
#     if mse_loss_indexes is None or not patchified_shapes:
#         return None, []
    
#     with torch.no_grad():
#         model.eval()
        
#         # 获取实际的模型（处理FSDP包装）
#         if hasattr(model, 'module'):  # FSDP wrapped model
#             bagel_model = model.module
#         else:
#             bagel_model = model
        
#         try:
#             # 创建数据批次副本并添加返回预测的标志
#             viz_data_batch = {k: v for k, v in data_batch.items()}
#             viz_data_batch['return_predictions'] = True
            
#             # 执行前向传播获取预测结果
#             outputs = bagel_model(**viz_data_batch)
            
#             # 获取预测的latents
#             predicted_latents = outputs.get('predicted_latents')
#             returned_shapes = outputs.get('patchified_vae_latent_shapes', patchified_shapes)
            
#             model.train()
#             return predicted_latents, returned_shapes
            
#         except Exception as e:
#             print(f"Error getting model predictions: {e}")
#             import traceback
#             traceback.print_exc()
#             model.train()
#             return None, []


# def visualize_latent_progression(
#     latents_over_time: List[torch.Tensor],
#     vae_model,
#     save_path: str,
#     titles: List[str] = None
# ):
#     """
#     可视化latents在训练过程中的变化
#     """
#     if not latents_over_time:
#         return
        
#     images = []
#     for i, latents in enumerate(latents_over_time):
#         # 简单起见，只可视化第一个样本
#         if latents.shape[0] > 0:
#             # 假设是标准的latent格式
#             latent = latents[0:1]  # 取第一个样本
            
#             with torch.no_grad():
#                 decoded = vae_model.decode(latent)
#                 decoded = (decoded * 0.5 + 0.5).clamp(0, 1)
#                 decoded = (decoded[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                
#             images.append(Image.fromarray(decoded))
    
#     # 创建网格显示
#     if images:
#         num_images = len(images)
#         cols = min(5, num_images)
#         rows = (num_images + cols - 1) // cols
        
#         fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
#         if num_images == 1:
#             axes = [axes]
#         else:
#             axes = axes.flatten()
        
#         for i, img in enumerate(images):
#             axes[i].imshow(img)
#             if titles and i < len(titles):
#                 axes[i].set_title(titles[i])
#             else:
#                 axes[i].set_title(f"Step {i}")
#             axes[i].axis('off')
        
#         # 隐藏未使用的子图
#         for i in range(num_images, len(axes)):
#             axes[i].axis('off')
            
#         plt.suptitle("Latent Progression During Training")
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
