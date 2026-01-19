#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene classification using CLIP (Contrastive Language-Image Pre-training)

升级自 Places365 ResNet-18 → OpenAI CLIP
优势:
- 零样本分类能力，无需预定义类别
- 多模态理解 (图像+文本)
- 更强的泛化能力
- 支持自定义场景描述

模型: openai/clip-vit-large-patch14 (HuggingFace)
备选: laion/CLIP-ViT-bigG-14-laion2B-39B-b160k (更大更强)
"""

from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

# 缓存目录
CACHE_DIR = Path.home() / ".cache" / "video_style_pipeline"

# CLIP 模型配置
CLIP_MODEL = "openai/clip-vit-large-patch14"
# 备选更强模型 (需要更多显存):
# "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
# "openai/clip-vit-base-patch32" (更快但精度略低)

# 全局模型缓存
_CLIP_MODEL = None
_CLIP_PROCESSOR = None

# 预定义的场景类别 (可扩展)
DEFAULT_SCENE_CATEGORIES = [
    # 厨房相关
    "a kitchen with cooking equipment",
    "a professional kitchen in a restaurant", 
    "a home kitchen with appliances",
    "a kitchen countertop with ingredients",
    "a dining room or eating area",
    
    # 室内场景
    "a living room",
    "a bedroom",
    "a bathroom",
    "an office or workspace",
    "a studio or workshop",
    
    # 室外场景
    "an outdoor garden or backyard",
    "a street or urban scene",
    "a park or natural setting",
    "a beach or waterfront",
    
    # 商业场景
    "a store or retail space",
    "a cafe or coffee shop",
    "a restaurant dining area",
    
    # 特殊场景
    "a food photography studio setup",
    "a close-up of food or dishes",
    "a cooking demonstration setup",
]

# 简化标签映射
LABEL_SIMPLIFY = {
    "a kitchen with cooking equipment": "Kitchen",
    "a professional kitchen in a restaurant": "Professional Kitchen",
    "a home kitchen with appliances": "Home Kitchen",
    "a kitchen countertop with ingredients": "Kitchen Countertop",
    "a dining room or eating area": "Dining Room",
    "a living room": "Living Room",
    "a bedroom": "Bedroom",
    "a bathroom": "Bathroom",
    "an office or workspace": "Office",
    "a studio or workshop": "Studio",
    "an outdoor garden or backyard": "Outdoor/Garden",
    "a street or urban scene": "Street/Urban",
    "a park or natural setting": "Park/Nature",
    "a beach or waterfront": "Beach/Waterfront",
    "a store or retail space": "Store/Retail",
    "a cafe or coffee shop": "Cafe",
    "a restaurant dining area": "Restaurant",
    "a food photography studio setup": "Food Photography Studio",
    "a close-up of food or dishes": "Food Close-up",
    "a cooking demonstration setup": "Cooking Demo",
}


def _load_clip_model():
    """Lazy-load CLIP model from HuggingFace."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR
    
    try:
        from transformers import CLIPProcessor, CLIPModel
        
        logger.info(f"Loading CLIP model: {CLIP_MODEL}")
        
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(
            CLIP_MODEL,
            cache_dir=str(CACHE_DIR / "hf_models")
        )
        _CLIP_MODEL = CLIPModel.from_pretrained(
            CLIP_MODEL,
            cache_dir=str(CACHE_DIR / "hf_models")
        )
        
        # 使用 GPU 如果可用
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL = _CLIP_MODEL.to(device)
        _CLIP_MODEL.eval()
        
        logger.info(f"CLIP model loaded successfully on {device}")
        return _CLIP_MODEL, _CLIP_PROCESSOR
        
    except ImportError:
        logger.error("transformers not installed. Install with: pip install transformers")
        raise ImportError("transformers is required for CLIP scene classification")
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        raise


def classify_scene_categories(
    frames: List[np.ndarray],
    topk: int = 3,
    custom_categories: Optional[List[str]] = None,
    return_all_scores: bool = False
) -> List[dict]:
    """
    Classify scene categories using CLIP zero-shot classification.
    
    Args:
        frames: List of BGR frames (numpy arrays)
        topk: Number of top predictions to return
        custom_categories: Custom scene descriptions (optional)
        return_all_scores: Whether to return scores for all categories
        
    Returns:
        list of dict: [{"label": str, "probability": float, "description": str}, ...]
    """
    if not frames:
        logger.error("No frames provided for scene classification")
        raise ValueError("No frames provided for scene classification")
    
    model, processor = _load_clip_model()
    device = next(model.parameters()).device
    
    # 使用自定义类别或默认类别
    categories = custom_categories if custom_categories else DEFAULT_SCENE_CATEGORIES
    
    # 准备图像
    pil_images = []
    for frame in frames[:min(12, len(frames))]:  # 最多处理12帧
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))
    
    # CLIP 推理
    all_probs = []
    
    with torch.no_grad():
        for pil_image in pil_images:
            inputs = processor(
                text=categories,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
            all_probs.append(probs.cpu())
    
    # 平均所有帧的概率
    avg_probs = torch.stack(all_probs).mean(dim=0)
    
    # 获取 top-k 结果
    top_probs, top_indices = torch.topk(avg_probs, k=min(topk, len(categories)))
    
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        description = categories[idx]
        label = LABEL_SIMPLIFY.get(description, description)
        predictions.append({
            "label": label,
            "probability": float(prob.item()),
            "description": description
        })
    
    # 可选: 返回所有分数
    if return_all_scores:
        all_scores = {
            LABEL_SIMPLIFY.get(cat, cat): float(avg_probs[i].item())
            for i, cat in enumerate(categories)
        }
        predictions.append({"all_scores": all_scores})
    
    logger.debug(f"CLIP scene classification: top={predictions[0]['label']} ({predictions[0]['probability']:.2%})")
    return predictions


def classify_with_custom_prompts(
    frames: List[np.ndarray],
    prompts: List[str],
    aggregate: str = "mean"
) -> dict:
    """
    Classify frames using custom text prompts.
    
    Args:
        frames: List of BGR frames
        prompts: List of text prompts to compare
        aggregate: How to aggregate frame scores ("mean", "max", "vote")
        
    Returns:
        dict: Classification results with scores for each prompt
    """
    model, processor = _load_clip_model()
    device = next(model.parameters()).device
    
    # 准备图像
    pil_images = []
    for frame in frames[:min(12, len(frames))]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))
    
    all_scores = []
    
    with torch.no_grad():
        for pil_image in pil_images:
            inputs = processor(
                text=prompts,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
            all_scores.append(probs.cpu().numpy())
    
    scores_array = np.array(all_scores)
    
    # 聚合策略
    if aggregate == "mean":
        final_scores = scores_array.mean(axis=0)
    elif aggregate == "max":
        final_scores = scores_array.max(axis=0)
    elif aggregate == "vote":
        votes = np.argmax(scores_array, axis=1)
        final_scores = np.bincount(votes, minlength=len(prompts)) / len(votes)
    else:
        final_scores = scores_array.mean(axis=0)
    
    # 构建结果
    results = {
        "scores": {prompt: float(score) for prompt, score in zip(prompts, final_scores)},
        "best_match": prompts[np.argmax(final_scores)],
        "best_score": float(np.max(final_scores)),
        "aggregate_method": aggregate
    }
    
    return results


def get_scene_embedding(frames: List[np.ndarray]) -> np.ndarray:
    """
    Get CLIP image embeddings for frames.
    
    Useful for similarity search or clustering.
    
    Args:
        frames: List of BGR frames
        
    Returns:
        np.ndarray: Averaged image embedding
    """
    model, processor = _load_clip_model()
    device = next(model.parameters()).device
    
    embeddings = []
    
    with torch.no_grad():
        for frame in frames[:min(12, len(frames))]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            inputs = processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            image_features = model.get_image_features(**inputs)
            embeddings.append(image_features.cpu().numpy())
    
    # 平均嵌入
    avg_embedding = np.mean(embeddings, axis=0)
    
    # L2 归一化
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    
    return avg_embedding.squeeze()
