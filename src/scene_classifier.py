#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene classification utilities using Places365 (ResNet-18).

Downloads official Places365 weights and category labels on first use,
and provides a helper to classify sampled video frames.
"""

from pathlib import Path
from typing import List

import cv2
import torch
from loguru import logger
from torchvision import models, transforms
import requests
from PIL import Image

MODEL_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
CATEGORY_URL = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
CACHE_DIR = Path.home() / ".cache" / "video_style_pipeline" / "places365"

_PLACES_MODEL = None
_PLACES_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_PLACES_CATEGORIES = None


def _download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    
    logger.info(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(dest, "wb") as handler:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handler.write(chunk)
    return dest


def _load_places_categories() -> List[str]:
    global _PLACES_CATEGORIES
    if _PLACES_CATEGORIES is not None:
        return _PLACES_CATEGORIES
    
    category_path = _download_file(CATEGORY_URL, CACHE_DIR / "categories_places365.txt")
    categories = []
    with open(category_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            # Expected format: "0 abbey"
            parts = line.split(" ")
            label = parts[-1]
            if label.startswith("/"):
                label = label.split("/")[-1]
            categories.append(label.replace("_", " "))
    
    _PLACES_CATEGORIES = categories
    return _PLACES_CATEGORIES


def _load_places_model():
    global _PLACES_MODEL
    if _PLACES_MODEL is not None:
        return _PLACES_MODEL
    
    weight_path = _download_file(MODEL_URL, CACHE_DIR / "resnet18_places365.pth.tar")
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
    
    model = models.resnet18(num_classes=365)
    model.load_state_dict(state_dict)
    model.eval()
    
    _PLACES_MODEL = model
    return _PLACES_MODEL


def classify_scene_categories(frames: List, topk: int = 3):
    """
    Classify scene categories using Places365.
    
    Args:
        frames: List of BGR frames (numpy arrays)
        topk: Number of top predictions to return
        
    Returns:
        list of dict: [{"label": str, "probability": float}, ...]
    """
    if not frames:
        logger.error("No frames provided for scene classification")
        raise ValueError("No frames provided for scene classification")
    
    model = _load_places_model()
    categories = _load_places_categories()
    
    tensors = []
    for frame in frames[: min(12, len(frames))]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensors.append(_PLACES_TRANSFORM(pil_image))
    
    batch = torch.stack(tensors)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        avg_probs = torch.mean(probs, dim=0)
        top_probs, top_idxs = torch.topk(avg_probs, k=topk)
    
    predictions = []
    for prob, idx in zip(top_probs, top_idxs):
        label = categories[int(idx)]
        predictions.append({
            "label": label,
            "probability": float(prob.item())
        })
    
    return predictions

