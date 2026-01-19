#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO object detection module (SOTA: YOLO26)

升级自 YOLO11 → YOLO26 (Ultralytics 2026 最新版本)
Reference: https://huggingface.co/collections/merve/yolo26-models

YOLO26 优势:
- 最新架构: 2026年1月发布
- 更高精度: 相比YOLO11进一步提升
- 支持零样本检测 (YOLOE26)
- 支持多种任务: 检测、分割、姿态、OBB、分类

模型选择:
- yolo26n.pt: 最快 (nano)
- yolo26s.pt: 平衡 (small) - 推荐
- yolo26m.pt: 中等 (medium)
- yolo26l.pt: 大型 (large)
- yolo26x.pt: 最强 (extra large)
"""

from collections import Counter
from typing import Dict, List, Set, Any, Optional
from loguru import logger

# Required dependency
try:
    from ultralytics import YOLO
    import ultralytics
    ULTRALYTICS_VERSION = ultralytics.__version__
    logger.debug(f"Ultralytics version: {ULTRALYTICS_VERSION}")
except ImportError as e:
    logger.error(f"Ultralytics not installed: {e}")
    logger.error("Install with: pip install ultralytics>=8.3.0")
    raise ImportError("ultralytics is required. Install with: pip install ultralytics>=8.3.0")

import cv2
import numpy as np

# ============================================================================
# 模型配置
# ============================================================================

# 默认使用 YOLO26x (最高精度)
DEFAULT_MODEL = "yolo26x.pt"
# 备选模型:
# "yolo26n.pt"  - 最快，适合实时处理
# "yolo26s.pt"  - 平衡速度和精度
# "yolo26m.pt"  - 较高精度
# "yolo26l.pt"  - 高精度
# "yolo26x.pt"  - 最高精度 (默认)

# 全局模型缓存
_YOLO_MODEL = None
_CURRENT_MODEL_NAME = None

# Kitchen and cooking-related objects in COCO dataset
KITCHEN_OBJECTS = {
    # Major appliances
    "refrigerator", "oven", "microwave", "sink", "toaster",
    # Utensils and cookware
    "bowl", "cup", "knife", "spoon", "fork", "bottle",
    "wine glass", "spatula", "pan", "pot",
    # Food and ingredients
    "apple", "orange", "banana", "broccoli", "carrot", "pizza",
    "hot dog", "cake", "donut", "sandwich",
    # Other relevant items
    "dining table", "chair", "vase", "clock",
    # 人物
    "person"
}


def _load_yolo_model(model_name: str = DEFAULT_MODEL) -> YOLO:
    """Lazy-load YOLO model with caching."""
    global _YOLO_MODEL, _CURRENT_MODEL_NAME
    
    if _YOLO_MODEL is not None and _CURRENT_MODEL_NAME == model_name:
        return _YOLO_MODEL
    
    logger.info(f"Loading YOLO model: {model_name}")
    _YOLO_MODEL = YOLO(model_name)
    _CURRENT_MODEL_NAME = model_name
    
    return _YOLO_MODEL


def detect_objects_in_frames(
    frames: List[np.ndarray],
    model_name: str = DEFAULT_MODEL, 
    target_objects: Optional[Set[str]] = None,
    sample_rate: int = 3
) -> Dict[str, Any]:
    """
    Detect objects in sampled video frames using YOLOv8.
    
    Args:
        frames: List of video frames (BGR format)
        model_name: YOLO model name (yolo26n, yolo26s, yolo26m, etc.)
        target_objects: Set of object names to track (None = all kitchen objects)
        sample_rate: Process every Nth frame (default: 3)
        
    Returns:
        dict: Detection results with object counts and statistics
        
    Raises:
        ValueError: If frames are invalid or detection fails
    """
    if not frames:
        logger.error("No frames provided for object detection")
        raise ValueError("No frames provided for object detection")
    
    if target_objects is None:
        target_objects = KITCHEN_OBJECTS
    
    try:
        # Load YOLO model (with caching)
        model = _load_yolo_model(model_name)
        
        # Track detection counts
        object_counts = Counter()
        frame_appearances = Counter()  # Number of frames each object appears in
        confidence_scores = {}  # Track confidence scores per object
        
        # Sample frames for efficiency
        sampled_frames = frames[::sample_rate]
        logger.debug(f"Processing {len(sampled_frames)} frames for object detection")
        
        for idx, frame in enumerate(sampled_frames):
            # Run detection
            results = model.predict(frame, verbose=False, conf=0.3)
            
            if not results:
                continue
            
            result = results[0]
            
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            
            # Track objects in this frame
            objects_in_frame = set()
            
            for box in result.boxes:
                # Get class ID and name
                cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                obj_name = result.names.get(cls_id, str(cls_id))
                
                # Filter for target objects
                if obj_name not in target_objects:
                    continue
                
                # Get confidence
                conf = float(box.conf[0].item()) if hasattr(box.conf[0], 'item') else float(box.conf[0])
                
                # Update counts
                object_counts[obj_name] += 1
                objects_in_frame.add(obj_name)
                
                # Track confidence scores
                if obj_name not in confidence_scores:
                    confidence_scores[obj_name] = []
                confidence_scores[obj_name].append(conf)
            
            # Update frame appearances
            for obj in objects_in_frame:
                frame_appearances[obj] += 1
        
        if not object_counts:
            logger.warning("No kitchen objects detected in frames")
            return {
                "model": model_name,
                "frames_processed": len(sampled_frames),
                "total_detections": 0,
                "unique_objects": 0,
                "object_counts": {},
                "frame_appearances": {},
                "avg_confidence": {},
                "top_objects": []
            }
        
        # Calculate average confidence per object
        avg_confidence = {
            obj: sum(scores) / len(scores)
            for obj, scores in confidence_scores.items()
        }
        
        # Get most common objects
        top_objects = object_counts.most_common(10)
        
        logger.info(f"Detected {len(object_counts)} unique objects, {sum(object_counts.values())} total detections")
        
        return {
            "model": model_name,
            "frames_processed": len(sampled_frames),
            "total_detections": sum(object_counts.values()),
            "unique_objects": len(object_counts),
            "object_counts": dict(object_counts),
            "frame_appearances": dict(frame_appearances),
            "avg_confidence": avg_confidence,
            "top_objects": top_objects
        }
        
    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        raise


def classify_kitchen_environment(detection_results):
    """
    Classify kitchen environment based on detected objects.
    
    Args:
        detection_results: Results from detect_objects_in_frames()
        
    Returns:
        dict: Kitchen environment classification
        
    Raises:
        ValueError: If detection results are invalid
    """
    if not detection_results:
        logger.error("Invalid detection results")
        raise ValueError("Invalid detection results")
    
    object_counts = detection_results.get("object_counts", {})
    
    # Classify environment type
    has_major_appliances = any(
        obj in object_counts 
        for obj in ["refrigerator", "oven", "microwave"]
    )
    
    has_utensils = any(
        obj in object_counts
        for obj in ["bowl", "knife", "spoon", "fork", "pan", "pot"]
    )
    
    has_ingredients = any(
        obj in object_counts
        for obj in ["apple", "orange", "banana", "broccoli", "carrot"]
    )
    
    # Environment description
    if has_major_appliances:
        env_type = "Full kitchen setup"
    elif has_utensils:
        env_type = "Cooking workspace"
    else:
        env_type = "Minimal setup"
    
    # Cooking style proxy
    if object_counts.get("bowl", 0) > 3:
        style = "Baking/mixing-heavy"
    elif object_counts.get("knife", 0) > 2:
        style = "Prep-focused"
    elif object_counts.get("pan", 0) > 0 or object_counts.get("pot", 0) > 0:
        style = "Stovetop cooking"
    else:
        style = "General cooking"
    
    # Estimate production value
    unique_objects = detection_results.get("unique_objects", 0)
    if unique_objects > 8:
        production = "High (diverse props)"
    elif unique_objects > 4:
        production = "Medium"
    else:
        production = "Minimal/focused"
    
    return {
        "environment_type": env_type,
        "cooking_style": style,
        "production_value": production,
        "has_appliances": has_major_appliances,
        "has_utensils": has_utensils,
        "has_ingredients": has_ingredients
    }


def analyze_object_colors(frames, detection_results, color_samples=5):
    """
    Analyze colors of detected objects.
    
    Args:
        frames: Video frames
        detection_results: YOLO detection results
        color_samples: Number of frames to sample for color analysis
        
    Returns:
        dict: Color analysis of detected objects
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not frames:
        logger.error("No frames provided for color analysis")
        raise ValueError("No frames provided for color analysis")
    
    if not detection_results or not detection_results.get("object_counts"):
        logger.error("Invalid detection results for color analysis")
        raise ValueError("Invalid detection results for color analysis")
    
    try:
        model = YOLO(detection_results.get("model", "yolo26x.pt"))
        
        # Sample frames
        sampled_frames = frames[::max(1, len(frames) // color_samples)][:color_samples]
        
        object_colors = {}
        
        for frame in sampled_frames:
            results = model.predict(frame, verbose=False, conf=0.4)
            
            if not results:
                continue
            
            result = results[0]
            
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            
            for box in result.boxes:
                cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                obj_name = result.names.get(cls_id, str(cls_id))
                
                if obj_name not in KITCHEN_OBJECTS:
                    continue
                
                # Get bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Extract ROI
                roi = frame[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # Analyze color
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                
                mean_hue = np.mean(h)
                mean_sat = np.mean(s)
                
                # Categorize color
                if mean_sat < 40:
                    color = "Neutral/Gray"
                else:
                    color = hue_to_color_name(mean_hue)
                
                # Store
                if obj_name not in object_colors:
                    object_colors[obj_name] = []
                object_colors[obj_name].append(color)
        
        # Aggregate colors per object with detailed distribution
        color_analysis = {}
        for obj, colors in object_colors.items():
            counter = Counter(colors)
            total = len(colors)
            distribution = [
                {
                    "color": color,
                    "count": count,
                    "percentage": round(count / total * 100, 1)
                }
                for color, count in counter.most_common()
            ]
            dominant = counter.most_common(1)[0][0] if counter else "Unknown"
            
            color_analysis[obj] = {
                "dominant": dominant,
                "distribution": distribution,
                "all_colors": list(counter.keys()),
                "sample_count": total
            }
        
        # Also create a summary of dominant colors for backward compatibility
        dominant_colors = {obj: data["dominant"] for obj, data in color_analysis.items()}
        
        logger.debug(f"Color analysis completed for {len(dominant_colors)} objects")
        
        return {
            "dominant_colors": dominant_colors,
            "detailed_analysis": color_analysis
        }
        
    except Exception as e:
        logger.error(f"Object color analysis failed: {e}")
        raise


def classify_material(roi):
    """
    Classify material type based on texture analysis (heuristic method).
    
    Args:
        roi: Region of interest (BGR image)
        
    Returns:
        str: Material type
    """
    if roi is None or roi.size == 0:
        return "Unknown"
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Texture analysis via edge density and variance
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / (edges.size + 1e-6)
    variance = float(np.var(gray))
    
    # Material classification heuristics
    if edge_density < 0.02 and variance < 500:
        return "Smooth/Plastic"
    elif edge_density < 0.05 and variance < 1000:
        return "Metal"
    elif edge_density < 0.08:
        return "Wood"
    elif edge_density < 0.12:
        return "Ceramic/Stone"
    else:
        return "Textured/Other"


def analyze_object_materials(frames, detection_results, material_samples=5):
    """
    Analyze materials of detected objects.
    
    Args:
        frames: Video frames
        detection_results: YOLO detection results
        material_samples: Number of frames to sample
        
    Returns:
        dict: Material analysis of detected objects
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not frames:
        logger.error("No frames provided for material analysis")
        raise ValueError("No frames provided for material analysis")
    
    if not detection_results or not detection_results.get("object_counts"):
        logger.error("Invalid detection results for material analysis")
        raise ValueError("Invalid detection results for material analysis")
    
    try:
        model = YOLO(detection_results.get("model", "yolo26x.pt"))
        
        # Sample frames
        sampled_frames = frames[::max(1, len(frames) // material_samples)][:material_samples]
        
        object_materials = {}
        
        for frame in sampled_frames:
            results = model.predict(frame, verbose=False, conf=0.4)
            
            if not results:
                continue
            
            result = results[0]
            
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            
            for box in result.boxes:
                cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                obj_name = result.names.get(cls_id, str(cls_id))
                
                if obj_name not in KITCHEN_OBJECTS:
                    continue
                
                # Get bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Extract ROI
                roi = frame[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # Classify material
                material = classify_material(roi)
                
                # Store
                if obj_name not in object_materials:
                    object_materials[obj_name] = []
                object_materials[obj_name].append(material)
        
        # Aggregate materials per object with detailed distribution
        material_analysis = {}
        for obj, materials in object_materials.items():
            counter = Counter(materials)
            total = len(materials)
            distribution = [
                {
                    "material": material,
                    "count": count,
                    "percentage": round(count / total * 100, 1)
                }
                for material, count in counter.most_common()
            ]
            dominant = counter.most_common(1)[0][0] if counter else "Unknown"
            
            material_analysis[obj] = {
                "dominant": dominant,
                "distribution": distribution,
                "all_materials": list(counter.keys()),
                "sample_count": total
            }
        
        # Also create a summary for backward compatibility
        dominant_materials = {obj: data["dominant"] for obj, data in material_analysis.items()}
        
        logger.debug(f"Material analysis completed for {len(dominant_materials)} objects")
        
        return {
            "dominant_materials": dominant_materials,
            "detailed_analysis": material_analysis
        }
        
    except Exception as e:
        logger.error(f"Object material analysis failed: {e}")
        raise


def hue_to_color_name(hue):
    """Convert HSV hue to color name."""
    h = float(hue) % 180
    
    if (0 <= h <= 10) or h >= 170:
        return "Red"
    elif h <= 25:
        return "Orange"
    elif h <= 45:
        return "Yellow"
    elif h <= 75:
        return "Yellow-green"
    elif h <= 100:
        return "Green"
    elif h <= 130:
        return "Cyan/Teal"
    elif h <= 150:
        return "Blue"
    else:
        return "Purple/Magenta"


def extract_full_yolo_metrics(frames, model_name="yolo26x.pt", 
                              enable_colors=True, enable_materials=True,
                              confidence_threshold=0.25):
    """
    Extract comprehensive YOLO-based metrics.
    
    Args:
        frames: List of video frames
        model_name: YOLO model name (e.g., yolo11s.pt)
        enable_colors: Enable color analysis
        enable_materials: Enable material analysis
        confidence_threshold: Detection confidence threshold
        
    Returns:
        dict: Complete object detection analysis
        
    Raises:
        ValueError: If frames are invalid or processing fails
    """
    if not frames:
        logger.error("No frames provided for YOLO analysis")
        raise ValueError("No frames provided for YOLO analysis")
    
    # Detect objects
    detection_results = detect_objects_in_frames(frames, model_name)
    
    if detection_results.get("total_detections", 0) == 0:
        logger.warning("No objects detected")
        return {
            "detection": detection_results,
            "environment": classify_kitchen_environment(detection_results),
            "colors": {"dominant_colors": {}} if enable_colors else None,
            "materials": {"dominant_materials": {}} if enable_materials else None
        }
    
    # Classify environment
    environment = classify_kitchen_environment(detection_results)
    
    result = {
        "detection": detection_results,
        "environment": environment
    }
    
    # Analyze colors
    if enable_colors:
        try:
            colors = analyze_object_colors(frames, detection_results)
            result["colors"] = colors
        except Exception as e:
            logger.error(f"Color analysis failed: {e}")
            raise
    
    # Analyze materials
    if enable_materials:
        try:
            materials = analyze_object_materials(frames, detection_results)
            result["materials"] = materials
        except Exception as e:
            logger.error(f"Material analysis failed: {e}")
            raise
    
    logger.info("YOLO metrics extracted successfully")
    return result
