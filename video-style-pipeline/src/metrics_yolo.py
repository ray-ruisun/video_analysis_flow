#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO object detection module.
Detects kitchen appliances, utensils, and common objects in cooking videos.
"""

from collections import Counter

# Optional dependency handling
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass


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
    "dining table", "chair", "vase", "clock"
}


def detect_objects_in_frames(frames, model_name="yolov8n.pt", 
                             target_objects=None, sample_rate=3):
    """
    Detect objects in sampled video frames using YOLOv8.
    
    Args:
        frames: List of video frames (BGR format)
        model_name: YOLOv8 model name (yolov8n, yolov8s, yolov8m, etc.)
        target_objects: Set of object names to track (None = all kitchen objects)
        sample_rate: Process every Nth frame (default: 3)
        
    Returns:
        dict: Detection results with object counts and statistics
    """
    if not YOLO_AVAILABLE:
        return {
            "available": False,
            "error": "YOLOv8 not installed. Install with: pip install ultralytics"
        }
    
    if not frames:
        return {
            "available": False,
            "error": "No frames provided"
        }
    
    if target_objects is None:
        target_objects = KITCHEN_OBJECTS
    
    try:
        # Load YOLO model
        model = YOLO(model_name)
        
        # Track detection counts
        object_counts = Counter()
        frame_appearances = Counter()  # Number of frames each object appears in
        confidence_scores = {}  # Track confidence scores per object
        
        # Sample frames for efficiency
        sampled_frames = frames[::sample_rate]
        
        for frame in sampled_frames:
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
        
        # Calculate average confidence per object
        avg_confidence = {
            obj: sum(scores) / len(scores)
            for obj, scores in confidence_scores.items()
        }
        
        # Get most common objects
        top_objects = object_counts.most_common(10)
        
        return {
            "available": True,
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
        return {
            "available": False,
            "error": str(e)
        }


def classify_kitchen_environment(detection_results):
    """
    Classify kitchen environment based on detected objects.
    
    Args:
        detection_results: Results from detect_objects_in_frames()
        
    Returns:
        dict: Kitchen environment classification
    """
    if not detection_results.get("available"):
        return {"available": False}
    
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
    if detection_results.get("unique_objects", 0) > 8:
        production = "High (diverse props)"
    elif detection_results.get("unique_objects", 0) > 4:
        production = "Medium"
    else:
        production = "Minimal/focused"
    
    return {
        "available": True,
        "environment_type": env_type,
        "cooking_style": style,
        "production_value": production,
        "has_appliances": has_major_appliances,
        "has_utensils": has_utensils,
        "has_ingredients": has_ingredients
    }


def analyze_object_colors(frames, detection_results, color_samples=5):
    """
    Analyze colors of detected objects (simplified approach).
    
    Args:
        frames: Video frames
        detection_results: YOLO detection results
        color_samples: Number of frames to sample for color analysis
        
    Returns:
        dict: Color analysis of detected objects
    """
    if not detection_results.get("available") or not YOLO_AVAILABLE:
        return {"available": False}
    
    try:
        import cv2
        import numpy as np
        
        model = YOLO(detection_results.get("model", "yolov8n.pt"))
        
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
        
        # Aggregate colors per object
        dominant_colors = {}
        for obj, colors in object_colors.items():
            counter = Counter(colors)
            dominant_colors[obj] = counter.most_common(1)[0][0] if counter else "Unknown"
        
        return {
            "available": True,
            "dominant_colors": dominant_colors
        }
        
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


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


def extract_full_yolo_metrics(frames, model_name="yolov8n.pt"):
    """
    Extract comprehensive YOLO-based metrics.
    
    Args:
        frames: List of video frames
        model_name: YOLOv8 model name
        
    Returns:
        dict: Complete object detection analysis
    """
    # Detect objects
    detection_results = detect_objects_in_frames(frames, model_name)
    
    if not detection_results.get("available"):
        return detection_results
    
    # Classify environment
    environment = classify_kitchen_environment(detection_results)
    
    # Analyze colors (optional, can be slow)
    # colors = analyze_object_colors(frames, detection_results)
    
    return {
        "available": True,
        "detection": detection_results,
        "environment": environment,
        # "colors": colors
    }

