#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual metrics extraction module.

Analyzes camera angles, focal length, motion, composition, color, lighting,
white balance, shot pacing, and transitions.
Uses computer vision techniques to extract stylistic patterns from video frames.
"""

import math
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
from PIL import Image
from loguru import logger
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from scene_classifier import classify_scene_categories


# Constants
MAX_WIDTH = 640


def sample_frames(video_path, target=36, max_width=MAX_WIDTH):
    """
    Sample frames uniformly from a video.
    
    Args:
        video_path: Path to video file
        target: Target number of frames to sample
        max_width: Maximum frame width (for memory efficiency)
        
    Returns:
        tuple: (frames, fps, total_frames, duration)
        
    Raises:
        FileNotFoundError: If video file not found
        ValueError: If video cannot be opened or has no frames
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise ValueError(f"Failed to open video: {video_path}")
    
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    if total == 0:
        cap.release()
        logger.error(f"Video has no frames: {video_path}")
        raise ValueError(f"Video has no frames: {video_path}")
    
    step = max(1, total // target) if total > 0 else 1
    idx = 0
    
    for k in range(target):
        target_frame = min(max(total - 1, 0), k * step)
        
        # Skip frames efficiently
        while idx < target_frame:
            cap.grab()
            idx += 1
        
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame {target_frame} from {video_path}")
            break
            
        # Resize if necessary
        h, w = frame.shape[:2]
        if w > max_width:
            new_h = int(max_width * h / w)
            frame = cv2.resize(frame, (max_width, new_h), cv2.INTER_AREA)
        
        frames.append(frame)
        idx = target_frame + 1
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = (total / fps) if (total > 0 and fps > 0) else 0.0
    
    cap.release()
    
    if not frames:
        logger.error(f"No frames extracted from video: {video_path}")
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    logger.debug(f"Sampled {len(frames)} frames from {video_path}, fps={fps:.2f}, duration={duration:.2f}s")
    return frames, fps, total, duration


def estimate_camera_angle(gray_frame):
    """
    Estimate camera angle based on edge orientation analysis.
    
    Args:
        gray_frame: Grayscale frame (numpy array)
        
    Returns:
        str: Camera angle label ("Top-down", "45° overhead", "Low-angle")
    """
    if gray_frame is None or gray_frame.size == 0:
        logger.error("Invalid grayscale frame provided to estimate_camera_angle")
        raise ValueError("Invalid grayscale frame provided")
    
    # Compute gradient directions
    gx = cv2.Sobel(gray_frame, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_frame, cv2.CV_32F, 0, 1, ksize=3)
    
    # Angle histogram (weighted by gradient magnitude)
    angles = np.arctan2(gy, gx) + np.pi
    magnitude = np.hypot(gx, gy)
    hist, _ = np.histogram(angles, bins=18, range=(0, 2*np.pi), 
                          weights=magnitude, density=True)
    
    # Anisotropy: measure of directional preference
    anisotropy = float(np.max(hist) - np.mean(hist))
    
    # Vertical vs horizontal edge strength
    vert_strength = float(np.sum(np.abs(gx))) / (np.sum(np.abs(gy)) + 1e-6)
    
    # Classification heuristics
    if anisotropy < 0.08 and vert_strength < 0.9:
        return "Top-down"
    elif anisotropy < 0.18:
        return "45° overhead"
    else:
        return "Low-angle"


def estimate_focal_length_tendency(frames):
    """
    Estimate focal length tendency (wide-angle vs telephoto) based on perspective distortion.
    
    Args:
        frames: List of BGR frames
        
    Returns:
        str: Focal length tendency ("Wide-angle", "Normal", "Telephoto")
    """
    if not frames:
        logger.error("No frames provided for focal length estimation")
        raise ValueError("No frames provided for focal length estimation")
    
    perspective_scores = []
    
    for frame in frames[:10]:  # Sample first 10 frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect lines using HoughLines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None or len(lines) == 0:
            continue
        
        # Analyze line convergence (perspective distortion)
        # Wide-angle: more convergence, Telephoto: less convergence
        angles = []
        for line in lines[:20]:  # Sample first 20 lines
            rho, theta = line[0]  # HoughLines returns [[[rho, theta], ...]]
            angles.append(theta)
        
        if len(angles) < 3:
            continue
        
        # Calculate angle variance (higher variance = more perspective distortion = wide-angle)
        angle_variance = np.var(angles)
        perspective_scores.append(angle_variance)
    
    if not perspective_scores:
        logger.warning("Could not estimate focal length, defaulting to Normal")
        return "Normal"
    
    avg_perspective = np.mean(perspective_scores)
    
    if avg_perspective > 0.15:
        return "Wide-angle"
    elif avg_perspective < 0.05:
        return "Telephoto"
    else:
        return "Normal"


def detect_camera_motion(frames, fps):
    """
    Detect camera motion using optical flow.
    
    Args:
        frames: List of BGR frames
        fps: Frames per second
        
    Returns:
        dict: Motion analysis with pan, tilt, zoom, dolly detection
    """
    if not frames or len(frames) < 2:
        logger.error("Need at least 2 frames for motion detection")
        raise ValueError("Need at least 2 frames for motion detection")
    
    # Convert to grayscale
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[:20]]  # Sample first 20
    
    # Detect features using Shi-Tomasi
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    motion_vectors = []
    
    for i in range(len(gray_frames) - 1):
        p0 = cv2.goodFeaturesToTrack(gray_frames[i], mask=None, **feature_params)
        
        if p0 is None or len(p0) < 10:
            continue
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_frames[i], gray_frames[i+1], p0, None, **lk_params)
        
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) < 5:
            continue
        
        # Calculate motion vectors
        vectors = good_new - good_old
        motion_vectors.extend(vectors.tolist())
    
    if not motion_vectors:
        logger.warning("No motion detected, camera appears static")
        return {
            "motion_type": "Static",
            "pan": False,
            "tilt": False,
            "zoom": False,
            "dolly": False
        }
    
    motion_array = np.array(motion_vectors)
    mean_motion = np.mean(motion_array, axis=0)
    std_motion = np.std(motion_array, axis=0)
    
    # Classify motion types
    pan_threshold = 2.0  # pixels
    tilt_threshold = 2.0
    zoom_threshold = 1.5  # variance in motion magnitude
    
    has_pan = abs(mean_motion[0]) > pan_threshold or std_motion[0] > pan_threshold
    has_tilt = abs(mean_motion[1]) > tilt_threshold or std_motion[1] > tilt_threshold
    
    # Zoom detection: check if motion vectors diverge/converge from center
    magnitudes = np.linalg.norm(motion_array, axis=1)
    zoom_variance = np.var(magnitudes)
    has_zoom = zoom_variance > zoom_threshold
    
    # Dolly: uniform forward/backward motion
    has_dolly = not has_pan and not has_tilt and zoom_variance > 0.5
    
    motion_types = []
    if has_pan:
        motion_types.append("Pan")
    if has_tilt:
        motion_types.append("Tilt")
    if has_zoom:
        motion_types.append("Zoom")
    if has_dolly:
        motion_types.append("Dolly")
    
    motion_type = " + ".join(motion_types) if motion_types else "Static"
    
    logger.debug(f"Camera motion detected: {motion_type}")
    
    return {
        "motion_type": motion_type,
        "pan": has_pan,
        "tilt": has_tilt,
        "zoom": has_zoom,
        "dolly": has_dolly,
        "mean_motion": mean_motion.tolist(),
        "motion_variance": float(zoom_variance)
    }


def analyze_composition_rules(frame):
    """
    Analyze composition rules (rule of thirds, center composition).
    
    Args:
        frame: BGR frame
        
    Returns:
        dict: Composition analysis
    """
    if frame is None or frame.size == 0:
        logger.error("Invalid frame provided for composition analysis")
        raise ValueError("Invalid frame provided")
    
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect key points (using corner detection as proxy for subject)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
    
    if corners is None or len(corners) < 5:
        logger.warning("Insufficient features for composition analysis")
        return {
            "rule_of_thirds": "Unknown",
            "center_composition": False,
            "composition_balance": "Unknown"
        }
    
    # Rule of thirds grid lines
    third_w = w / 3
    third_h = h / 3
    
    # Count points in each third
    points_in_thirds = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 3x3 grid
    
    for corner in corners:
        x, y = corner[0]
        col = min(2, int(x / third_w))
        row = min(2, int(y / third_h))
        idx = row * 3 + col
        points_in_thirds[idx] += 1
    
    # Check if points cluster in rule-of-thirds intersections
    intersection_indices = [1, 3, 5, 7]  # Four intersection points
    intersection_count = sum(points_in_thirds[i] for i in intersection_indices)
    total_count = sum(points_in_thirds)
    
    rule_of_thirds_ratio = intersection_count / total_count if total_count > 0 else 0
    
    # Center composition: check if points cluster in center
    center_idx = 4
    center_ratio = points_in_thirds[center_idx] / total_count if total_count > 0 else 0
    
    # Classify
    if rule_of_thirds_ratio > 0.4:
        rule_of_thirds = "Strong"
    elif rule_of_thirds_ratio > 0.2:
        rule_of_thirds = "Moderate"
    else:
        rule_of_thirds = "Weak"
    
    center_composition = center_ratio > 0.3
    
    # Balance: check left vs right distribution
    left_points = sum(points_in_thirds[i] for i in [0, 3, 6])
    right_points = sum(points_in_thirds[i] for i in [2, 5, 8])
    balance_ratio = min(left_points, right_points) / max(left_points, right_points) if max(left_points, right_points) > 0 else 0
    
    if balance_ratio > 0.7:
        balance = "Balanced"
    elif balance_ratio > 0.4:
        balance = "Moderately balanced"
    else:
        balance = "Unbalanced"
    
    return {
        "rule_of_thirds": rule_of_thirds,
        "center_composition": center_composition,
        "composition_balance": balance,
        "rule_of_thirds_ratio": float(rule_of_thirds_ratio),
        "center_ratio": float(center_ratio)
    }


def estimate_color_temperature_rgb(bgr_frame):
    """
    Estimate Correlated Color Temperature (CCT) from RGB values.
    
    Args:
        bgr_frame: BGR image (numpy array)
        
    Returns:
        tuple: (cct_kelvin, luminance) where CCT is in Kelvin (1000-20000)
    """
    if bgr_frame is None or bgr_frame.size == 0:
        logger.error("Invalid frame provided for CCT estimation")
        raise ValueError("Invalid frame provided")
    
    # Convert to RGB and normalize
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    avg = np.clip(np.mean(rgb.reshape(-1, 3), axis=0), 1e-6, 1.0)
    
    # Chromaticity coordinates
    r, g, b = avg
    sum_rgb = r + g + b
    x = r / sum_rgb
    y = g / sum_rgb
    
    # McCamy's approximation for CCT
    n = (x - 0.3320) / (0.1858 - y + 1e-6)
    cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
    
    # Luminance (Y in XYZ)
    Y = np.mean(0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2])
    
    return float(np.clip(cct, 1000, 20000)), float(Y)


def detect_lighting_type(bgr_frame):
    """
    Detect natural vs artificial light ratio based on spectral characteristics.
    
    Args:
        bgr_frame: BGR image
        
    Returns:
        dict: Lighting type analysis
    """
    if bgr_frame is None or bgr_frame.size == 0:
        logger.error("Invalid frame provided for lighting detection")
        raise ValueError("Invalid frame provided")
    
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Natural light typically has smoother spectrum, artificial light has spikes
    # Analyze color temperature distribution
    cct, _ = estimate_color_temperature_rgb(bgr_frame)
    
    # Natural light: typically 5000-6500K (daylight)
    # Artificial light: typically 2700-4000K (warm) or 4000-5000K (cool white)
    if 5000 <= cct <= 6500:
        lighting_type = "Natural (daylight)"
        natural_ratio = 0.8
    elif 2700 <= cct < 4000:
        lighting_type = "Artificial (warm)"
        natural_ratio = 0.2
    elif 4000 <= cct < 5000:
        lighting_type = "Artificial (cool white)"
        natural_ratio = 0.3
    else:
        lighting_type = "Mixed"
        natural_ratio = 0.5
    
    # Additional analysis: check for uniform vs directional lighting
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    brightness_std = np.std(gray)
    
    if brightness_std < 30:
        light_distribution = "Even/soft"
    elif brightness_std < 60:
        light_distribution = "Moderate"
    else:
        light_distribution = "Directional/harsh"
    
    return {
        "lighting_type": lighting_type,
        "natural_light_ratio": float(natural_ratio),
        "artificial_light_ratio": float(1.0 - natural_ratio),
        "light_distribution": light_distribution,
        "cct_kelvin": float(cct)
    }


def analyze_countertop_region(bgr_frame):
    """
    Analyze countertop color and texture using lower frame region.
    
    Args:
        bgr_frame: BGR image (numpy array)
        
    Returns:
        tuple: (color_description, texture_description)
    """
    if bgr_frame is None or bgr_frame.size == 0:
        logger.error("Invalid frame provided for countertop analysis")
        raise ValueError("Invalid frame provided")
    
    h, w = bgr_frame.shape[:2]
    
    # Region of interest (lower 55% of frame, typical countertop location)
    roi = bgr_frame[int(h * 0.45):, :]
    
    if roi.size < 64:
        logger.warning("Countertop region too small for analysis")
        return ("Unknown", "Unknown")
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    h_ch, s_ch, v_ch = cv2.split(hsv_roi)
    
    # Texture analysis via edge density
    edges = cv2.Canny(gray_roi, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / (edges.size + 1e-6)
    
    # Texture classification
    if edge_density < 0.02:
        texture = "Smooth/Laminated"
    elif edge_density < 0.07:
        texture = "Wood-grain"
    else:
        texture = "Stone-like"
    
    # Color classification
    mean_sat = np.mean(s_ch)
    mean_val = np.mean(v_ch)
    mean_hue = np.mean(h_ch)
    
    if mean_sat > 35:
        color = hue_to_description(mean_hue)
    elif mean_val > 40:
        color = "White/Gray/Black range"
    else:
        color = "Very dark"
    
    return color, texture


def hue_to_description(hue):
    """Convert HSV hue value to color description."""
    h = float(hue) % 180
    
    if (0 <= h <= 10) or h >= 170:
        return "Red-ish"
    elif h <= 25:
        return "Orange-warm"
    elif h <= 45:
        return "Warm yellow"
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


def saturation_band(sat):
    """Classify saturation level."""
    if sat < 40:
        return "Low"
    elif sat < 100:
        return "Moderate"
    else:
        return "High"


def brightness_band(val):
    """Classify brightness level."""
    if val < 60:
        return "Dark"
    elif val < 150:
        return "Medium"
    else:
        return "Bright"


def contrast_description(v_std):
    """Describe lighting consistency based on value std dev."""
    if v_std < 20:
        return "Even"
    elif v_std < 40:
        return "Mild contrast"
    else:
        return "High contrast"


def detect_shot_cuts(video_path):
    """
    Detect shot boundaries using PySceneDetect's ContentDetector.
    
    Args:
        video_path: Path to video file
        
    Returns:
        tuple: (num_cuts, cut_timestamps, avg_shot_length, transition_type)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video file not found for shot detection: {video_path}")
        raise FileNotFoundError(f"Video file not found for shot detection: {video_path}")
    
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    if not scene_list:
        logger.error("PySceneDetect failed to detect any scenes")
        raise ValueError("PySceneDetect failed to detect any scenes")
    
    cut_timestamps = [scene_list[idx][0].get_seconds() for idx in range(1, len(scene_list))]
    scene_durations = [
        max(1e-6, scene[1].get_seconds() - scene[0].get_seconds())
        for scene in scene_list
    ]
    
    avg_shot_length = float(np.mean(scene_durations)) if scene_durations else 0.0
    transition_type = "PySceneDetect ContentDetector"
    
    logger.debug(f"Detected {len(cut_timestamps)} cuts via PySceneDetect")
    return len(cut_timestamps), cut_timestamps, avg_shot_length, transition_type


def create_contact_sheet(frames, mode, output_path, cols=4, pad=6):
    """
    Create a contact sheet visualization from frames.
    
    Args:
        frames: List of frames
        mode: "edge", "mosaic", or "off"
        output_path: Where to save the sheet
        cols: Number of columns
        pad: Padding between images
        
    Returns:
        str or None: Path to saved sheet, or None if mode is "off"
    """
    if mode == "off" or not frames:
        return None
    
    thumbs = []
    for frame in frames:
        if mode == "edge":
            # Edge detection visualization
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            img = Image.fromarray(255 - edges).convert("RGB")
        else:  # mosaic
            # Pixelated effect
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (w // 8, h // 8), cv2.INTER_NEAREST)
            img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            img = img.resize((w, h), Image.NEAREST)
        
        thumbs.append(np.array(img))
    
    # Calculate dimensions
    max_h = max(im.shape[0] for im in thumbs)
    max_w = max(im.shape[1] for im in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    
    # Create sheet
    sheet_w = cols * max_w + (cols + 1) * pad
    sheet_h = rows * max_h + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 245, 245))
    
    for i, im in enumerate(thumbs):
        row = i // cols
        col = i % cols
        y = pad + row * (max_h + pad)
        x = pad + col * (max_w + pad)
        sheet.paste(Image.fromarray(im), (x, y))
    
    sheet.save(output_path)
    return str(output_path)


def extract_visual_metrics(video_path, output_dir, frame_mode="edge"):
    """
    Extract all visual metrics from a video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory for output assets
        frame_mode: Contact sheet mode ("edge", "mosaic", "off")
        
    Returns:
        dict: Visual metrics dictionary
        
    Raises:
        FileNotFoundError: If video file not found
        ValueError: If video processing fails
    """
    try:
        # Sample frames
        frames, fps, total_frames, duration = sample_frames(video_path)
        
        # Color and lighting analysis
        hues, sats, vals, v_stds = [], [], [], []
        ccts, luminances = [], []
        camera_angles = []
        ct_colors, ct_textures = [], []
        lighting_analyses = []
        composition_analyses = []
        
        for frame in frames:
            # HSV analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            hues.append(np.mean(h))
            sats.append(np.mean(s))
            vals.append(np.mean(v))
            v_stds.append(np.std(v))
            
            # Camera angle
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            camera_angles.append(estimate_camera_angle(gray))
            
            # Color temperature
            cct, lum = estimate_color_temperature_rgb(frame)
            ccts.append(cct)
            luminances.append(lum)
            
            # Lighting type
            lighting_analyses.append(detect_lighting_type(frame))
            
            # Composition (sample every 5th frame for efficiency)
            if len(composition_analyses) < 5:
                composition_analyses.append(analyze_composition_rules(frame))
            
            # Countertop analysis
            ct_col, ct_tex = analyze_countertop_region(frame)
            ct_colors.append(ct_col)
            ct_textures.append(ct_tex)
        
        # Focal length tendency
        focal_tendency = estimate_focal_length_tendency(frames)
        
        # Camera motion
        camera_motion = detect_camera_motion(frames, fps)
        
        # Scene classification (Places365)
        scene_categories = classify_scene_categories(frames, topk=3)
        
        # Shot cut detection via PySceneDetect
        num_cuts, cut_timestamps, avg_shot_length, trans_type = detect_shot_cuts(video_path)
        cut_frame_indices = []
        if duration > 0:
            for timestamp in cut_timestamps:
                idx = int((timestamp / duration) * len(frames))
                idx = max(0, min(len(frames) - 1, idx))
                cut_frame_indices.append(idx)
        
        # Aggregate lighting analysis
        avg_natural_ratio = np.mean([l["natural_light_ratio"] for l in lighting_analyses])
        avg_artificial_ratio = np.mean([l["artificial_light_ratio"] for l in lighting_analyses])
        dominant_lighting_type = Counter([l["lighting_type"] for l in lighting_analyses]).most_common(1)[0][0]
        
        # Aggregate composition
        avg_rule_of_thirds = Counter([c["rule_of_thirds"] for c in composition_analyses]).most_common(1)[0][0] if composition_analyses else "Unknown"
        avg_balance = Counter([c["composition_balance"] for c in composition_analyses]).most_common(1)[0][0] if composition_analyses else "Unknown"
        
        # Create contact sheet from key frames
        key_frame_indices = sorted(set([
            0,
            len(frames) // 2,
            len(frames) - 1
        ] + [max(0, k - 1) for k in cut_frame_indices] + [min(len(frames) - 1, k + 1) for k in cut_frame_indices]))
        
        key_frames = [frames[i] for i in key_frame_indices if i < len(frames)]
        
        output_path = Path(output_dir) / f"{Path(video_path).stem}_contact.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        contact_path = create_contact_sheet(key_frames, frame_mode, str(output_path))
        
        # Aggregate results
        result = {
            "fps": float(fps),
            "total_frames": int(total_frames),
            "duration": float(duration),
            "hue_family": hue_to_description(np.mean(hues)),
            "saturation_band": saturation_band(np.mean(sats)),
            "brightness_band": brightness_band(np.mean(vals)),
            "contrast": contrast_description(np.mean(v_stds)),
            "camera_angle": Counter(camera_angles).most_common(1)[0][0] if camera_angles else "Unknown",
            "focal_length_tendency": focal_tendency,
            "camera_motion": camera_motion,
            "composition": {
                "rule_of_thirds": avg_rule_of_thirds,
                "balance": avg_balance
            },
            "scene_categories": scene_categories,
            "cuts": int(num_cuts),
            "cut_timestamps": [float(ts) for ts in cut_timestamps],
            "avg_shot_length": float(avg_shot_length),
            "transition_type": trans_type,
            "countertop_color": Counter(ct_colors).most_common(1)[0][0] if ct_colors else "Unknown",
            "countertop_texture": Counter(ct_textures).most_common(1)[0][0] if ct_textures else "Unknown",
            "cct_mean": float(np.mean(ccts)) if ccts else None,
            "lighting": {
                "type": dominant_lighting_type,
                "natural_light_ratio": float(avg_natural_ratio),
                "artificial_light_ratio": float(avg_artificial_ratio)
            },
            "contact_sheet": contact_path
        }
        
        logger.info(f"Visual metrics extracted successfully for {Path(video_path).name}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract visual metrics from {video_path}: {e}")
        raise
