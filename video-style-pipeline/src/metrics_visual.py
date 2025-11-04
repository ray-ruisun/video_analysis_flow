#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual metrics extraction module.
Analyzes camera angles, color, lighting, white balance, shot pacing, and transitions.
"""

import math
import numpy as np
import cv2
from pathlib import Path
from collections import Counter
from PIL import Image


# Constants
H_BINS, S_BINS = 30, 30
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
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
    return frames, fps, total, duration


def estimate_camera_angle(gray_frame):
    """
    Estimate camera angle based on edge orientation analysis.
    
    Args:
        gray_frame: Grayscale frame
        
    Returns:
        str: Camera angle label ("Top-down", "45° overhead", "Low-angle")
    """
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


def estimate_color_temperature_rgb(bgr_frame):
    """
    Estimate Correlated Color Temperature (CCT) from RGB values.
    
    Args:
        bgr_frame: BGR image
        
    Returns:
        tuple: (cct_kelvin, luminance)
    """
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


def analyze_countertop_region(bgr_frame):
    """
    Analyze countertop color and texture (proxy estimation).
    
    Args:
        bgr_frame: BGR image
        
    Returns:
        tuple: (color_description, texture_description)
    """
    h, w = bgr_frame.shape[:2]
    
    # Region of interest (lower 55% of frame, typical countertop location)
    roi = bgr_frame[int(h * 0.45):, :]
    
    if roi.size < 64:
        return ("Unknown", "Unknown")
    
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    h_ch, s_ch, v_ch = cv2.split(hsv_roi)
    
    # Texture analysis via edge density
    edges = cv2.Canny(gray_roi, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / (edges.size + 1e-6)
    
    # Texture classification
    if edge_density < 0.02:
        texture = "Smooth/Laminated (proxy)"
    elif edge_density < 0.07:
        texture = "Wood-grain (proxy)"
    else:
        texture = "Stone-like (proxy)"
    
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


def detect_shot_cuts(frames):
    """
    Detect shot boundaries using histogram comparison.
    
    Args:
        frames: List of BGR frames
        
    Returns:
        tuple: (num_cuts, cut_indices, transition_type)
    """
    if not frames:
        return 0, [], "Unknown"
    
    prev_hist = None
    bhattacharyya_dists = []
    
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256])
        hist = cv2.normalize(hist, None).astype(np.float32)
        
        if prev_hist is not None:
            dist = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            bhattacharyya_dists.append(dist)
        
        prev_hist = hist
    
    if not bhattacharyya_dists:
        return 0, [], "Unknown"
    
    # Adaptive threshold
    mu = float(np.mean(bhattacharyya_dists))
    sd = float(np.std(bhattacharyya_dists))
    threshold = mu + sd
    
    arr = np.array(bhattacharyya_dists)
    cut_indices = np.where(arr > threshold)[0].tolist()
    
    # Classify transition type based on cut clustering
    runs = []
    i = 0
    while i < len(arr):
        if arr[i] > threshold:
            j = i
            while j + 1 < len(arr) and arr[j + 1] > threshold * 0.95:
                j += 1
            runs.append(j - i + 1)
            i = j + 1
        else:
            i += 1
    
    if runs and np.median(runs) >= 3:
        transition_type = "Gradual/dissolve-like (proxy)"
    else:
        transition_type = "Hard cuts (proxy)"
    
    return len(cut_indices), cut_indices, transition_type


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
    """
    # Sample frames
    frames, fps, total_frames, duration = sample_frames(video_path)
    
    if not frames:
        return {
            "fps": 0.0, "total_frames": 0, "duration": 0.0,
            "hue_family": "Unknown", "saturation_band": "Unknown",
            "brightness_band": "Unknown", "contrast": "Unknown",
            "camera_angle": "Unknown", "cuts": 0, "transition_type": "Unknown",
            "countertop_color": "Unknown", "countertop_texture": "Unknown",
            "cct_mean": None, "contact_sheet": None
        }
    
    # Color and lighting analysis
    hues, sats, vals, v_stds = [], [], [], []
    ccts, luminances = [], []
    camera_angles = []
    ct_colors, ct_textures = [], []
    
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
        
        # Countertop analysis
        ct_col, ct_tex = analyze_countertop_region(frame)
        ct_colors.append(ct_col)
        ct_textures.append(ct_tex)
    
    # Shot cut detection
    num_cuts, cut_indices, trans_type = detect_shot_cuts(frames)
    
    # Create contact sheet from key frames
    key_frame_indices = sorted(set([
        0,
        len(frames) // 2,
        len(frames) - 1
    ] + [max(0, k - 1) for k in cut_indices] + [min(len(frames) - 1, k + 1) for k in cut_indices]))
    
    key_frames = [frames[i] for i in key_frame_indices if i < len(frames)]
    
    output_path = Path(output_dir) / f"{Path(video_path).stem}_contact.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact_path = create_contact_sheet(key_frames, frame_mode, str(output_path))
    
    # Aggregate results
    return {
        "fps": float(fps),
        "total_frames": int(total_frames),
        "duration": float(duration),
        "hue_family": hue_to_description(np.mean(hues)),
        "saturation_band": saturation_band(np.mean(sats)),
        "brightness_band": brightness_band(np.mean(vals)),
        "contrast": contrast_description(np.mean(v_stds)),
        "camera_angle": Counter(camera_angles).most_common(1)[0][0] if camera_angles else "Unknown",
        "cuts": int(num_cuts),
        "transition_type": trans_type,
        "countertop_color": Counter(ct_colors).most_common(1)[0][0] if ct_colors else "Unknown",
        "countertop_texture": Counter(ct_textures).most_common(1)[0][0] if ct_textures else "Unknown",
        "cct_mean": float(np.mean(ccts)) if ccts else None,
        "contact_sheet": contact_path
    }

