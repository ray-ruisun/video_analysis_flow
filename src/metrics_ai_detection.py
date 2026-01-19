#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Generated Video Detection Module (SOTA 2025/2026)

Multi-model ensemble detection:
1. Deep-Fake-Detector-v2 - Face deepfake detection (ViT-based, 92.12% acc)
2. CLIP-based - Zero-shot synthetic detection
3. Temporal Analysis - Motion inconsistency detection (disabled by default)
4. Face Detection - No-face video analysis

Models:
- prithivMLmods/Deep-Fake-Detector-v2-Model (ViT-based deepfake detector)
- openai/clip-vit-large-patch14 (zero-shot synthetic detection)
- OpenCV (face detection, temporal analysis)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from loguru import logger

# Model cache
_MODELS = {}


@dataclass
class AIDetectionResult:
    """Comprehensive AI detection result"""
    # Overall verdict
    is_ai_generated: bool = False
    confidence: float = 0.0
    verdict: str = "Real"  # Real, Deepfake, Synthetic, Suspicious, Unknown
    
    # Individual model scores
    deepfake_score: float = 0.0
    deepfake_available: bool = False
    
    clip_synthetic_score: float = 0.0
    clip_available: bool = False
    
    temporal_score: float = 0.0
    temporal_anomalies: int = 0
    
    # AIGC detection (AI-generated content)
    aigc_score: float = 0.0
    aigc_available: bool = False
    
    # Audio deepfake detection
    audio_deepfake_score: float = 0.0
    audio_deepfake_available: bool = False
    
    # Face analysis
    faces_detected: int = 0
    frames_with_faces: int = 0
    frames_analyzed: int = 0
    no_face_ratio: float = 0.0
    
    # Frame-level details
    frame_scores: List[float] = field(default_factory=list)
    
    # Analysis metadata
    models_used: List[str] = field(default_factory=list)
    analysis_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_ai_generated": self.is_ai_generated,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "deepfake_score": self.deepfake_score,
            "deepfake_available": self.deepfake_available,
            "clip_synthetic_score": self.clip_synthetic_score,
            "clip_available": self.clip_available,
            "temporal_score": self.temporal_score,
            "temporal_anomalies": self.temporal_anomalies,
            "aigc_score": self.aigc_score,
            "aigc_available": self.aigc_available,
            "audio_deepfake_score": self.audio_deepfake_score,
            "audio_deepfake_available": self.audio_deepfake_available,
            "faces_detected": self.faces_detected,
            "frames_with_faces": self.frames_with_faces,
            "frames_analyzed": self.frames_analyzed,
            "no_face_ratio": self.no_face_ratio,
            "models_used": self.models_used,
            "analysis_details": self.analysis_details,
        }


def _check_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def _get_device():
    return "cuda" if _check_cuda() else "cpu"


# =============================================================================
# Model Loaders
# =============================================================================
def _load_deepfake_detector():
    """Load Deep-Fake-Detector-v2-Model
    
    Model: prithivMLmods/Deep-Fake-Detector-v2-Model
    Architecture: Vision Transformer (ViT) - google/vit-base-patch16-224-in21k
    Accuracy: 92.12%
    Output: "Realism" or "Deepfake"
    
    Reference: https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model
    """
    if "deepfake" not in _MODELS:
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
            import torch
            
            model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
            logger.info(f"Loading Deep-Fake-Detector-v2: {model_name}")
            
            processor = ViTImageProcessor.from_pretrained(model_name)
            model = ViTForImageClassification.from_pretrained(model_name)
            
            device = _get_device()
            model = model.to(device)
            model.eval()
            
            _MODELS["deepfake"] = {
                "processor": processor,
                "model": model,
                "device": device,
                "id2label": model.config.id2label
            }
            logger.info(f"Deep-Fake-Detector-v2 loaded successfully. Labels: {model.config.id2label}")
        except Exception as e:
            logger.warning(f"Failed to load Deep-Fake-Detector-v2: {e}")
            _MODELS["deepfake"] = None
    
    return _MODELS.get("deepfake")


def _load_clip_detector():
    """Load CLIP for zero-shot synthetic detection"""
    if "clip" not in _MODELS:
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            model_name = "openai/clip-vit-large-patch14"
            logger.info(f"Loading CLIP: {model_name}")
            
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            
            device = _get_device()
            model = model.to(device)
            model.eval()
            
            _MODELS["clip"] = {
                "processor": processor,
                "model": model,
                "device": device
            }
            logger.info("CLIP loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CLIP: {e}")
            _MODELS["clip"] = None
    
    return _MODELS.get("clip")


def _load_face_cascade():
    """Load OpenCV face cascade"""
    if "face_cascade" not in _MODELS:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _MODELS["face_cascade"] = cv2.CascadeClassifier(cascade_path)
    return _MODELS["face_cascade"]


def _load_aigc_detector():
    """Load AIGC (AI-Generated Content) image detector
    
    Model: umm-maybe/AI-image-detector
    Detects AI-generated images (Stable Diffusion, DALL-E, Midjourney, etc.)
    """
    if "aigc" not in _MODELS:
        try:
            from transformers import pipeline
            import torch
            
            model_name = "umm-maybe/AI-image-detector"
            logger.info(f"Loading AIGC detector: {model_name}")
            
            device = 0 if _check_cuda() else -1
            pipe = pipeline("image-classification", model=model_name, device=device)
            
            _MODELS["aigc"] = {"pipeline": pipe}
            logger.info("AIGC detector loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load AIGC detector: {e}")
            _MODELS["aigc"] = None
    
    return _MODELS.get("aigc")


def _load_audio_deepfake_detector():
    """Load audio deepfake detector
    
    Model: MelodyMachine/Deepfake-audio-detection
    Detects synthetic/cloned voice audio
    """
    if "audio_deepfake" not in _MODELS:
        try:
            from transformers import pipeline
            import torch
            
            model_name = "MelodyMachine/Deepfake-audio-detection"
            logger.info(f"Loading audio deepfake detector: {model_name}")
            
            device = 0 if _check_cuda() else -1
            pipe = pipeline("audio-classification", model=model_name, device=device)
            
            _MODELS["audio_deepfake"] = {"pipeline": pipe}
            logger.info("Audio deepfake detector loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load audio deepfake detector: {e}")
            _MODELS["audio_deepfake"] = None
    
    return _MODELS.get("audio_deepfake")


# =============================================================================
# Frame Extraction
# =============================================================================
def extract_frames(video_path: Path, num_frames: int = 16) -> List[np.ndarray]:
    """Extract uniformly sampled frames"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames


def extract_consecutive_frames(video_path: Path, start_ratio: float = 0.3, num_frames: int = 30) -> List[np.ndarray]:
    """Extract consecutive frames for temporal analysis"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    start_frame = int(total_frames * start_ratio)
    frames = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    
    cap.release()
    return frames


# =============================================================================
# Detection Methods
# =============================================================================
def detect_with_deepfake_detector(frames: List[np.ndarray]) -> Tuple[float, List[float]]:
    """Detect deepfakes using Deep-Fake-Detector-v2-Model
    
    Model outputs "Realism" (real) or "Deepfake" (fake)
    Returns the probability of being a deepfake.
    """
    model_data = _load_deepfake_detector()
    
    if model_data is None:
        return 0.0, []
    
    import torch
    from PIL import Image
    
    processor = model_data["processor"]
    model = model_data["model"]
    device = model_data["device"]
    id2label = model_data["id2label"]
    
    scores = []
    
    with torch.no_grad():
        for frame in frames:
            try:
                pil_image = Image.fromarray(frame)
                inputs = processor(images=pil_image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # Find "Deepfake" class probability
                fake_prob = 0.0
                for idx, label in id2label.items():
                    if "deepfake" in label.lower() or "fake" in label.lower():
                        fake_prob = probs[0, idx].item()
                        break
                
                # If no "Deepfake" label found, assume index 1 is fake
                if fake_prob == 0.0 and len(probs[0]) >= 2:
                    # Check if "Realism" is at index 0
                    if "realism" in id2label.get(0, "").lower() or "real" in id2label.get(0, "").lower():
                        fake_prob = probs[0, 1].item()
                    else:
                        fake_prob = probs[0, 1].item()
                
                scores.append(fake_prob)
            except Exception as e:
                logger.debug(f"Deepfake detector frame error: {e}")
                scores.append(0.0)
    
    avg_score = np.mean(scores) if scores else 0.0
    return avg_score, scores


def detect_with_clip(frames: List[np.ndarray]) -> float:
    """Zero-shot synthetic detection using CLIP"""
    model_data = _load_clip_detector()
    
    if model_data is None:
        return 0.0
    
    import torch
    from PIL import Image
    
    processor = model_data["processor"]
    model = model_data["model"]
    device = model_data["device"]
    
    # Zero-shot prompts for synthetic detection
    real_prompts = [
        "a real photograph",
        "a natural photo taken by camera",
        "authentic video frame",
        "genuine photograph of real scene",
    ]
    
    fake_prompts = [
        "AI generated image",
        "synthetic artificial image",
        "computer generated fake image",
        "deepfake manipulated image",
        "artificially created digital art",
    ]
    
    all_prompts = real_prompts + fake_prompts
    
    scores = []
    
    with torch.no_grad():
        for frame in frames[:8]:  # Sample 8 frames for efficiency
            try:
                pil_image = Image.fromarray(frame)
                inputs = processor(
                    text=all_prompts,
                    images=pil_image,
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                outputs = model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()
                
                # Real prompts are first, fake prompts are last
                real_score = np.mean(probs[:len(real_prompts)])
                fake_score = np.mean(probs[len(real_prompts):])
                
                # Normalize to get fake probability
                total = real_score + fake_score
                fake_prob = fake_score / total if total > 0 else 0.5
                scores.append(fake_prob)
                
            except Exception as e:
                logger.debug(f"CLIP detection error: {e}")
    
    return np.mean(scores) if scores else 0.0


def detect_temporal_with_clip(frames: List[np.ndarray]) -> Tuple[float, int, Dict[str, Any]]:
    """Detect temporal inconsistencies using CLIP embeddings
    
    This is more reliable than optical flow because:
    1. CLIP understands semantic content, not just pixels
    2. Less sensitive to lighting changes and camera motion
    3. Can detect when content changes unnaturally (like face swaps)
    
    Returns:
        score: 0-1, higher = more likely AI-generated
        anomalies: number of detected temporal anomalies
        details: additional analysis information
    """
    if len(frames) < 3:
        return 0.0, 0, {}
    
    model_data = _load_clip_detector()
    if model_data is None:
        # Fallback to simple method if CLIP unavailable
        return _detect_temporal_simple(frames)
    
    import torch
    from PIL import Image
    
    processor = model_data["processor"]
    model = model_data["model"]
    device = model_data["device"]
    
    # Extract CLIP embeddings for each frame
    embeddings = []
    
    with torch.no_grad():
        for frame in frames:
            try:
                pil_image = Image.fromarray(frame)
                inputs = processor(images=pil_image, return_tensors="pt").to(device)
                image_features = model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy().flatten())
            except Exception as e:
                logger.debug(f"CLIP embedding error: {e}")
    
    if len(embeddings) < 3:
        return 0.0, 0, {}
    
    # Compute cosine similarities between consecutive frames
    similarities = []
    for i in range(1, len(embeddings)):
        sim = np.dot(embeddings[i], embeddings[i-1])
        similarities.append(sim)
    
    # Compute statistics
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    min_sim = np.min(similarities)
    
    # Detect anomalies: sudden drops in similarity
    anomalies = 0
    anomaly_threshold = mean_sim - 2 * std_sim
    anomaly_frames = []
    
    for i, sim in enumerate(similarities):
        if sim < anomaly_threshold:
            anomalies += 1
            anomaly_frames.append(i + 1)
    
    # Calculate temporal consistency score
    # Lower similarity variance = more consistent (real)
    # Higher variance or sudden drops = potential AI artifacts
    
    # Factors that indicate AI-generated:
    # 1. High variance in frame similarities (flickering)
    # 2. Very low minimum similarity (sudden content change)
    # 3. Multiple anomalies
    
    variance_score = min(1.0, std_sim * 10)  # Normalize std to 0-1
    consistency_score = max(0, 1 - mean_sim)  # Lower mean = less consistent
    anomaly_score = min(1.0, anomalies / 5)   # Normalize anomaly count
    
    # Weighted combination
    temporal_score = (
        0.3 * variance_score +
        0.3 * consistency_score + 
        0.4 * anomaly_score
    )
    
    # Cap at 1.0
    temporal_score = min(1.0, temporal_score)
    
    details = {
        "mean_similarity": float(mean_sim),
        "std_similarity": float(std_sim),
        "min_similarity": float(min_sim),
        "anomaly_frames": anomaly_frames,
        "method": "CLIP-Temporal"
    }
    
    logger.debug(f"CLIP Temporal: mean_sim={mean_sim:.3f}, std={std_sim:.3f}, anomalies={anomalies}")
    
    return temporal_score, anomalies, details


def _detect_temporal_simple(frames: List[np.ndarray]) -> Tuple[float, int, Dict[str, Any]]:
    """Simple fallback temporal detection using frame differences
    
    Note: Less reliable, used only when CLIP is unavailable.
    """
    if len(frames) < 3:
        return 0.0, 0, {}
    
    anomalies = 0
    diffs = []
    
    # Convert to grayscale and compute frame differences
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f for f in frames]
    
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
        diff_score = np.mean(diff)
        diffs.append(diff_score)
    
    if len(diffs) < 2:
        return 0.0, 0, {}
    
    # Compute second-order differences (acceleration)
    second_diffs = []
    for i in range(1, len(diffs)):
        second_diff = abs(diffs[i] - diffs[i-1])
        second_diffs.append(second_diff)
    
    # Detect anomalies (sudden changes in motion)
    mean_diff = np.mean(second_diffs)
    std_diff = np.std(second_diffs)
    threshold = mean_diff + 2 * std_diff
    
    for sd in second_diffs:
        if sd > threshold:
            anomalies += 1
    
    # Normalize to 0-1 scale (conservative)
    temporal_score = min(1.0, anomalies / 10.0)
    
    details = {
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "method": "Simple-FrameDiff"
    }
    
    return temporal_score, anomalies, details


def detect_temporal_anomalies(frames: List[np.ndarray]) -> Tuple[float, int]:
    """Detect temporal inconsistencies in video
    
    Uses CLIP-based temporal consistency analysis when available,
    falls back to simple frame difference method otherwise.
    
    CLIP-based analysis is more reliable because:
    - Understands semantic content, not just pixels
    - Less sensitive to lighting/camera motion
    - Better at detecting unnatural content changes
    """
    score, anomalies, _ = detect_temporal_with_clip(frames)
    return score, anomalies


def detect_faces(frames: List[np.ndarray], min_size: int = 30) -> Tuple[int, int]:
    """Detect faces in frames"""
    cascade = _load_face_cascade()
    
    total_faces = 0
    frames_with_faces = 0
    
    for frame in frames:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
        )
        
        num_faces = len(faces)
        total_faces += num_faces
        if num_faces > 0:
            frames_with_faces += 1
    
    return total_faces, frames_with_faces


def detect_with_aigc(frames: List[np.ndarray]) -> Tuple[float, List[float]]:
    """Detect AI-generated content using umm-maybe/AI-image-detector
    
    This model detects images generated by:
    - Stable Diffusion
    - DALL-E
    - Midjourney
    - Other diffusion models
    
    Returns:
        score: Average AI probability across frames
        frame_scores: Per-frame AI probabilities
    """
    model_data = _load_aigc_detector()
    
    if model_data is None:
        return 0.0, []
    
    from PIL import Image
    
    pipe = model_data["pipeline"]
    scores = []
    
    for frame in frames[:8]:  # Sample 8 frames
        try:
            pil_image = Image.fromarray(frame)
            result = pipe(pil_image)
            
            # Find "artificial" or "ai" label probability
            ai_prob = 0.0
            for item in result:
                label = item["label"].lower()
                if "artificial" in label or "ai" in label or "fake" in label:
                    ai_prob = item["score"]
                    break
            
            scores.append(ai_prob)
        except Exception as e:
            logger.debug(f"AIGC detection error: {e}")
    
    avg_score = np.mean(scores) if scores else 0.0
    return avg_score, scores


def detect_audio_deepfake(audio_path: Path) -> Tuple[float, Dict[str, Any]]:
    """Detect audio deepfakes using MelodyMachine/Deepfake-audio-detection
    
    Detects:
    - Voice cloning
    - Text-to-speech synthesis
    - Voice conversion
    
    Returns:
        score: Fake probability (0-1)
        details: Detection details
    """
    model_data = _load_audio_deepfake_detector()
    
    if model_data is None:
        return 0.0, {"available": False}
    
    if audio_path is None or not Path(audio_path).exists():
        return 0.0, {"available": False, "error": "No audio file"}
    
    pipe = model_data["pipeline"]
    
    try:
        result = pipe(str(audio_path))
        
        # Find "fake" or "spoof" label probability
        fake_prob = 0.0
        real_prob = 0.0
        
        for item in result:
            label = item["label"].lower()
            if "fake" in label or "spoof" in label or "synthetic" in label:
                fake_prob = item["score"]
            elif "real" in label or "bonafide" in label or "genuine" in label:
                real_prob = item["score"]
        
        # If we found real probability but not fake, compute it
        if fake_prob == 0.0 and real_prob > 0:
            fake_prob = 1.0 - real_prob
        
        details = {
            "available": True,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "raw_result": result
        }
        
        logger.info(f"Audio deepfake detection: fake={fake_prob:.3f}, real={real_prob:.3f}")
        return fake_prob, details
        
    except Exception as e:
        logger.warning(f"Audio deepfake detection failed: {e}")
        return 0.0, {"available": False, "error": str(e)}


# =============================================================================
# Main Detection Function
# =============================================================================
def detect_ai_generated_video(
    video_path: Path,
    audio_path: Optional[Path] = None,
    # Model selection
    use_deepfake: bool = True,
    use_clip: bool = True,
    use_temporal: bool = True,
    use_face_detection: bool = True,
    use_aigc: bool = True,           # AI-generated content detection
    use_audio_deepfake: bool = True, # Audio deepfake detection
    # Frame sampling
    num_frames: int = 16,
    temporal_frames: int = 16,
    # Thresholds
    fake_threshold: float = 0.5,
    no_face_threshold: float = 0.9,
    # Ensemble weights
    deepfake_weight: float = 0.3,
    clip_weight: float = 0.2,
    temporal_weight: float = 0.15,
    face_weight: float = 0.05,
    aigc_weight: float = 0.2,        # AIGC weight
    audio_deepfake_weight: float = 0.1,  # Audio deepfake weight
) -> AIDetectionResult:
    """
    Comprehensive AI-generated video detection using multiple models
    
    Ensemble approach:
    1. Deep-Fake-Detector-v2 - ViT-based deepfake detection (92.12% accuracy)
    2. CLIP - Zero-shot synthetic detection
    3. CLIP-Temporal - Semantic temporal consistency analysis
    4. Face Detection - No-face video analysis
    5. AIGC Detector - AI-generated content (Stable Diffusion, DALL-E, etc.)
    6. Audio Deepfake - Voice cloning / speech synthesis detection
    
    References:
    - https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model
    - https://huggingface.co/umm-maybe/AI-image-detector
    - https://huggingface.co/MelodyMachine/Deepfake-audio-detection
    """
    result = AIDetectionResult()
    
    logger.info(f"Starting AI detection for: {video_path}")
    
    # Extract frames
    frames = extract_frames(video_path, num_frames)
    result.frames_analyzed = len(frames)
    
    if not frames:
        logger.warning("No frames extracted")
        result.verdict = "Unknown"
        return result
    
    scores = []
    weights = []
    
    # 1. Deep-Fake-Detector-v2 Detection
    if use_deepfake:
        logger.info("Running Deep-Fake-Detector-v2...")
        deepfake_score, frame_scores = detect_with_deepfake_detector(frames)
        result.deepfake_score = deepfake_score
        result.frame_scores = frame_scores
        result.deepfake_available = deepfake_score > 0 or len(frame_scores) > 0
        
        if result.deepfake_available:
            scores.append(deepfake_score)
            weights.append(deepfake_weight)
            result.models_used.append("Deep-Fake-Detector-v2")
            logger.info(f"Deepfake score: {deepfake_score:.3f}")
    
    # 2. CLIP Zero-Shot Detection (Synthetic)
    if use_clip:
        logger.info("Running CLIP zero-shot detection...")
        clip_score = detect_with_clip(frames)
        result.clip_synthetic_score = clip_score
        result.clip_available = clip_score > 0
        
        if result.clip_available:
            scores.append(clip_score)
            weights.append(clip_weight)
            result.models_used.append("CLIP-ZeroShot")
            logger.info(f"CLIP synthetic score: {clip_score:.3f}")
    
    # 3. Temporal Analysis (Motion Inconsistency) - Disabled by default
    if use_temporal and temporal_weight > 0:
        logger.info("Running temporal analysis...")
        consecutive_frames = extract_consecutive_frames(video_path, num_frames=temporal_frames)
        if consecutive_frames:
            temporal_score, anomalies = detect_temporal_anomalies(consecutive_frames)
            result.temporal_score = temporal_score
            result.temporal_anomalies = anomalies
            
            if temporal_score > 0:
                scores.append(temporal_score)
                weights.append(temporal_weight)
                result.models_used.append("Temporal-Analysis")
                logger.info(f"Temporal score: {temporal_score:.3f}, anomalies: {anomalies}")
    
    # 4. Face Detection
    if use_face_detection:
        logger.info("Running face detection...")
        total_faces, frames_with_faces = detect_faces(frames)
        result.faces_detected = total_faces
        result.frames_with_faces = frames_with_faces
        result.no_face_ratio = 1.0 - (frames_with_faces / len(frames)) if frames else 0.0
        
        # No-face videos with synthetic features are suspicious
        if result.no_face_ratio > no_face_threshold:
            # If no faces and other signals suggest synthetic, increase score
            no_face_score = 0.3  # Base score for no-face
            scores.append(no_face_score)
            weights.append(face_weight)
            result.models_used.append("Face-Analysis")
        
        logger.info(f"Face detection: {total_faces} faces in {frames_with_faces}/{len(frames)} frames")
    
    # 5. AIGC Detection (AI-Generated Content)
    if use_aigc:
        logger.info("Running AIGC detection...")
        aigc_score, aigc_frame_scores = detect_with_aigc(frames)
        result.aigc_score = aigc_score
        result.aigc_available = aigc_score > 0 or len(aigc_frame_scores) > 0
        
        if result.aigc_available:
            scores.append(aigc_score)
            weights.append(aigc_weight)
            result.models_used.append("AIGC-Detector")
            logger.info(f"AIGC score: {aigc_score:.3f}")
    
    # 6. Audio Deepfake Detection
    if use_audio_deepfake and audio_path is not None:
        logger.info("Running audio deepfake detection...")
        audio_score, audio_details = detect_audio_deepfake(audio_path)
        result.audio_deepfake_score = audio_score
        result.audio_deepfake_available = audio_details.get("available", False)
        
        if result.audio_deepfake_available and audio_score > 0:
            scores.append(audio_score)
            weights.append(audio_deepfake_weight)
            result.models_used.append("Audio-Deepfake")
            logger.info(f"Audio deepfake score: {audio_score:.3f}")
    
    # 7. Calculate Ensemble Score
    if scores and weights:
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average
        ensemble_score = sum(s * w for s, w in zip(scores, normalized_weights))
        result.confidence = ensemble_score
        
        # Determine verdict
        if ensemble_score >= 0.7:
            result.is_ai_generated = True
            # Classify type based on which model contributed most
            if result.deepfake_score > 0.6:
                result.verdict = "Deepfake"
            elif result.audio_deepfake_score > 0.6:
                result.verdict = "Audio-Deepfake"
            elif result.aigc_score > 0.6:
                result.verdict = "AIGC"
            elif result.no_face_ratio > no_face_threshold and result.clip_synthetic_score > 0.5:
                result.verdict = "Synthetic"
            else:
                result.verdict = "AI-Generated"
        elif ensemble_score >= 0.4:
            result.is_ai_generated = False
            result.verdict = "Suspicious"
        else:
            result.is_ai_generated = False
            result.verdict = "Real"
    else:
        result.verdict = "Unknown"
        result.confidence = 0.0
    
    # Store analysis details
    result.analysis_details = {
        "weights": {
            "deepfake": deepfake_weight,
            "clip": clip_weight,
            "temporal": temporal_weight,
            "face": face_weight,
            "aigc": aigc_weight,
            "audio_deepfake": audio_deepfake_weight
        },
        "thresholds": {
            "fake": fake_threshold,
            "no_face": no_face_threshold
        },
        "frames_sampled": num_frames,
        "temporal_frames": temporal_frames,
        "models": [
            "prithivMLmods/Deep-Fake-Detector-v2-Model",
            "openai/clip-vit-large-patch14",
            "umm-maybe/AI-image-detector",
            "MelodyMachine/Deepfake-audio-detection"
        ]
    }
    
    logger.info(f"AI Detection complete: {result.verdict} (confidence: {result.confidence:.1%})")
    
    return result


def quick_detect(video_path: str) -> Dict[str, Any]:
    """Quick detection with default settings"""
    result = detect_ai_generated_video(Path(video_path))
    return result.to_dict()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        result = quick_detect(video_path)
        
        print(f"\n{'='*50}")
        print(f"AI Detection Result: {result['verdict']}")
        print(f"{'='*50}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Deepfake Score: {result['deepfake_score']:.1%}")
        print(f"CLIP Synthetic Score: {result['clip_synthetic_score']:.1%}")
        print(f"Temporal Score: {result['temporal_score']:.1%}")
        print(f"No-Face Ratio: {result['no_face_ratio']:.1%}")
        print(f"Models Used: {', '.join(result['models_used'])}")
