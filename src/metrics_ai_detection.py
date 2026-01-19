#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Generated Video Detection Module (SOTA 2025/2026)

Multi-model ensemble detection:
1. GenConViT - Face deepfake detection (HuggingFace)
2. CLIP-based - Zero-shot synthetic detection
3. Temporal Analysis - Motion inconsistency detection
4. Face Detection - No-face video analysis

Models:
- Deressa/GenConViT (video-level deepfake, ~95.8% acc)
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
    genconvit_score: float = 0.0
    genconvit_available: bool = False
    
    clip_synthetic_score: float = 0.0
    clip_available: bool = False
    
    temporal_score: float = 0.0
    temporal_anomalies: int = 0
    
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
            "genconvit_score": self.genconvit_score,
            "genconvit_available": self.genconvit_available,
            "clip_synthetic_score": self.clip_synthetic_score,
            "clip_available": self.clip_available,
            "temporal_score": self.temporal_score,
            "temporal_anomalies": self.temporal_anomalies,
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
def _load_genconvit():
    """Load GenConViT deepfake detector"""
    if "genconvit" not in _MODELS:
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
            
            model_name = "Deressa/GenConViT"
            logger.info(f"Loading GenConViT: {model_name}")
            
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
            
            device = _get_device()
            model = model.to(device)
            model.eval()
            
            _MODELS["genconvit"] = {
                "processor": processor,
                "model": model,
                "device": device
            }
            logger.info("GenConViT loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load GenConViT: {e}")
            _MODELS["genconvit"] = None
    
    return _MODELS.get("genconvit")


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
def detect_with_genconvit(frames: List[np.ndarray]) -> Tuple[float, List[float]]:
    """Detect deepfakes using GenConViT"""
    model_data = _load_genconvit()
    
    if model_data is None:
        return 0.0, []
    
    import torch
    from PIL import Image
    
    processor = model_data["processor"]
    model = model_data["model"]
    device = model_data["device"]
    
    scores = []
    
    with torch.no_grad():
        for frame in frames:
            try:
                pil_image = Image.fromarray(frame)
                inputs = processor(images=pil_image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # Check model labels
                labels = model.config.id2label if hasattr(model.config, 'id2label') else {}
                
                # Find fake probability (usually index 1 or labeled as 'fake')
                fake_prob = 0.0
                if len(probs[0]) >= 2:
                    # Assume index 1 is fake
                    fake_prob = probs[0, 1].item()
                    
                    # Check labels if available
                    for idx, prob in enumerate(probs[0]):
                        label = labels.get(idx, "").lower()
                        if "fake" in label or "synthetic" in label:
                            fake_prob = prob.item()
                            break
                
                scores.append(fake_prob)
            except Exception as e:
                logger.debug(f"GenConViT frame error: {e}")
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


def detect_temporal_anomalies(frames: List[np.ndarray]) -> Tuple[float, int]:
    """Detect temporal inconsistencies in video (AI videos often have flickering)"""
    if len(frames) < 3:
        return 0.0, 0
    
    anomalies = 0
    diffs = []
    
    # Convert to grayscale and compute frame differences
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f for f in frames]
    
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
        diff_score = np.mean(diff)
        diffs.append(diff_score)
    
    if len(diffs) < 2:
        return 0.0, 0
    
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
    
    # Compute optical flow consistency
    flow_scores = []
    for i in range(1, min(len(gray_frames), 10)):
        try:
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i-1], gray_frames[i],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_scores.append(np.std(magnitude))
        except:
            pass
    
    # High variance in flow suggests potential AI generation
    if flow_scores:
        flow_variance = np.var(flow_scores)
        # Normalize to 0-1 scale
        temporal_score = min(1.0, flow_variance / 50.0 + anomalies / 10.0)
    else:
        temporal_score = anomalies / 10.0
    
    return min(1.0, temporal_score), anomalies


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


# =============================================================================
# Main Detection Function
# =============================================================================
def detect_ai_generated_video(
    video_path: Path,
    # Model selection
    use_genconvit: bool = True,
    use_clip: bool = True,
    use_temporal: bool = True,
    use_face_detection: bool = True,
    # Frame sampling
    num_frames: int = 16,
    temporal_frames: int = 30,
    # Thresholds
    fake_threshold: float = 0.5,
    no_face_threshold: float = 0.9,
    # Ensemble weights
    genconvit_weight: float = 0.4,
    clip_weight: float = 0.3,
    temporal_weight: float = 0.2,
    face_weight: float = 0.1,
) -> AIDetectionResult:
    """
    Comprehensive AI-generated video detection using multiple models
    
    Ensemble approach:
    1. GenConViT - Face deepfake detection
    2. CLIP - Zero-shot synthetic detection
    3. Temporal Analysis - Motion inconsistency
    4. Face Detection - No-face video analysis
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
    
    # 1. GenConViT Detection (Face Deepfake)
    if use_genconvit:
        logger.info("Running GenConViT detection...")
        genconvit_score, frame_scores = detect_with_genconvit(frames)
        result.genconvit_score = genconvit_score
        result.frame_scores = frame_scores
        result.genconvit_available = genconvit_score > 0 or len(frame_scores) > 0
        
        if result.genconvit_available:
            scores.append(genconvit_score)
            weights.append(genconvit_weight)
            result.models_used.append("GenConViT")
            logger.info(f"GenConViT score: {genconvit_score:.3f}")
    
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
    
    # 3. Temporal Analysis (Motion Inconsistency)
    if use_temporal:
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
    
    # 5. Calculate Ensemble Score
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
            if result.genconvit_score > 0.6:
                result.verdict = "Deepfake"
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
            "genconvit": genconvit_weight,
            "clip": clip_weight,
            "temporal": temporal_weight,
            "face": face_weight
        },
        "thresholds": {
            "fake": fake_threshold,
            "no_face": no_face_threshold
        },
        "frames_sampled": num_frames,
        "temporal_frames": temporal_frames
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
        print(f"GenConViT Score: {result['genconvit_score']:.1%}")
        print(f"CLIP Synthetic Score: {result['clip_synthetic_score']:.1%}")
        print(f"Temporal Score: {result['temporal_score']:.1%}")
        print(f"No-Face Ratio: {result['no_face_ratio']:.1%}")
        print(f"Models Used: {', '.join(result['models_used'])}")
