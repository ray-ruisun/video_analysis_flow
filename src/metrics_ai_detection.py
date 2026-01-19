#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Generated Video Detection Module

Detects:
- Deepfake videos (face-swapped)
- Fully synthetic videos (AI-generated)
- No-face AI videos

Models:
- GenConViT (video-level deepfake)
- ViT/SigLip (frame-level fake detection)
- OpenCV/MTCNN (face detection)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

# Lazy imports for models
_GENCONVIT_MODEL = None
_FRAME_DETECTOR = None
_FACE_CASCADE = None


@dataclass
class AIDetectionResult:
    """AI detection result for a single video"""
    # Overall verdict
    is_ai_generated: bool = False
    confidence: float = 0.0
    verdict: str = "Real"  # Real, Deepfake, Synthetic, Unknown
    
    # Video-level detection
    video_fake_score: float = 0.0
    video_model_used: str = ""
    
    # Frame-level detection
    frame_fake_scores: List[float] = field(default_factory=list)
    frame_avg_fake_score: float = 0.0
    frame_model_used: str = ""
    
    # Face detection
    faces_detected: int = 0
    frames_with_faces: int = 0
    frames_analyzed: int = 0
    no_face_ratio: float = 0.0
    
    # Detailed analysis
    analysis_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_ai_generated": self.is_ai_generated,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "video_fake_score": self.video_fake_score,
            "video_model_used": self.video_model_used,
            "frame_avg_fake_score": self.frame_avg_fake_score,
            "frame_model_used": self.frame_model_used,
            "faces_detected": self.faces_detected,
            "frames_with_faces": self.frames_with_faces,
            "frames_analyzed": self.frames_analyzed,
            "no_face_ratio": self.no_face_ratio,
            "analysis_details": self.analysis_details,
        }


def _load_face_cascade():
    """Load OpenCV face cascade classifier"""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def _load_frame_detector(model_name: str = "prithivMLmods/deepfake-detector-model-v1"):
    """Load frame-level fake detector"""
    global _FRAME_DETECTOR
    if _FRAME_DETECTOR is None:
        try:
            from transformers import pipeline
            logger.info(f"Loading frame detector: {model_name}")
            _FRAME_DETECTOR = pipeline(
                "image-classification",
                model=model_name,
                device="cuda" if _check_cuda() else "cpu"
            )
            logger.info("Frame detector loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load frame detector: {e}")
            _FRAME_DETECTOR = "failed"
    return _FRAME_DETECTOR if _FRAME_DETECTOR != "failed" else None


def _load_genconvit_model():
    """Load GenConViT video deepfake detector"""
    global _GENCONVIT_MODEL
    if _GENCONVIT_MODEL is None:
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
            
            model_name = "Deressa/GenConViT"
            logger.info(f"Loading GenConViT model: {model_name}")
            
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
            
            device = "cuda" if _check_cuda() else "cpu"
            model = model.to(device)
            model.eval()
            
            _GENCONVIT_MODEL = {"processor": processor, "model": model, "device": device}
            logger.info("GenConViT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load GenConViT: {e}")
            _GENCONVIT_MODEL = "failed"
    return _GENCONVIT_MODEL if _GENCONVIT_MODEL != "failed" else None


def _check_cuda() -> bool:
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def extract_frames(video_path: Path, num_frames: int = 16) -> List[np.ndarray]:
    """Extract uniformly sampled frames from video"""
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
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return frames


def detect_faces_in_frame(frame: np.ndarray, min_size: int = 30) -> int:
    """Detect faces in a single frame using OpenCV"""
    cascade = _load_face_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size)
    )
    
    return len(faces)


def analyze_frames_with_detector(
    frames: List[np.ndarray],
    model_name: str = "prithivMLmods/deepfake-detector-model-v1",
    threshold: float = 0.5
) -> Tuple[List[float], float]:
    """Analyze frames with frame-level fake detector"""
    detector = _load_frame_detector(model_name)
    
    if detector is None:
        logger.warning("Frame detector not available")
        return [], 0.0
    
    from PIL import Image
    
    fake_scores = []
    for frame in frames:
        try:
            pil_image = Image.fromarray(frame)
            results = detector(pil_image)
            
            # Find fake probability
            fake_score = 0.0
            for r in results:
                label = r['label'].lower()
                if 'fake' in label or 'synthetic' in label or 'ai' in label:
                    fake_score = r['score']
                    break
                elif 'real' in label:
                    fake_score = 1.0 - r['score']
                    break
            
            fake_scores.append(fake_score)
        except Exception as e:
            logger.warning(f"Frame analysis failed: {e}")
            fake_scores.append(0.0)
    
    avg_score = np.mean(fake_scores) if fake_scores else 0.0
    return fake_scores, avg_score


def analyze_video_with_genconvit(
    frames: List[np.ndarray],
    threshold: float = 0.5
) -> Tuple[float, str]:
    """Analyze video with GenConViT model"""
    model_data = _load_genconvit_model()
    
    if model_data is None:
        logger.warning("GenConViT model not available")
        return 0.0, "unavailable"
    
    import torch
    from PIL import Image
    
    processor = model_data["processor"]
    model = model_data["model"]
    device = model_data["device"]
    
    # Process each frame and average predictions
    all_probs = []
    
    with torch.no_grad():
        for frame in frames:
            try:
                pil_image = Image.fromarray(frame)
                inputs = processor(images=pil_image, return_tensors="pt").to(device)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # Assume index 1 is "fake" class (check model config if different)
                fake_prob = probs[0, 1].item() if probs.shape[1] > 1 else probs[0, 0].item()
                all_probs.append(fake_prob)
            except Exception as e:
                logger.warning(f"GenConViT frame analysis failed: {e}")
    
    avg_fake_score = np.mean(all_probs) if all_probs else 0.0
    return avg_fake_score, "Deressa/GenConViT"


def detect_ai_generated_video(
    video_path: Path,
    # Video model config
    video_model: str = "Deressa/GenConViT",
    video_fake_threshold: float = 0.5,
    video_frames: int = 16,
    # Frame model config  
    frame_detection_enabled: bool = True,
    frame_model: str = "prithivMLmods/deepfake-detector-model-v1",
    frame_fake_threshold: float = 0.5,
    frame_sample_count: int = 10,
    # Face detection config
    face_detection_enabled: bool = True,
    min_face_size: int = 30,
    no_face_threshold: float = 0.9,
    # Ensemble weights
    video_weight: float = 0.5,
    frame_weight: float = 0.3,
    face_weight: float = 0.2,
) -> AIDetectionResult:
    """
    Comprehensive AI-generated video detection
    
    Combines:
    - Video-level deepfake detection (GenConViT)
    - Frame-level fake detection
    - Face detection analysis
    """
    result = AIDetectionResult()
    
    logger.info(f"Starting AI detection for: {video_path}")
    
    # Extract frames for analysis
    max_frames = max(video_frames, frame_sample_count)
    frames = extract_frames(video_path, max_frames)
    result.frames_analyzed = len(frames)
    
    if not frames:
        logger.warning("No frames extracted from video")
        result.verdict = "Unknown"
        return result
    
    scores = []
    weights = []
    
    # 1. Video-level deepfake detection
    if video_model and video_model != "none":
        logger.info("Running video-level deepfake detection...")
        video_frames_subset = frames[:video_frames]
        
        if "genconvit" in video_model.lower():
            video_score, model_used = analyze_video_with_genconvit(
                video_frames_subset, video_fake_threshold
            )
        else:
            video_score, model_used = 0.0, "none"
        
        result.video_fake_score = video_score
        result.video_model_used = model_used
        
        if model_used != "unavailable":
            scores.append(video_score)
            weights.append(video_weight)
        
        logger.info(f"Video-level fake score: {video_score:.3f}")
    
    # 2. Frame-level fake detection
    if frame_detection_enabled:
        logger.info("Running frame-level fake detection...")
        frame_subset = frames[:frame_sample_count]
        
        frame_scores, frame_avg = analyze_frames_with_detector(
            frame_subset, frame_model, frame_fake_threshold
        )
        
        result.frame_fake_scores = frame_scores
        result.frame_avg_fake_score = frame_avg
        result.frame_model_used = frame_model
        
        if frame_scores:
            scores.append(frame_avg)
            weights.append(frame_weight)
        
        logger.info(f"Frame-level avg fake score: {frame_avg:.3f}")
    
    # 3. Face detection
    if face_detection_enabled:
        logger.info("Running face detection...")
        faces_per_frame = []
        
        for frame in frames:
            num_faces = detect_faces_in_frame(frame, min_face_size)
            faces_per_frame.append(num_faces)
        
        result.faces_detected = sum(faces_per_frame)
        result.frames_with_faces = sum(1 for f in faces_per_frame if f > 0)
        result.no_face_ratio = 1.0 - (result.frames_with_faces / len(frames))
        
        # No-face videos are potentially synthetic
        no_face_score = result.no_face_ratio if result.no_face_ratio > no_face_threshold else 0.0
        
        if no_face_score > 0:
            scores.append(no_face_score)
            weights.append(face_weight)
        
        logger.info(f"Face detection: {result.faces_detected} faces in {result.frames_with_faces}/{len(frames)} frames")
    
    # 4. Calculate ensemble score
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
            if result.no_face_ratio > no_face_threshold and result.frame_avg_fake_score > frame_fake_threshold:
                result.verdict = "Synthetic"
            elif result.video_fake_score > video_fake_threshold:
                result.verdict = "Deepfake"
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
    
    # Store detailed analysis
    result.analysis_details = {
        "video_model": video_model,
        "frame_model": frame_model if frame_detection_enabled else "disabled",
        "face_detection": "enabled" if face_detection_enabled else "disabled",
        "weights": {
            "video": video_weight,
            "frame": frame_weight,
            "face": face_weight
        },
        "thresholds": {
            "video_fake": video_fake_threshold,
            "frame_fake": frame_fake_threshold,
            "no_face": no_face_threshold
        }
    }
    
    logger.info(f"AI Detection complete: {result.verdict} (confidence: {result.confidence:.3f})")
    
    return result


# Convenience function for quick detection
def quick_detect(video_path: str) -> Dict[str, Any]:
    """Quick AI detection with default settings"""
    result = detect_ai_generated_video(Path(video_path))
    return result.to_dict()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        result = quick_detect(video_path)
        print(f"\nResult: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Video fake score: {result['video_fake_score']:.1%}")
        print(f"Frame avg score: {result['frame_avg_fake_score']:.1%}")
        print(f"No-face ratio: {result['no_face_ratio']:.1%}")
