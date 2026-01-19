#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Detection Step - Multi-Model Ensemble (SOTA 2025/2026)

Models:
- Deep-Fake-Detector-v2: ViT-based deepfake detection (92.12% accuracy)
- CLIP: Zero-shot synthetic detection
- CLIP-Temporal: Semantic temporal consistency analysis
- AIGC Detector: AI-generated content (Stable Diffusion, DALL-E, etc.)
- Audio Deepfake: Voice cloning / speech synthesis detection
- Face Detection: No-face video analysis
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

from .base import PipelineStep, StepInput, StepOutput


@dataclass
class AIDetectionInput(StepInput):
    """AI detection input configuration"""
    video_path: Path = None
    audio_path: Optional[Path] = None
    
    # Model selection
    use_deepfake: bool = True
    use_clip: bool = True
    use_temporal: bool = True
    use_face_detection: bool = True
    use_aigc: bool = True
    use_audio_deepfake: bool = True
    
    # Frame sampling
    num_frames: int = 16
    temporal_frames: int = 16
    
    # Thresholds
    fake_threshold: float = 0.5
    no_face_threshold: float = 0.9
    
    # Ensemble weights
    deepfake_weight: float = 0.30
    clip_weight: float = 0.20
    temporal_weight: float = 0.15
    aigc_weight: float = 0.20
    audio_deepfake_weight: float = 0.10
    face_weight: float = 0.05


@dataclass
class AIDetectionOutput(StepOutput):
    """AI detection output"""
    # Overall verdict
    is_ai_generated: bool = False
    confidence: float = 0.0
    verdict: str = "Unknown"  # Real, Deepfake, Synthetic, AIGC, Audio-Deepfake, Suspicious, Unknown
    
    # Individual model scores
    deepfake_score: float = 0.0
    deepfake_available: bool = False
    
    clip_synthetic_score: float = 0.0
    clip_available: bool = False
    
    temporal_score: float = 0.0
    temporal_anomalies: int = 0
    
    aigc_score: float = 0.0
    aigc_available: bool = False
    
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
            "success": self.success,
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


class AIDetectionStep(PipelineStep[AIDetectionInput, AIDetectionOutput]):
    """
    AI-Generated Video Detection Step (SOTA 2025/2026)
    
    Multi-model ensemble:
    1. Deep-Fake-Detector-v2 - ViT-based deepfake detection (92.12% accuracy)
    2. CLIP - Zero-shot synthetic detection
    3. CLIP-Temporal - Semantic temporal consistency (uses CLIP embeddings)
    4. AIGC Detector - AI-generated content (Stable Diffusion, DALL-E, Midjourney)
    5. Audio Deepfake - Voice cloning / speech synthesis detection
    6. Face Detection - No-face video analysis
    
    References:
    - https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model
    - https://huggingface.co/umm-maybe/AI-image-detector
    - https://huggingface.co/MelodyMachine/Deepfake-audio-detection
    """
    
    @property
    def name(self) -> str:
        return "AI Detection"
    
    @property
    def description(self) -> str:
        return "AI生成检测 (DeepFake + CLIP + AIGC + Audio)"
    
    def run(self, input_data: AIDetectionInput) -> AIDetectionOutput:
        """Run AI detection analysis"""
        self.log_start(input_data)
        
        try:
            from metrics_ai_detection import detect_ai_generated_video
            
            result = detect_ai_generated_video(
                video_path=input_data.video_path,
                audio_path=input_data.audio_path,
                use_deepfake=input_data.use_deepfake,
                use_clip=input_data.use_clip,
                use_temporal=input_data.use_temporal,
                use_face_detection=input_data.use_face_detection,
                use_aigc=input_data.use_aigc,
                use_audio_deepfake=input_data.use_audio_deepfake,
                num_frames=input_data.num_frames,
                temporal_frames=input_data.temporal_frames,
                fake_threshold=input_data.fake_threshold,
                no_face_threshold=input_data.no_face_threshold,
                deepfake_weight=input_data.deepfake_weight,
                clip_weight=input_data.clip_weight,
                temporal_weight=input_data.temporal_weight,
                aigc_weight=input_data.aigc_weight,
                audio_deepfake_weight=input_data.audio_deepfake_weight,
                face_weight=input_data.face_weight,
            )
            
            output = AIDetectionOutput(
                success=True,
                is_ai_generated=result.is_ai_generated,
                confidence=result.confidence,
                verdict=result.verdict,
                deepfake_score=result.deepfake_score,
                deepfake_available=result.deepfake_available,
                clip_synthetic_score=result.clip_synthetic_score,
                clip_available=result.clip_available,
                temporal_score=result.temporal_score,
                temporal_anomalies=result.temporal_anomalies,
                aigc_score=result.aigc_score,
                aigc_available=result.aigc_available,
                audio_deepfake_score=result.audio_deepfake_score,
                audio_deepfake_available=result.audio_deepfake_available,
                faces_detected=result.faces_detected,
                frames_with_faces=result.frames_with_faces,
                frames_analyzed=result.frames_analyzed,
                no_face_ratio=result.no_face_ratio,
                frame_scores=result.frame_scores,
                models_used=result.models_used,
                analysis_details=result.analysis_details,
            )
            
            self.log_complete(output)
            self._log_summary(output)
            
            return output
            
        except Exception as e:
            logger.error(f"AI detection failed: {e}")
            error_output = AIDetectionOutput(
                success=False,
                error_message=str(e),
                verdict="Error"
            )
            self.log_complete(error_output)
            raise
    
    def _log_summary(self, output: AIDetectionOutput):
        """Log detection summary"""
        verdict_emoji = {
            "Real": "✅",
            "Suspicious": "⚠️",
            "Deepfake": "🎭",
            "AIGC": "🎨",
            "Audio-Deepfake": "🔊",
            "Synthetic": "🤖",
            "AI-Generated": "🤖",
            "Unknown": "❓",
            "Error": "❌"
        }
        emoji = verdict_emoji.get(output.verdict, "❓")
        
        models_str = ", ".join(output.models_used) if output.models_used else "None"
        
        logger.info(
            f"  → {emoji} {output.verdict} | "
            f"Conf: {output.confidence:.1%} | "
            f"DeepFake: {output.deepfake_score:.1%} | "
            f"CLIP: {output.clip_synthetic_score:.1%} | "
            f"AIGC: {output.aigc_score:.1%} | "
            f"Audio: {output.audio_deepfake_score:.1%}"
        )
        logger.info(f"  → Models: {models_str}")
