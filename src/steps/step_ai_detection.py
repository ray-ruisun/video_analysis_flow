#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Detection Step - Deepfake & Synthetic Video Detection

Uses ensemble of:
- GenConViT (video-level deepfake)
- Frame-level fake detector
- Face detection analysis
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

from .base import BaseStep, StepInput, StepOutput


@dataclass
class AIDetectionInput(StepInput):
    """AI detection input configuration"""
    video_path: Path = None
    
    # Video model config
    video_model: str = "Deressa/GenConViT"
    video_fake_threshold: float = 0.5
    video_frames: int = 16
    
    # Frame model config
    frame_detection_enabled: bool = True
    frame_model: str = "prithivMLmods/deepfake-detector-model-v1"
    frame_fake_threshold: float = 0.5
    frame_sample_count: int = 10
    
    # Face detection config
    face_detection_enabled: bool = True
    min_face_size: int = 30
    no_face_threshold: float = 0.9
    
    # Ensemble weights
    video_weight: float = 0.5
    frame_weight: float = 0.3
    face_weight: float = 0.2


@dataclass
class AIDetectionOutput(StepOutput):
    """AI detection output"""
    # Overall verdict
    is_ai_generated: bool = False
    confidence: float = 0.0
    verdict: str = "Unknown"  # Real, Deepfake, Synthetic, Suspicious, Unknown
    
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
    
    # Analysis details
    analysis_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "is_ai_generated": self.is_ai_generated,
            "confidence": self.confidence,
            "verdict": self.verdict,
            "video_fake_score": self.video_fake_score,
            "video_model_used": self.video_model_used,
            "frame_fake_scores": self.frame_fake_scores,
            "frame_avg_fake_score": self.frame_avg_fake_score,
            "frame_model_used": self.frame_model_used,
            "faces_detected": self.faces_detected,
            "frames_with_faces": self.frames_with_faces,
            "frames_analyzed": self.frames_analyzed,
            "no_face_ratio": self.no_face_ratio,
            "analysis_details": self.analysis_details,
        }


class AIDetectionStep(BaseStep):
    """
    AI-Generated Video Detection Step
    
    Detects deepfakes, synthetic videos, and no-face AI videos
    using an ensemble of models.
    """
    
    step_name = "AI Detection"
    step_name_zh = "AI生成检测"
    
    def run(self, input_data: AIDetectionInput) -> AIDetectionOutput:
        """Run AI detection analysis"""
        self.log_start(input_data)
        
        try:
            from metrics_ai_detection import detect_ai_generated_video
            
            result = detect_ai_generated_video(
                video_path=input_data.video_path,
                video_model=input_data.video_model,
                video_fake_threshold=input_data.video_fake_threshold,
                video_frames=input_data.video_frames,
                frame_detection_enabled=input_data.frame_detection_enabled,
                frame_model=input_data.frame_model,
                frame_fake_threshold=input_data.frame_fake_threshold,
                frame_sample_count=input_data.frame_sample_count,
                face_detection_enabled=input_data.face_detection_enabled,
                min_face_size=input_data.min_face_size,
                no_face_threshold=input_data.no_face_threshold,
                video_weight=input_data.video_weight,
                frame_weight=input_data.frame_weight,
                face_weight=input_data.face_weight,
            )
            
            output = AIDetectionOutput(
                success=True,
                is_ai_generated=result.is_ai_generated,
                confidence=result.confidence,
                verdict=result.verdict,
                video_fake_score=result.video_fake_score,
                video_model_used=result.video_model_used,
                frame_fake_scores=result.frame_fake_scores,
                frame_avg_fake_score=result.frame_avg_fake_score,
                frame_model_used=result.frame_model_used,
                faces_detected=result.faces_detected,
                frames_with_faces=result.frames_with_faces,
                frames_analyzed=result.frames_analyzed,
                no_face_ratio=result.no_face_ratio,
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
            "Synthetic": "🤖",
            "AI-Generated": "🤖",
            "Unknown": "❓",
            "Error": "❌"
        }
        emoji = verdict_emoji.get(output.verdict, "❓")
        
        logger.info(
            f"  → {emoji} Verdict: {output.verdict} | "
            f"Confidence: {output.confidence:.1%} | "
            f"Video: {output.video_fake_score:.1%} | "
            f"Frame: {output.frame_avg_fake_score:.1%} | "
            f"NoFace: {output.no_face_ratio:.1%}"
        )
