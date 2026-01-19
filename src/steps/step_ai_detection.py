#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Detection Step - Multi-Model Ensemble (SOTA 2025/2026)

Models:
- GenConViT: Face deepfake detection (~95.8% accuracy)
- CLIP: Zero-shot synthetic detection
- Temporal Analysis: Motion inconsistency
- Face Detection: No-face video analysis
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
    
    # Model selection
    use_genconvit: bool = True
    use_clip: bool = True
    use_temporal: bool = True
    use_face_detection: bool = True
    
    # Frame sampling
    num_frames: int = 16
    temporal_frames: int = 30
    
    # Thresholds
    fake_threshold: float = 0.5
    no_face_threshold: float = 0.9
    
    # Ensemble weights
    genconvit_weight: float = 0.4
    clip_weight: float = 0.3
    temporal_weight: float = 0.2
    face_weight: float = 0.1


@dataclass
class AIDetectionOutput(StepOutput):
    """AI detection output"""
    # Overall verdict
    is_ai_generated: bool = False
    confidence: float = 0.0
    verdict: str = "Unknown"  # Real, Deepfake, Synthetic, Suspicious, Unknown
    
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
            "success": self.success,
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


class AIDetectionStep(BaseStep):
    """
    AI-Generated Video Detection Step (SOTA 2025/2026)
    
    Multi-model ensemble:
    1. GenConViT - Face deepfake detection (~95.8% accuracy)
    2. CLIP - Zero-shot synthetic detection
    3. Temporal Analysis - Motion inconsistency detection
    4. Face Detection - No-face video analysis
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
                use_genconvit=input_data.use_genconvit,
                use_clip=input_data.use_clip,
                use_temporal=input_data.use_temporal,
                use_face_detection=input_data.use_face_detection,
                num_frames=input_data.num_frames,
                temporal_frames=input_data.temporal_frames,
                fake_threshold=input_data.fake_threshold,
                no_face_threshold=input_data.no_face_threshold,
                genconvit_weight=input_data.genconvit_weight,
                clip_weight=input_data.clip_weight,
                temporal_weight=input_data.temporal_weight,
                face_weight=input_data.face_weight,
            )
            
            output = AIDetectionOutput(
                success=True,
                is_ai_generated=result.is_ai_generated,
                confidence=result.confidence,
                verdict=result.verdict,
                genconvit_score=result.genconvit_score,
                genconvit_available=result.genconvit_available,
                clip_synthetic_score=result.clip_synthetic_score,
                clip_available=result.clip_available,
                temporal_score=result.temporal_score,
                temporal_anomalies=result.temporal_anomalies,
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
            f"GenConViT: {output.genconvit_score:.1%} | "
            f"CLIP: {output.clip_synthetic_score:.1%} | "
            f"Temporal: {output.temporal_score:.1%} | "
            f"NoFace: {output.no_face_ratio:.1%}"
        )
        logger.info(f"  → Models: {models_str}")