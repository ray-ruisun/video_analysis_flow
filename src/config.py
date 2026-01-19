#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized Configuration for Video Style Analysis Pipeline

All hyperparameters are defined here - no magic values in code!
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class VisualConfig:
    """Visual analysis hyperparameters"""
    # Frame sampling
    target_frames: int = 50  # Number of frames to sample
    frame_mode: str = "edge"  # Frame selection mode: "uniform", "edge", "scene"
    
    # CLIP scene classification
    clip_model: str = "openai/clip-vit-large-patch14"
    clip_top_k: int = 5  # Top K scene categories to return
    
    # Color analysis
    color_bins: int = 180  # Hue histogram bins
    saturation_bins: int = 256
    brightness_bins: int = 256
    
    # Camera motion detection
    motion_threshold: float = 2.0  # Optical flow magnitude threshold
    
    # Scene detection
    scene_threshold: float = 27.0  # PySceneDetect threshold
    min_scene_len: int = 15  # Minimum frames per scene
    
    # Contact sheet
    contact_sheet_cols: int = 5
    contact_sheet_rows: int = 4
    thumbnail_size: tuple = (320, 180)


@dataclass
class AudioConfig:
    """Audio analysis hyperparameters"""
    # Sample rate
    sample_rate: int = 22050
    
    # CLAP model
    clap_model: str = "laion/larger_clap_music_and_speech"
    
    # BGM style categories
    bgm_styles: List[str] = field(default_factory=lambda: [
        "upbeat pop music", "calm ambient music", "energetic electronic music",
        "acoustic guitar music", "cinematic orchestral music", "jazz music",
        "classical piano music", "lo-fi hip hop music", "rock music",
        "country music", "R&B soul music", "silence or no music"
    ])
    
    # Mood categories
    moods: List[str] = field(default_factory=lambda: [
        "happy and cheerful", "calm and relaxing", "energetic and exciting",
        "sad and melancholic", "tense and dramatic", "mysterious",
        "romantic", "inspirational", "peaceful", "intense"
    ])
    
    # Instrument categories
    instruments: List[str] = field(default_factory=lambda: [
        "piano", "guitar", "drums", "bass", "violin", "synthesizer",
        "trumpet", "saxophone", "flute", "vocals"
    ])
    
    # Beat detection
    tempo_min: float = 60.0
    tempo_max: float = 200.0
    
    # Energy analysis
    hop_length: int = 512
    n_fft: int = 2048


@dataclass
class ASRConfig:
    """ASR (Speech Recognition) hyperparameters"""
    # Whisper model
    whisper_model: str = "large-v3-turbo"  # Options: tiny, base, small, medium, large-v3, large-v3-turbo
    whisper_compute_type: str = "float16"  # Options: float16, int8, int8_float16
    whisper_beam_size: int = 5
    whisper_vad_filter: bool = True
    
    # Language
    default_language: str = "en"  # Default language for ASR
    
    # Prosody analysis (librosa)
    pitch_fmin: float = 75.0  # Minimum pitch frequency
    pitch_fmax: float = 600.0  # Maximum pitch frequency
    
    # Emotion model (HuBERT)
    emotion_model: str = "superb/hubert-large-superb-er"
    
    # Catchphrase detection
    catchphrase_min_count: int = 2  # Minimum occurrences to be a catchphrase
    catchphrase_max_words: int = 4  # Maximum words in a catchphrase


@dataclass
class YOLOConfig:
    """YOLO object detection hyperparameters"""
    # Model
    model_name: str = "yolo11s.pt"  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt
    
    # Detection
    target_frames: int = 36  # Frames to analyze
    confidence_threshold: float = 0.25  # Minimum confidence
    iou_threshold: float = 0.45  # NMS IoU threshold
    max_detections: int = 300  # Maximum detections per frame
    
    # Object filtering
    target_classes: Optional[List[str]] = None  # None = all classes
    
    # Color analysis
    enable_colors: bool = True
    color_sample_size: int = 100  # Pixels to sample for color
    
    # Material analysis
    enable_materials: bool = True


@dataclass
class AIDetectionConfig:
    """AI-generated video detection hyperparameters (SOTA 2025/2026)"""
    # Enable/disable detection
    enabled: bool = True
    
    # Deep-Fake-Detector-v2 (Face Deepfake Detection)
    # Model: prithivMLmods/Deep-Fake-Detector-v2-Model - 92.12% accuracy
    # Architecture: ViT (google/vit-base-patch16-224-in21k)
    # Reference: https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model
    use_deepfake: bool = True
    deepfake_model: str = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    
    # CLIP Zero-Shot (Synthetic Detection)
    # Model: openai/clip-vit-large-patch14 - zero-shot synthetic detection
    use_clip: bool = True
    clip_model: str = "openai/clip-vit-large-patch14"
    
    # Temporal Analysis (CLIP-based Consistency)
    # Uses CLIP embeddings to detect semantic inconsistencies between frames
    # More reliable than optical flow - understands content, not just pixels
    use_temporal: bool = True
    temporal_frames: int = 16
    
    # AIGC Detection (AI-Generated Content)
    # Model: umm-maybe/AI-image-detector
    # Detects: Stable Diffusion, DALL-E, Midjourney, etc.
    use_aigc: bool = True
    aigc_model: str = "umm-maybe/AI-image-detector"
    
    # Audio Deepfake Detection
    # Model: MelodyMachine/Deepfake-audio-detection
    # Detects: Voice cloning, TTS synthesis, voice conversion
    use_audio_deepfake: bool = True
    audio_deepfake_model: str = "MelodyMachine/Deepfake-audio-detection"
    
    # Face Detection (No-Face Video Analysis)
    use_face_detection: bool = True
    face_detector: str = "opencv"  # Options: opencv
    min_face_size: int = 30
    no_face_threshold: float = 0.9  # If >90% frames have no face
    
    # Frame sampling
    num_frames: int = 16
    
    # Thresholds
    fake_threshold: float = 0.5
    
    # Ensemble weights (total = 1.0)
    deepfake_weight: float = 0.30   # ViT-based deepfake
    clip_weight: float = 0.20       # Zero-shot synthetic
    temporal_weight: float = 0.15   # CLIP-based temporal
    aigc_weight: float = 0.20       # AIGC detection
    audio_deepfake_weight: float = 0.10  # Audio deepfake
    face_weight: float = 0.05       # No-face analysis


@dataclass 
class ConsensusConfig:
    """Cross-video consensus hyperparameters"""
    # Minimum videos for consensus
    min_videos: int = 1
    
    # Majority voting threshold
    majority_threshold: float = 0.5  # >50% to be dominant
    
    # Numerical aggregation
    use_median: bool = True  # True=median, False=mean


@dataclass
class ReportConfig:
    """Report generation hyperparameters"""
    # Word report
    include_screenshots: bool = True
    max_screenshots: int = 6
    
    # PDF conversion
    pdf_timeout: int = 60  # Seconds
    
    # JSON export
    json_indent: int = 2


@dataclass
class UIConfig:
    """UI/Frontend hyperparameters"""
    # Gallery
    gallery_frames: int = 12
    gallery_columns: int = 4
    gallery_rows: int = 3
    
    # Language
    default_language: str = "en"  # Options: en, zh
    
    # Server
    default_port: int = 8088
    
    # Progress messages
    show_progress: bool = True


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    visual: VisualConfig = field(default_factory=VisualConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    ai_detection: AIDetectionConfig = field(default_factory=AIDetectionConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        import dataclasses
        return {
            "visual": dataclasses.asdict(self.visual),
            "audio": dataclasses.asdict(self.audio),
            "asr": dataclasses.asdict(self.asr),
            "yolo": dataclasses.asdict(self.yolo),
            "ai_detection": dataclasses.asdict(self.ai_detection),
            "consensus": dataclasses.asdict(self.consensus),
            "report": dataclasses.asdict(self.report),
            "ui": dataclasses.asdict(self.ui),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary"""
        return cls(
            visual=VisualConfig(**data.get("visual", {})),
            audio=AudioConfig(**data.get("audio", {})),
            asr=ASRConfig(**data.get("asr", {})),
            yolo=YOLOConfig(**data.get("yolo", {})),
            ai_detection=AIDetectionConfig(**data.get("ai_detection", {})),
            consensus=ConsensusConfig(**data.get("consensus", {})),
            report=ReportConfig(**data.get("report", {})),
            ui=UIConfig(**data.get("ui", {})),
        )


# Global default configuration
DEFAULT_CONFIG = PipelineConfig()


def get_default_config() -> PipelineConfig:
    """Get default configuration"""
    return PipelineConfig()


def load_config_from_file(path: str) -> PipelineConfig:
    """Load configuration from JSON file"""
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return PipelineConfig.from_dict(data)


def save_config_to_file(config: PipelineConfig, path: str):
    """Save configuration to JSON file"""
    import json
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
