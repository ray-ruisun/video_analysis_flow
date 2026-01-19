#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流水线步骤基础类和数据结构定义

设计原则:
- 每个步骤有明确的输入类型和输出类型
- 步骤之间通过数据类传递结果，确保类型安全
- 使用 dataclass 确保数据结构清晰可追溯
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Generic
from loguru import logger


# ============================================================================
# 基础输入/输出类型
# ============================================================================

@dataclass
class StepInput:
    """步骤输入基类"""
    pass


@dataclass
class StepOutput:
    """步骤输出基类"""
    success: bool = True
    error_message: Optional[str] = None


# ============================================================================
# 视频/音频输入定义
# ============================================================================

@dataclass
class VideoInput(StepInput):
    """
    视频输入配置
    
    Attributes:
        video_path: 视频文件路径
        audio_path: 可选的音频文件路径（用于音频分析）
        work_dir: 工作目录，用于存放中间文件
        frame_mode: 截图模式 ("edge", "mosaic", "off")
    """
    video_path: Path
    audio_path: Optional[Path] = None
    work_dir: Path = field(default_factory=lambda: Path("work"))
    frame_mode: str = "edge"
    
    def __post_init__(self):
        """验证并转换路径类型"""
        if isinstance(self.video_path, str):
            self.video_path = Path(self.video_path)
        if isinstance(self.audio_path, str):
            self.audio_path = Path(self.audio_path)
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
    
    def validate(self) -> None:
        """验证输入有效性"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")


# ============================================================================
# 各步骤输出定义
# ============================================================================

@dataclass
class VisualOutput(StepOutput):
    """
    视觉分析输出
    
    包含: 色彩分析、镜头角度、构图、场景分类、剪辑节奏等
    
    每个分类指标都有两个字段:
    - xxx: 主导值 (dominant value)
    - xxx_detail: 详细分布 (包含 distribution, all_values, unique_count)
    """
    # 基础信息
    fps: float = 0.0
    total_frames: int = 0
    duration: float = 0.0
    sampled_frames: int = 0
    
    # 色彩分析 (带详细分布)
    hue_family: str = "Unknown"
    hue_detail: Dict[str, Any] = field(default_factory=dict)
    saturation_band: str = "Unknown"
    saturation_detail: Dict[str, Any] = field(default_factory=dict)
    brightness_band: str = "Unknown"
    brightness_detail: Dict[str, Any] = field(default_factory=dict)
    contrast: str = "Unknown"
    contrast_detail: Dict[str, Any] = field(default_factory=dict)
    
    # 色温
    cct_mean: Optional[float] = None
    cct_std: Optional[float] = None
    cct_range: Optional[Dict[str, float]] = None
    
    # 镜头分析 (带详细分布)
    camera_angle: str = "Unknown"
    camera_angle_detail: Dict[str, Any] = field(default_factory=dict)
    focal_length_tendency: str = "Unknown"
    camera_motion: Dict[str, Any] = field(default_factory=dict)
    
    # 构图分析
    composition: Dict[str, Any] = field(default_factory=dict)
    
    # 场景分类
    scene_categories: List[Dict[str, Any]] = field(default_factory=list)
    
    # 剪辑分析
    cuts: int = 0
    cut_timestamps: List[float] = field(default_factory=list)
    avg_shot_length: float = 0.0
    transition_type: str = "Unknown"
    
    # 台面分析 (带详细分布)
    countertop_color: str = "Unknown"
    countertop_color_detail: Dict[str, Any] = field(default_factory=dict)
    countertop_texture: str = "Unknown"
    countertop_texture_detail: Dict[str, Any] = field(default_factory=dict)
    
    # 光线分析 (带详细分布)
    lighting: Dict[str, Any] = field(default_factory=dict)
    
    # 截图路径
    contact_sheet: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（包含详细分布）"""
        return {
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "sampled_frames": self.sampled_frames,
            # 色彩分析 (含详细分布)
            "hue_family": self.hue_family,
            "hue_detail": self.hue_detail,
            "saturation_band": self.saturation_band,
            "saturation_detail": self.saturation_detail,
            "brightness_band": self.brightness_band,
            "brightness_detail": self.brightness_detail,
            "contrast": self.contrast,
            "contrast_detail": self.contrast_detail,
            # 色温
            "cct_mean": self.cct_mean,
            "cct_std": self.cct_std,
            "cct_range": self.cct_range,
            # 镜头分析 (含详细分布)
            "camera_angle": self.camera_angle,
            "camera_angle_detail": self.camera_angle_detail,
            "focal_length_tendency": self.focal_length_tendency,
            "camera_motion": self.camera_motion,
            # 构图
            "composition": self.composition,
            # 场景分类
            "scene_categories": self.scene_categories,
            # 剪辑分析
            "cuts": self.cuts,
            "cut_timestamps": self.cut_timestamps,
            "avg_shot_length": self.avg_shot_length,
            "transition_type": self.transition_type,
            # 台面分析 (含详细分布)
            "countertop_color": self.countertop_color,
            "countertop_color_detail": self.countertop_color_detail,
            "countertop_texture": self.countertop_texture,
            "countertop_texture_detail": self.countertop_texture_detail,
            # 光线分析
            "lighting": self.lighting,
            "contact_sheet": self.contact_sheet,
        }


@dataclass
class AudioOutput(StepOutput):
    """
    音频分析输出 (CLAP 版本)
    
    包含: 节拍、BPM、BGM风格、乐器检测、情绪分析等
    所有分类结果都包含详细分布信息
    """
    tempo_bpm: float = 0.0
    beat_times: List[float] = field(default_factory=list)
    num_beats: int = 0
    percussive_ratio: float = 0.0
    
    # 频谱特征
    spectral_centroid: float = 0.0
    spectral_flatness: float = 0.0
    zero_crossing_rate: float = 0.0
    
    # 能量特征
    mean_energy: float = 0.0
    energy_variance: float = 0.0
    spectral_rolloff: float = 0.0
    
    # 语音特征
    speech_ratio: float = 0.0
    
    # CLAP 分类结果 (BGM 风格)
    bgm_style: str = "Unknown"
    bgm_style_confidence: float = 0.0
    bgm_style_detail: Dict[str, Any] = field(default_factory=dict)
    
    # CLAP 分类结果 (情绪)
    mood: str = "Unknown"
    mood_confidence: float = 0.0
    mood_detail: Dict[str, Any] = field(default_factory=dict)
    mood_tags: List[Any] = field(default_factory=list)
    
    # 乐器检测
    instruments: Dict[str, Any] = field(default_factory=dict)
    
    # 调式
    key_signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式 (包含详细分布)"""
        return {
            "tempo_bpm": self.tempo_bpm,
            "beat_times": self.beat_times,
            "num_beats": self.num_beats,
            "percussive_ratio": self.percussive_ratio,
            "spectral_centroid": self.spectral_centroid,
            "spectral_flatness": self.spectral_flatness,
            "zero_crossing_rate": self.zero_crossing_rate,
            "mean_energy": self.mean_energy,
            "energy_variance": self.energy_variance,
            "spectral_rolloff": self.spectral_rolloff,
            "speech_ratio": self.speech_ratio,
            # BGM 风格 (含详细分布)
            "bgm_style": self.bgm_style,
            "bgm_style_confidence": self.bgm_style_confidence,
            "bgm_style_detail": self.bgm_style_detail,
            # 情绪 (含详细分布)
            "mood": self.mood,
            "mood_confidence": self.mood_confidence,
            "mood_detail": self.mood_detail,
            "mood_tags": self.mood_tags,
            # 乐器和调式
            "instruments": self.instruments,
            "key_signature": self.key_signature,
        }


@dataclass
class ASROutput(StepOutput):
    """
    语音识别分析输出
    
    包含: 转录文本、语速、口头禅、停顿分析、韵律、情感等
    """
    text: str = ""
    implementation: str = ""
    
    # 语速分析
    num_words: int = 0
    words_per_second: float = 0.0
    words_per_minute: float = 0.0
    pace: str = "Unknown"
    
    # 口头禅
    catchphrases: List[str] = field(default_factory=list)
    
    # 停顿分析
    num_pauses: int = 0
    pause_style: str = "Unknown"
    
    # 韵律分析（可选）
    prosody: Optional[Dict[str, Any]] = None
    
    # 情感分析（可选）
    emotion: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "text": self.text,
            "implementation": self.implementation,
            "num_words": self.num_words,
            "words_per_second": self.words_per_second,
            "words_per_minute": self.words_per_minute,
            "pace": self.pace,
            "catchphrases": self.catchphrases,
            "num_pauses": self.num_pauses,
            "pause_style": self.pause_style,
        }
        if self.prosody:
            result["prosody"] = self.prosody
        if self.emotion:
            result["emotion"] = self.emotion
        return result


@dataclass
class YOLOOutput(StepOutput):
    """
    YOLO 目标检测输出
    
    包含: 检测结果、环境分类、颜色分析、材质分析等
    """
    # 检测结果
    detection: Dict[str, Any] = field(default_factory=dict)
    
    # 环境分类
    environment: Dict[str, Any] = field(default_factory=dict)
    
    # 颜色分析
    colors: Optional[Dict[str, Any]] = None
    
    # 材质分析
    materials: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "detection": self.detection,
            "environment": self.environment,
        }
        if self.colors:
            result["colors"] = self.colors
        if self.materials:
            result["materials"] = self.materials
        return result


@dataclass
class VideoMetrics:
    """
    单个视频的完整分析结果
    
    聚合所有模块的输出
    """
    path: str
    visual: Optional[VisualOutput] = None
    audio: Optional[AudioOutput] = None
    asr: Optional[ASROutput] = None
    yolo: Optional[YOLOOutput] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {"path": self.path}
        if self.visual:
            result["visual"] = self.visual.to_dict()
        if self.audio:
            result["audio"] = self.audio.to_dict()
        if self.asr:
            result["asr"] = self.asr.to_dict()
        if self.yolo:
            result["yolo"] = self.yolo.to_dict()
        return result


@dataclass
class ConsensusInput(StepInput):
    """
    共识计算输入
    
    接收多个视频的分析结果
    """
    video_metrics: List[VideoMetrics] = field(default_factory=list)


@dataclass
class ConsensusOutput(StepOutput):
    """
    跨视频共识输出 (含详细分布)
    
    包含: 多数票决定的离散特征、中位数计算的数值特征等
    每个分类特征都有对应的 _detail 字段包含完整分布
    """
    # 视觉共识 (镜头)
    camera_angle: str = "N/A"
    camera_angle_detail: Dict[str, Any] = field(default_factory=dict)
    focal_length_tendency: str = "N/A"
    focal_length_detail: Dict[str, Any] = field(default_factory=dict)
    camera_motion: str = "N/A"
    camera_motion_detail: Dict[str, Any] = field(default_factory=dict)
    
    # 视觉共识 (构图)
    composition_rule_of_thirds: str = "N/A"
    composition_detail: Dict[str, Any] = field(default_factory=dict)
    
    # 视觉共识 (场景)
    scene_category: str = "N/A"
    scene_category_detail: Dict[str, Any] = field(default_factory=dict)
    
    # 视觉共识 (色彩)
    hue_family: str = "N/A"
    hue_detail: Dict[str, Any] = field(default_factory=dict)
    saturation: str = "N/A"
    saturation_detail: Dict[str, Any] = field(default_factory=dict)
    brightness: str = "N/A"
    brightness_detail: Dict[str, Any] = field(default_factory=dict)
    contrast: str = "N/A"
    contrast_detail: Dict[str, Any] = field(default_factory=dict)
    
    # 视觉共识 (数值)
    cct: Optional[float] = None
    natural_light_ratio: Optional[float] = None
    artificial_light_ratio: Optional[float] = None
    cuts_per_minute: Optional[float] = None
    avg_shot_length: Optional[float] = None
    
    # 视觉共识 (其他)
    transition_type: str = "N/A"
    transition_detail: Dict[str, Any] = field(default_factory=dict)
    countertop_color: str = "N/A"
    countertop_color_detail: Dict[str, Any] = field(default_factory=dict)
    countertop_texture: str = "N/A"
    countertop_texture_detail: Dict[str, Any] = field(default_factory=dict)
    
    # 音频共识
    beat_alignment: Optional[float] = None
    bgm_style: str = "N/A"
    bgm_style_detail: Dict[str, Any] = field(default_factory=dict)
    bgm_mood: str = "N/A"
    bgm_mood_detail: Dict[str, Any] = field(default_factory=dict)
    bgm_instruments: List[str] = field(default_factory=list)
    tempo_bpm: Optional[float] = None
    percussive_ratio: Optional[float] = None
    speech_ratio: Optional[float] = None
    key_signature: str = "N/A"
    
    # YOLO 共识
    yolo_available: bool = False
    yolo_environment: str = "N/A"
    yolo_style: str = "N/A"
    yolo_object_colors: Dict[str, str] = field(default_factory=dict)
    yolo_object_materials: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式 (含详细分布)"""
        return {
            # 镜头
            "camera_angle": self.camera_angle,
            "camera_angle_detail": self.camera_angle_detail,
            "focal_length_tendency": self.focal_length_tendency,
            "focal_length_detail": self.focal_length_detail,
            "camera_motion": self.camera_motion,
            "camera_motion_detail": self.camera_motion_detail,
            # 构图
            "composition_rule_of_thirds": self.composition_rule_of_thirds,
            "composition_detail": self.composition_detail,
            # 场景
            "scene_category": self.scene_category,
            "scene_category_detail": self.scene_category_detail,
            # 色彩
            "hue_family": self.hue_family,
            "hue_detail": self.hue_detail,
            "saturation": self.saturation,
            "saturation_detail": self.saturation_detail,
            "brightness": self.brightness,
            "brightness_detail": self.brightness_detail,
            "contrast": self.contrast,
            "contrast_detail": self.contrast_detail,
            # 数值
            "cct": self.cct,
            "natural_light_ratio": self.natural_light_ratio,
            "artificial_light_ratio": self.artificial_light_ratio,
            "cuts_per_minute": self.cuts_per_minute,
            "avg_shot_length": self.avg_shot_length,
            # 其他视觉
            "transition_type": self.transition_type,
            "transition_detail": self.transition_detail,
            "countertop_color": self.countertop_color,
            "countertop_color_detail": self.countertop_color_detail,
            "countertop_texture": self.countertop_texture,
            "countertop_texture_detail": self.countertop_texture_detail,
            # 音频
            "beat_alignment": self.beat_alignment,
            "bgm_style": self.bgm_style,
            "bgm_style_detail": self.bgm_style_detail,
            "bgm_mood": self.bgm_mood,
            "bgm_mood_detail": self.bgm_mood_detail,
            "bgm_instruments": self.bgm_instruments,
            "tempo_bpm": self.tempo_bpm,
            "percussive_ratio": self.percussive_ratio,
            "speech_ratio": self.speech_ratio,
            "key_signature": self.key_signature,
            # YOLO
            "yolo_available": self.yolo_available,
            "yolo_environment": self.yolo_environment,
            "yolo_style": self.yolo_style,
            "yolo_object_colors": self.yolo_object_colors,
            "yolo_object_materials": self.yolo_object_materials,
        }


@dataclass
class ReportInput(StepInput):
    """
    报告生成输入
    """
    video_metrics: List[VideoMetrics] = field(default_factory=list)
    consensus: Optional[ConsensusOutput] = None
    output_path: str = "style_report.docx"
    show_screenshots: bool = True


@dataclass
class ReportOutput(StepOutput):
    """
    报告生成输出
    """
    report_path: str = ""


# ============================================================================
# 步骤基类
# ============================================================================

InputT = TypeVar("InputT", bound=StepInput)
OutputT = TypeVar("OutputT", bound=StepOutput)


class PipelineStep(ABC, Generic[InputT, OutputT]):
    """
    流水线步骤基类
    
    每个步骤必须实现:
    - name: 步骤名称
    - description: 步骤描述
    - run(): 执行步骤的主方法
    
    使用示例:
        step = VisualAnalysisStep()
        input_data = VideoInput(video_path=Path("video.mp4"))
        output = step.run(input_data)
        print(f"分析完成: {output.camera_angle}")
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """步骤名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """步骤描述"""
        pass
    
    @abstractmethod
    def run(self, input_data: InputT) -> OutputT:
        """
        执行步骤
        
        Args:
            input_data: 步骤输入数据
            
        Returns:
            步骤输出数据
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
    
    def log_start(self, input_data: InputT) -> None:
        """记录步骤开始"""
        logger.info(f"[{self.name}] 开始执行...")
    
    def log_complete(self, output: OutputT) -> None:
        """记录步骤完成"""
        if output.success:
            logger.info(f"[{self.name}] ✓ 执行完成")
        else:
            logger.error(f"[{self.name}] ✗ 执行失败: {output.error_message}")
