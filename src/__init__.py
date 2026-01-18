#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Style Analysis Pipeline / 视频风格分析流水线

A modular pipeline for extracting stylistic patterns from video content.

模块化架构:
- steps/: 独立的分析步骤，每个步骤有明确的输入输出
- pipeline_runner: 流水线运行器，串联各个步骤
- metrics_*: 底层指标提取函数
"""

__version__ = "2.0.0"
__author__ = "Video Analysis Research"

# ============================================================================
# 模块化步骤 (推荐使用)
# ============================================================================

from .steps import (
    # 基础类型
    StepInput,
    StepOutput,
    PipelineStep,
    # 输入类型
    VideoInput,
    AudioInput,
    ASRInput,
    YOLOInput,
    ConsensusInput,
    ReportInput,
    # 输出类型
    VisualOutput,
    AudioOutput,
    ASROutput,
    YOLOOutput,
    ConsensusOutput,
    ReportOutput,
    # 聚合类型
    VideoMetrics,
    # 步骤类
    VisualAnalysisStep,
    AudioAnalysisStep,
    ASRAnalysisStep,
    YOLOAnalysisStep,
    ConsensusStep,
    ReportGenerationStep,
)

from .pipeline_runner import (
    PipelineConfig,
    PipelineResult,
    ModularPipeline,
    VideoStylePipeline,  # 兼容层
)

# ============================================================================
# 底层函数 (高级用户)
# ============================================================================

from .metrics_visual import extract_visual_metrics, sample_frames
from .metrics_audio import extract_audio_metrics, calculate_beat_alignment
from .metrics_asr import extract_full_asr_metrics, transcribe_audio
from .metrics_yolo import extract_full_yolo_metrics, detect_objects_in_frames
from .report_word import generate_word_report

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # 模块化步骤 - 基础类型
    "StepInput",
    "StepOutput",
    "PipelineStep",
    
    # 模块化步骤 - 输入类型
    "VideoInput",
    "AudioInput",
    "ASRInput",
    "YOLOInput",
    "ConsensusInput",
    "ReportInput",
    
    # 模块化步骤 - 输出类型
    "VisualOutput",
    "AudioOutput",
    "ASROutput",
    "YOLOOutput",
    "ConsensusOutput",
    "ReportOutput",
    
    # 聚合类型
    "VideoMetrics",
    
    # 模块化步骤 - 步骤类
    "VisualAnalysisStep",
    "AudioAnalysisStep",
    "ASRAnalysisStep",
    "YOLOAnalysisStep",
    "ConsensusStep",
    "ReportGenerationStep",
    
    # 流水线
    "PipelineConfig",
    "PipelineResult",
    "ModularPipeline",
    "VideoStylePipeline",  # 兼容层
    
    # 底层函数
    "extract_visual_metrics",
    "sample_frames",
    "extract_audio_metrics",
    "calculate_beat_alignment",
    "extract_full_asr_metrics",
    "transcribe_audio",
    "extract_full_yolo_metrics",
    "detect_objects_in_frames",
    "generate_word_report",
]
