#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化流水线步骤包

提供独立的、可组合的分析步骤，每个步骤有明确的输入输出定义。

使用示例:
    # 单独使用视觉分析步骤
    from steps import VisualAnalysisStep, VideoInput
    from pathlib import Path
    
    step = VisualAnalysisStep()
    input_data = VideoInput(video_path=Path("video.mp4"))
    output = step.run(input_data)
    print(f"镜头角度: {output.camera_angle}")

    # 使用音频分析步骤
    from steps import AudioAnalysisStep, AudioInput
    
    step = AudioAnalysisStep()
    input_data = AudioInput(audio_path=Path("audio.wav"))
    output = step.run(input_data)
    print(f"BPM: {output.tempo_bpm}")
"""

from .base import (
    # 基础类型
    StepInput,
    StepOutput,
    PipelineStep,
    # 输入/输出类型
    VideoInput,
    VisualOutput,
    AudioOutput,
    ASROutput,
    YOLOOutput,
    ConsensusInput,
    ConsensusOutput,
    ReportInput,
    ReportOutput,
    VideoMetrics,
)
from .step_visual import VisualAnalysisStep
from .step_audio import AudioAnalysisStep, AudioInput
from .step_asr import ASRAnalysisStep, ASRInput
from .step_yolo import YOLOAnalysisStep, YOLOInput
from .step_consensus import ConsensusStep
from .step_report import ReportGenerationStep

__all__ = [
    # 基础类型
    "StepInput",
    "StepOutput",
    "PipelineStep",
    # 输入类型
    "VideoInput",
    "AudioInput",
    "ASRInput",
    "YOLOInput",
    "ConsensusInput",
    "ReportInput",
    # 输出类型
    "VisualOutput",
    "AudioOutput",
    "ASROutput",
    "YOLOOutput",
    "ConsensusOutput",
    "ReportOutput",
    # 聚合类型
    "VideoMetrics",
    # 步骤类
    "VisualAnalysisStep",
    "AudioAnalysisStep",
    "ASRAnalysisStep",
    "YOLOAnalysisStep",
    "ConsensusStep",
    "ReportGenerationStep",
]
