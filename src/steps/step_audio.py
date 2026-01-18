#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频分析步骤

负责提取音频的特征:
- 节拍分析 (BPM、节拍时间点)
- 能量分析 (打击乐比例、平均能量)
- 频谱分析 (质心、平坦度、过零率)
- BGM 风格和情绪
- 乐器检测
- 调式识别
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from loguru import logger

from .base import PipelineStep, StepInput, AudioOutput

# 导入原有的音频分析函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics_audio import extract_audio_metrics


@dataclass
class AudioInput(StepInput):
    """
    音频分析输入
    
    Attributes:
        audio_path: 音频文件路径 (建议 22.05kHz mono wav)
    """
    audio_path: Path
    
    def __post_init__(self):
        """验证并转换路径类型"""
        if isinstance(self.audio_path, str):
            self.audio_path = Path(self.audio_path)
    
    def validate(self) -> None:
        """验证输入有效性"""
        if not self.audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {self.audio_path}")


class AudioAnalysisStep(PipelineStep[AudioInput, AudioOutput]):
    """
    音频分析步骤
    
    输入: AudioInput (音频路径)
    输出: AudioOutput (BPM、节拍、能量、BGM 风格、情绪等)
    
    使用示例:
        step = AudioAnalysisStep()
        input_data = AudioInput(audio_path=Path("audio.wav"))
        output = step.run(input_data)
        print(f"BPM: {output.tempo_bpm}")
        print(f"BGM 风格: {output.bgm_style}")
        print(f"情绪: {output.mood}")
    """
    
    @property
    def name(self) -> str:
        return "音频分析"
    
    @property
    def description(self) -> str:
        return "提取音频的节拍、能量、BGM 风格、情绪等特征"
    
    def run(self, input_data: AudioInput) -> AudioOutput:
        """
        执行音频分析
        
        处理流程:
        1. 验证输入
        2. 调用音频指标提取
        3. 转换为结构化输出
        """
        self.log_start(input_data)
        
        try:
            # 验证输入
            input_data.validate()
            
            # 调用原有的音频分析函数
            raw_result = extract_audio_metrics(str(input_data.audio_path))
            
            # 转换为结构化输出
            output = self._convert_to_output(raw_result)
            
            self.log_complete(output)
            self._log_summary(output, input_data.audio_path)
            
            return output
            
        except Exception as e:
            error_output = AudioOutput(
                success=False,
                error_message=str(e)
            )
            self.log_complete(error_output)
            raise
    
    def _convert_to_output(self, raw_result: dict) -> AudioOutput:
        """将原始字典结果转换为 AudioOutput"""
        return AudioOutput(
            success=True,
            tempo_bpm=raw_result.get("tempo_bpm", 0.0),
            beat_times=raw_result.get("beat_times", []),
            num_beats=raw_result.get("num_beats", 0),
            percussive_ratio=raw_result.get("percussive_ratio", 0.0),
            spectral_centroid=raw_result.get("spectral_centroid", 0.0),
            spectral_flatness=raw_result.get("spectral_flatness", 0.0),
            zero_crossing_rate=raw_result.get("zero_crossing_rate", 0.0),
            mean_energy=raw_result.get("mean_energy", 0.0),
            energy_variance=raw_result.get("energy_variance", 0.0),
            spectral_rolloff=raw_result.get("spectral_rolloff", 0.0),
            speech_ratio=raw_result.get("speech_ratio", 0.0),
            bgm_style=raw_result.get("bgm_style", "Unknown"),
            mood=raw_result.get("mood", "Unknown"),
            mood_tags=raw_result.get("mood_tags", []),
            instruments=raw_result.get("instruments", {}),
            key_signature=raw_result.get("key_signature"),
        )
    
    def _log_summary(self, output: AudioOutput, audio_path: Path) -> None:
        """记录分析摘要"""
        logger.info(
            f"  → BGM: {output.bgm_style} | "
            f"情绪: {output.mood} | "
            f"BPM: {output.tempo_bpm:.1f} | "
            f"打击乐比例: {output.percussive_ratio:.2f} | "
            f"调式: {output.key_signature or 'N/A'}"
        )
