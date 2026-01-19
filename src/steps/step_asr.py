#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR (自动语音识别) 分析步骤

负责提取语音特征:
- 语音转录 (Whisper)
- 语速分析 (每秒/每分钟词数)
- 口头禅检测 (高频短语)
- 停顿分析
- 韵律分析 (使用 librosa)
- 情感分析 (使用 HuggingFace wav2vec2)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from loguru import logger

from .base import PipelineStep, StepInput, ASROutput

# 导入原有的 ASR 分析函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics_asr import extract_full_asr_metrics


@dataclass
class ASRInput(StepInput):
    """
    ASR 分析输入
    
    Attributes:
        audio_path: 音频文件路径
        language: 语言代码 (默认 "en")
        model_size: Whisper 模型大小 ("tiny", "base", "small", "medium", "large", "large-v3-turbo")
        beam_size: Whisper beam search 大小
        enable_prosody: 是否启用韵律分析 (使用 librosa)
        enable_emotion: 是否启用情感分析 (使用 HuggingFace)
    """
    audio_path: Path
    language: str = "en"
    model_size: str = "large-v3-turbo"
    beam_size: int = 5
    enable_prosody: bool = True
    enable_emotion: bool = True
    
    def __post_init__(self):
        """验证并转换路径类型"""
        if isinstance(self.audio_path, str):
            self.audio_path = Path(self.audio_path)
    
    def validate(self) -> None:
        """验证输入有效性"""
        if not self.audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {self.audio_path}")


class ASRAnalysisStep(PipelineStep[ASRInput, ASROutput]):
    """
    ASR 分析步骤
    
    输入: ASRInput (音频路径、语言、模型大小、可选功能开关)
    输出: ASROutput (转录文本、语速、口头禅、停顿、韵律、情感等)
    
    使用示例:
        step = ASRAnalysisStep()
        input_data = ASRInput(
            audio_path=Path("audio.wav"),
            language="en",
            enable_prosody=True,
            enable_emotion=True
        )
        output = step.run(input_data)
        print(f"转录文本: {output.text[:100]}...")
        print(f"语速: {output.words_per_minute:.1f} wpm")
        print(f"口头禅: {output.catchphrases}")
    """
    
    @property
    def name(self) -> str:
        return "ASR 分析"
    
    @property
    def description(self) -> str:
        return "语音转录、语速分析、口头禅检测、韵律与情感分析"
    
    def run(self, input_data: ASRInput) -> ASROutput:
        """
        执行 ASR 分析
        
        处理流程:
        1. 验证输入
        2. 调用 Whisper 转录
        3. 分析语速、口头禅、停顿
        4. 可选: 韵律分析、情感分析
        5. 转换为结构化输出
        """
        self.log_start(input_data)
        
        try:
            # 验证输入
            input_data.validate()
            
            # 调用原有的 ASR 分析函数
            raw_result = extract_full_asr_metrics(
                str(input_data.audio_path),
                language=input_data.language,
                model_size=input_data.model_size,
                enable_prosody=input_data.enable_prosody,
                enable_emotion=input_data.enable_emotion
            )
            
            # 转换为结构化输出
            output = self._convert_to_output(raw_result)
            
            self.log_complete(output)
            self._log_summary(output, input_data.audio_path)
            
            return output
            
        except Exception as e:
            error_output = ASROutput(
                success=False,
                error_message=str(e)
            )
            self.log_complete(error_output)
            raise
    
    def _convert_to_output(self, raw_result: dict) -> ASROutput:
        """将原始字典结果转换为 ASROutput"""
        return ASROutput(
            success=True,
            text=raw_result.get("text", ""),
            implementation=raw_result.get("implementation", ""),
            num_words=raw_result.get("num_words", 0),
            words_per_second=raw_result.get("words_per_second", 0.0),
            words_per_minute=raw_result.get("words_per_minute", 0.0),
            pace=raw_result.get("pace", "Unknown"),
            catchphrases=raw_result.get("catchphrases", []),
            num_pauses=raw_result.get("num_pauses", 0),
            pause_style=raw_result.get("pause_style", "Unknown"),
            prosody=raw_result.get("prosody"),
            emotion=raw_result.get("emotion"),
        )
    
    def _log_summary(self, output: ASROutput, audio_path: Path) -> None:
        """记录分析摘要"""
        catchphrase_preview = "、".join(output.catchphrases[:3]) if output.catchphrases else "无"
        logger.info(
            f"  → 语速: {output.words_per_second:.2f} w/s ({output.words_per_minute:.1f} wpm) | "
            f"节奏: {output.pace} | "
            f"词数: {output.num_words} | "
            f"口头禅: {catchphrase_preview}"
        )
