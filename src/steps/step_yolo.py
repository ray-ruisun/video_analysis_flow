#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 目标检测分析步骤

负责提取视频中的物体特征:
- 物体检测 (厨房物品、餐具、食材等)
- 环境分类 (厨房类型、烹饪风格)
- 物体颜色分析
- 物体材质分析
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from loguru import logger
import numpy as np

from .base import PipelineStep, StepInput, YOLOOutput

# 导入原有的 YOLO 分析函数和帧采样
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics_yolo import extract_full_yolo_metrics
from metrics_visual import sample_frames


@dataclass
class YOLOInput(StepInput):
    """
    YOLO 分析输入
    
    Attributes:
        video_path: 视频文件路径
        frames: 可选的预采样帧列表 (如果提供则直接使用)
        target_frames: 目标采样帧数 (默认 36)
        model_name: YOLO 模型名称
        enable_colors: 是否启用颜色分析
        enable_materials: 是否启用材质分析
    """
    video_path: Optional[Path] = None
    frames: Optional[List[np.ndarray]] = field(default=None, repr=False)
    target_frames: int = 36
    model_name: str = "yolov8n.pt"
    enable_colors: bool = True
    enable_materials: bool = True
    
    def __post_init__(self):
        """验证并转换路径类型"""
        if isinstance(self.video_path, str):
            self.video_path = Path(self.video_path)
    
    def validate(self) -> None:
        """验证输入有效性"""
        if self.frames is None and self.video_path is None:
            raise ValueError("必须提供 video_path 或 frames")
        if self.video_path is not None and not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")


class YOLOAnalysisStep(PipelineStep[YOLOInput, YOLOOutput]):
    """
    YOLO 目标检测分析步骤
    
    输入: YOLOInput (视频路径或帧列表、模型配置)
    输出: YOLOOutput (检测结果、环境分类、颜色、材质)
    
    使用示例:
        step = YOLOAnalysisStep()
        input_data = YOLOInput(
            video_path=Path("video.mp4"),
            enable_colors=True,
            enable_materials=True
        )
        output = step.run(input_data)
        print(f"环境类型: {output.environment.get('environment_type')}")
        print(f"检测物体数: {output.detection.get('unique_objects')}")
    """
    
    @property
    def name(self) -> str:
        return "YOLO 目标检测"
    
    @property
    def description(self) -> str:
        return "检测视频中的物体、分析环境类型、颜色和材质"
    
    def run(self, input_data: YOLOInput) -> YOLOOutput:
        """
        执行 YOLO 分析
        
        处理流程:
        1. 验证输入
        2. 采样帧 (如果未提供帧列表)
        3. 执行物体检测
        4. 分析环境、颜色、材质
        5. 转换为结构化输出
        """
        self.log_start(input_data)
        
        try:
            # 验证输入
            input_data.validate()
            
            # 获取帧
            if input_data.frames is not None:
                frames = input_data.frames
                logger.debug(f"使用提供的 {len(frames)} 帧")
            else:
                logger.debug(f"从视频采样 {input_data.target_frames} 帧...")
                frames, _, _, _ = sample_frames(
                    str(input_data.video_path),
                    target=input_data.target_frames
                )
            
            # 调用原有的 YOLO 分析函数
            raw_result = extract_full_yolo_metrics(
                frames,
                model_name=input_data.model_name,
                enable_colors=input_data.enable_colors,
                enable_materials=input_data.enable_materials
            )
            
            # 转换为结构化输出
            output = self._convert_to_output(raw_result)
            
            self.log_complete(output)
            self._log_summary(output, input_data.video_path)
            
            return output
            
        except Exception as e:
            error_output = YOLOOutput(
                success=False,
                error_message=str(e)
            )
            self.log_complete(error_output)
            raise
    
    def _convert_to_output(self, raw_result: dict) -> YOLOOutput:
        """将原始字典结果转换为 YOLOOutput"""
        return YOLOOutput(
            success=True,
            detection=raw_result.get("detection", {}),
            environment=raw_result.get("environment", {}),
            colors=raw_result.get("colors"),
            materials=raw_result.get("materials"),
        )
    
    def _log_summary(self, output: YOLOOutput, video_path: Optional[Path]) -> None:
        """记录分析摘要"""
        detection = output.detection
        environment = output.environment
        
        top_objects = detection.get("top_objects", [])
        top_preview = "、".join([f"{name}:{count}" for name, count in top_objects[:3]]) if top_objects else "无"
        
        video_name = video_path.name if video_path else "frames"
        logger.info(
            f"  → 环境: {environment.get('environment_type', 'N/A')}/"
            f"{environment.get('cooking_style', 'N/A')} | "
            f"物体: {detection.get('unique_objects', 0)}类/"
            f"{detection.get('total_detections', 0)}次 | "
            f"Top: {top_preview}"
        )
