#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉分析步骤

负责提取视频的视觉特征:
- 色彩分析 (色调、饱和度、亮度、对比度)
- 镜头分析 (角度、焦距、运动)
- 构图分析 (三分法、平衡)
- 场景分类 (Places365)
- 剪辑分析 (镜头切换、平均镜头时长)
- 光线分析 (自然光/人工光、色温)
- 台面分析 (颜色、纹理)
"""

from pathlib import Path
from loguru import logger

from .base import PipelineStep, VideoInput, VisualOutput

# 导入原有的视觉分析函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics_visual import extract_visual_metrics


class VisualAnalysisStep(PipelineStep[VideoInput, VisualOutput]):
    """
    视觉分析步骤
    
    输入: VideoInput (视频路径、工作目录、截图模式)
    输出: VisualOutput (色彩、镜头、构图、场景、剪辑等分析结果)
    
    使用示例:
        step = VisualAnalysisStep()
        input_data = VideoInput(
            video_path=Path("video.mp4"),
            work_dir=Path("work"),
            frame_mode="edge"
        )
        output = step.run(input_data)
        print(f"镜头角度: {output.camera_angle}")
        print(f"色调: {output.hue_family}")
        print(f"剪辑数: {output.cuts}")
    """
    
    @property
    def name(self) -> str:
        return "视觉分析"
    
    @property
    def description(self) -> str:
        return "提取视频的色彩、镜头、构图、场景、剪辑等视觉特征"
    
    def run(self, input_data: VideoInput) -> VisualOutput:
        """
        执行视觉分析
        
        处理流程:
        1. 验证输入
        2. 准备输出目录
        3. 调用视觉指标提取
        4. 转换为结构化输出
        """
        self.log_start(input_data)
        
        try:
            # 验证输入
            input_data.validate()
            
            # 准备输出目录
            frames_dir = input_data.work_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # 调用原有的视觉分析函数
            raw_result = extract_visual_metrics(
                str(input_data.video_path),
                str(frames_dir),
                input_data.frame_mode,
                target_frames=input_data.target_frames,
                scene_threshold=input_data.scene_threshold
            )
            
            # 转换为结构化输出
            output = self._convert_to_output(raw_result)
            
            self.log_complete(output)
            self._log_summary(output, input_data.video_path)
            
            return output
            
        except Exception as e:
            error_output = VisualOutput(
                success=False,
                error_message=str(e)
            )
            self.log_complete(error_output)
            raise
    
    def _convert_to_output(self, raw_result: dict) -> VisualOutput:
        """将原始字典结果转换为 VisualOutput"""
        return VisualOutput(
            success=True,
            # 基础信息
            fps=raw_result.get("fps", 0.0),
            total_frames=raw_result.get("total_frames", 0),
            duration=raw_result.get("duration", 0.0),
            sampled_frames=raw_result.get("sampled_frames", 0),
            # 色彩分析 (含详细分布)
            hue_family=raw_result.get("hue_family", "Unknown"),
            hue_detail=raw_result.get("hue_detail", {}),
            saturation_band=raw_result.get("saturation_band", "Unknown"),
            saturation_detail=raw_result.get("saturation_detail", {}),
            brightness_band=raw_result.get("brightness_band", "Unknown"),
            brightness_detail=raw_result.get("brightness_detail", {}),
            contrast=raw_result.get("contrast", "Unknown"),
            contrast_detail=raw_result.get("contrast_detail", {}),
            # 色温
            cct_mean=raw_result.get("cct_mean"),
            cct_std=raw_result.get("cct_std"),
            cct_range=raw_result.get("cct_range"),
            # 镜头分析 (含详细分布)
            camera_angle=raw_result.get("camera_angle", "Unknown"),
            camera_angle_detail=raw_result.get("camera_angle_detail", {}),
            focal_length_tendency=raw_result.get("focal_length_tendency", "Unknown"),
            camera_motion=raw_result.get("camera_motion", {}),
            # 构图分析
            composition=raw_result.get("composition", {}),
            # 场景分类
            scene_categories=raw_result.get("scene_categories", []),
            # 剪辑分析
            cuts=raw_result.get("cuts", 0),
            cut_timestamps=raw_result.get("cut_timestamps", []),
            avg_shot_length=raw_result.get("avg_shot_length", 0.0),
            transition_type=raw_result.get("transition_type", "Unknown"),
            # 台面分析 (含详细分布)
            countertop_color=raw_result.get("countertop_color", "Unknown"),
            countertop_color_detail=raw_result.get("countertop_color_detail", {}),
            countertop_texture=raw_result.get("countertop_texture", "Unknown"),
            countertop_texture_detail=raw_result.get("countertop_texture_detail", {}),
            # 光线分析
            lighting=raw_result.get("lighting", {}),
            # 截图路径
            contact_sheet=raw_result.get("contact_sheet"),
        )
    
    def _log_summary(self, output: VisualOutput, video_path: Path) -> None:
        """记录分析摘要"""
        scene_label = "N/A"
        if output.scene_categories:
            scene_label = output.scene_categories[0].get("label", "N/A")
        
        logger.info(
            f"  → 镜头: {output.camera_angle}/{output.focal_length_tendency} | "
            f"色彩: {output.hue_family}/{output.saturation_band}/{output.brightness_band} | "
            f"场景: {scene_label} | "
            f"剪辑: {output.cuts} cuts, avg {output.avg_shot_length:.2f}s"
        )
