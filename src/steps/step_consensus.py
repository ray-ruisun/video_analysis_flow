#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨视频共识计算步骤

负责从多个视频的分析结果中提取共识特征:
- 离散特征: 使用多数票规则
- 数值特征: 使用中位数计算
- 支持自动处理缺失值和异常值
"""

from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

from .base import (
    PipelineStep, 
    ConsensusInput, 
    ConsensusOutput, 
    VideoMetrics
)

# 导入原有的 beat alignment 计算
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics_audio import calculate_beat_alignment


class ConsensusStep(PipelineStep[ConsensusInput, ConsensusOutput]):
    """
    跨视频共识计算步骤
    
    输入: ConsensusInput (多个视频的分析结果列表)
    输出: ConsensusOutput (跨视频共识特征)
    
    使用示例:
        step = ConsensusStep()
        input_data = ConsensusInput(video_metrics=[vm1, vm2, vm3])
        output = step.run(input_data)
        print(f"共识镜头角度: {output.camera_angle}")
        print(f"共识 BGM 风格: {output.bgm_style}")
    """
    
    @property
    def name(self) -> str:
        return "共识计算"
    
    @property
    def description(self) -> str:
        return "从多个视频的分析结果中提取跨视频共识特征"
    
    def run(self, input_data: ConsensusInput) -> ConsensusOutput:
        """
        执行共识计算
        
        处理流程:
        1. 收集所有视频的各项指标
        2. 对离散特征使用多数票规则
        3. 对数值特征使用中位数计算
        4. 计算音画节拍对齐
        5. 输出结构化共识结果
        """
        self.log_start(input_data)
        
        try:
            video_metrics = input_data.video_metrics
            
            if not video_metrics:
                logger.warning("没有视频指标可供计算共识")
                return ConsensusOutput(success=True)
            
            # 转换为字典格式以便处理
            metrics_dicts = [vm.to_dict() for vm in video_metrics]
            
            # 提取各类共识
            visual_consensus = self._extract_visual_consensus(metrics_dicts)
            audio_consensus = self._extract_audio_consensus(metrics_dicts)
            yolo_consensus = self._extract_yolo_consensus(metrics_dicts)
            
            # 计算音画节拍对齐
            beat_alignment = self._calculate_beat_alignment(metrics_dicts)
            
            # 构建输出
            output = ConsensusOutput(
                success=True,
                # 视觉共识
                **visual_consensus,
                # 音频共识
                **audio_consensus,
                beat_alignment=beat_alignment,
                # YOLO 共识
                **yolo_consensus
            )
            
            self.log_complete(output)
            self._log_summary(output)
            
            return output
            
        except Exception as e:
            error_output = ConsensusOutput(
                success=False,
                error_message=str(e)
            )
            self.log_complete(error_output)
            raise
    
    def _majority_value(self, values: List[Any], min_count: int = 1) -> str:
        """
        多数票规则：返回最常见的值（不再返回 Varied）
        """
        if not values:
            return "N/A"
        # 过滤掉 None 和空字符串
        valid_values = [v for v in values if v is not None and v != ""]
        if not valid_values:
            return "N/A"
        counter = Counter(valid_values)
        most_common = counter.most_common(1)
        if most_common:
            return str(most_common[0][0])
        return "N/A"
    
    def _detailed_distribution(self, values: List[Any]) -> Dict[str, Any]:
        """
        返回详细的分布统计（不再返回 Varied）
        """
        if not values:
            return {"dominant": "N/A", "distribution": [], "all_values": []}
        
        # 过滤掉 None 和空字符串
        valid_values = [v for v in values if v is not None and v != ""]
        if not valid_values:
            return {"dominant": "N/A", "distribution": [], "all_values": []}
        
        counter = Counter(valid_values)
        total = len(valid_values)
        
        distribution = [
            {
                "value": str(val),
                "count": count,
                "percentage": round(count / total * 100, 1)
            }
            for val, count in counter.most_common()
        ]
        
        dominant = str(counter.most_common(1)[0][0]) if counter else "N/A"
        
        return {
            "dominant": dominant,
            "distribution": distribution,
            "all_values": [str(v) for v in counter.keys()],
            "unique_count": len(counter),
            "total_samples": total
        }
    
    def _median_value(self, values: List[Any]) -> Optional[float]:
        """
        对数值取中位数，并过滤 NaN/Inf 等异常值
        """
        valid = [
            v for v in values 
            if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v))
        ]
        return float(np.median(valid)) if valid else None
    
    def _extract_visual_consensus(self, metrics_dicts: List[Dict]) -> Dict[str, Any]:
        """提取视觉共识"""
        visual_entries = [v.get("visual", {}) for v in metrics_dicts if v.get("visual")]
        
        if not visual_entries:
            return {}
        
        # 收集各项指标
        camera_angles = [e.get("camera_angle") for e in visual_entries]
        focal_tendencies = [e.get("focal_length_tendency") for e in visual_entries]
        camera_motions = [e.get("camera_motion", {}).get("motion_type") for e in visual_entries]
        compositions = [e.get("composition", {}).get("rule_of_thirds") for e in visual_entries]
        hue_families = [e.get("hue_family") for e in visual_entries]
        saturations = [e.get("saturation_band") for e in visual_entries]
        brightnesses = [e.get("brightness_band") for e in visual_entries]
        contrasts = [e.get("contrast") for e in visual_entries]
        transition_types = [e.get("transition_type") for e in visual_entries]
        countertop_colors = [e.get("countertop_color") for e in visual_entries]
        countertop_textures = [e.get("countertop_texture") for e in visual_entries]
        
        # 数值指标
        ccts = [e.get("cct_mean") for e in visual_entries if e.get("cct_mean") is not None]
        
        # 场景分类
        scene_labels = []
        for entry in visual_entries:
            scenes = entry.get("scene_categories", [])
            if scenes:
                scene_labels.append(scenes[0].get("label"))
        
        # 计算每分钟剪辑数和平均镜头时长
        cuts_per_min = []
        avg_shot_lengths = []
        natural_ratios = []
        artificial_ratios = []
        
        for entry in visual_entries:
            duration = entry.get("duration") or 0
            cuts = entry.get("cuts") or 0
            if duration > 0 and cuts > 0:
                cuts_per_min.append(cuts / (duration / 60.0))
            
            shot_len = entry.get("avg_shot_length")
            if isinstance(shot_len, (int, float)) and not (np.isnan(shot_len) or np.isinf(shot_len)):
                avg_shot_lengths.append(shot_len)
            
            lighting = entry.get("lighting", {})
            if lighting.get("natural_light_ratio") is not None:
                natural_ratios.append(lighting["natural_light_ratio"])
            if lighting.get("artificial_light_ratio") is not None:
                artificial_ratios.append(lighting["artificial_light_ratio"])
        
        return {
            # 镜头分析 (含详细分布)
            "camera_angle": self._majority_value(camera_angles),
            "camera_angle_detail": self._detailed_distribution(camera_angles),
            "focal_length_tendency": self._majority_value(focal_tendencies),
            "focal_length_detail": self._detailed_distribution(focal_tendencies),
            "camera_motion": self._majority_value(camera_motions),
            "camera_motion_detail": self._detailed_distribution(camera_motions),
            
            # 构图分析 (含详细分布)
            "composition_rule_of_thirds": self._majority_value(compositions),
            "composition_detail": self._detailed_distribution(compositions),
            
            # 场景分类 (含详细分布)
            "scene_category": self._majority_value(scene_labels),
            "scene_category_detail": self._detailed_distribution(scene_labels),
            
            # 色彩分析 (含详细分布)
            "hue_family": self._majority_value(hue_families),
            "hue_detail": self._detailed_distribution(hue_families),
            "saturation": self._majority_value(saturations),
            "saturation_detail": self._detailed_distribution(saturations),
            "brightness": self._majority_value(brightnesses),
            "brightness_detail": self._detailed_distribution(brightnesses),
            "contrast": self._majority_value(contrasts),
            "contrast_detail": self._detailed_distribution(contrasts),
            
            # 数值指标
            "cct": self._median_value(ccts),
            "natural_light_ratio": self._median_value(natural_ratios),
            "artificial_light_ratio": self._median_value(artificial_ratios),
            "cuts_per_minute": self._median_value(cuts_per_min),
            "avg_shot_length": self._median_value(avg_shot_lengths),
            
            # 其他分类 (含详细分布)
            "transition_type": self._majority_value(transition_types),
            "transition_detail": self._detailed_distribution(transition_types),
            "countertop_color": self._majority_value(countertop_colors),
            "countertop_color_detail": self._detailed_distribution(countertop_colors),
            "countertop_texture": self._majority_value(countertop_textures),
            "countertop_texture_detail": self._detailed_distribution(countertop_textures),
        }
    
    def _extract_audio_consensus(self, metrics_dicts: List[Dict]) -> Dict[str, Any]:
        """提取音频共识"""
        audio_entries = [v.get("audio", {}) for v in metrics_dicts if v.get("audio")]
        
        if not audio_entries:
            return {
                "bgm_style": "N/A",
                "bgm_mood": "N/A",
                "bgm_instruments": [],
                "tempo_bpm": None,
                "percussive_ratio": None,
                "speech_ratio": None,
                "key_signature": "N/A",
            }
        
        # 收集指标
        bgm_styles = [e.get("bgm_style") for e in audio_entries]
        moods = [e.get("mood") for e in audio_entries]
        tempos = [e.get("tempo_bpm") for e in audio_entries]
        percussive_ratios = [e.get("percussive_ratio") for e in audio_entries]
        speech_ratios = [e.get("speech_ratio") for e in audio_entries]
        
        key_signatures = []
        instruments_list = []
        for entry in audio_entries:
            insts = entry.get("instruments", {}).get("detected_instruments", [])
            instruments_list.extend(insts)
            key_sig = entry.get("key_signature")
            if key_sig:
                key_signatures.append(key_sig)
        
        return {
            "bgm_style": self._majority_value(bgm_styles),
            "bgm_mood": self._majority_value(moods),
            "bgm_instruments": list(set(instruments_list)) if instruments_list else [],
            "tempo_bpm": self._median_value(tempos),
            "percussive_ratio": self._median_value(percussive_ratios),
            "speech_ratio": self._median_value(speech_ratios),
            "key_signature": self._majority_value(key_signatures),
        }
    
    def _extract_yolo_consensus(self, metrics_dicts: List[Dict]) -> Dict[str, Any]:
        """提取 YOLO 共识"""
        yolo_environments = []
        yolo_styles = []
        yolo_colors = {}
        yolo_materials = {}
        
        for v in metrics_dicts:
            yolo_data = v.get("yolo")
            if not yolo_data:
                continue
            env = yolo_data.get("environment", {})
            yolo_environments.append(env.get("environment_type", "Unknown"))
            yolo_styles.append(env.get("cooking_style", "Unknown"))
            
            colors_data = yolo_data.get("colors", {}).get("dominant_colors", {})
            for obj, color in colors_data.items():
                yolo_colors.setdefault(obj, []).append(color)
            
            materials_data = yolo_data.get("materials", {}).get("dominant_materials", {})
            for obj, material in materials_data.items():
                yolo_materials.setdefault(obj, []).append(material)
        
        consensus_colors = {
            obj: self._majority_value(colors, min_count=1) 
            for obj, colors in yolo_colors.items()
        }
        consensus_materials = {
            obj: self._majority_value(materials, min_count=1) 
            for obj, materials in yolo_materials.items()
        }
        
        return {
            "yolo_environment": self._majority_value(yolo_environments),
            "yolo_style": self._majority_value(yolo_styles),
            "yolo_object_colors": consensus_colors,
            "yolo_object_materials": consensus_materials,
        }
    
    def _calculate_beat_alignment(self, metrics_dicts: List[Dict]) -> Optional[float]:
        """计算音画节拍对齐"""
        beat_alignments = []
        
        for v in metrics_dicts:
            audio_data = v.get("audio")
            visual_data = v.get("visual")
            if not audio_data or not visual_data:
                continue
            
            cuts_count = visual_data.get("cuts", 0)
            duration = visual_data.get("duration")
            beat_times = audio_data.get("beat_times")
            
            if beat_times and cuts_count > 0 and duration and duration > 0:
                try:
                    alignment = calculate_beat_alignment(
                        duration,
                        cuts_count,
                        beat_times
                    )
                    beat_alignments.append(alignment)
                except ValueError:
                    pass
        
        return self._median_value(beat_alignments)
    
    def _log_summary(self, output: ConsensusOutput) -> None:
        """记录共识摘要"""
        cuts_per_min = output.cuts_per_minute or 0
        logger.info(
            f"  → 镜头: {output.camera_angle} | "
            f"色彩: {output.hue_family}/{output.saturation}/{output.brightness} | "
            f"场景: {output.scene_category} | "
            f"剪辑: {cuts_per_min:.2f} cuts/min"
        )
        logger.info(
            f"  → BGM: {output.bgm_style} | "
            f"情绪: {output.bgm_mood} | "
            f"调式: {output.key_signature}"
        )
