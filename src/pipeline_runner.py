#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化流水线运行器

设计原则:
- 每个步骤独立执行，有明确的输入和输出
- 支持选择性执行模块
- 提供步骤级别的调试钩子
- 保持向后兼容原有的调用方式
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Sequence
from loguru import logger

from steps.base import (
    VideoInput,
    VideoMetrics,
    VisualOutput,
    AudioOutput,
    ASROutput,
    YOLOOutput,
    ConsensusInput,
    ConsensusOutput,
    ReportInput,
    ReportOutput,
)
from steps.step_visual import VisualAnalysisStep
from steps.step_audio import AudioAnalysisStep, AudioInput
from steps.step_asr import ASRAnalysisStep, ASRInput
from steps.step_yolo import YOLOAnalysisStep, YOLOInput
from steps.step_consensus import ConsensusStep
from steps.step_report import ReportGenerationStep

from utils import setup_logger, log_execution_time

# 初始化日志器
logger = setup_logger()

# 类型定义
StepHook = Callable[[str, Dict[str, Any]], None]

# 有效模块列表
VALID_MODULES = ["visual", "audio", "asr", "yolo"]

# 模块显示名称
MODULE_DISPLAY_NAMES = {
    "visual": "视觉分析",
    "audio": "音频分析",
    "asr": "ASR 分析",
    "yolo": "YOLO 检测"
}


@dataclass
class PipelineConfig:
    """
    流水线配置
    
    Attributes:
        video_paths: 视频文件路径列表
        audio_paths: 音频文件路径列表 (可选)
        work_dir: 工作目录
        modules: 要执行的模块列表
        frame_mode: 截图模式
        output_report: 输出报告路径
        asr_language: ASR 语言
        asr_model_size: ASR 模型大小
        enable_prosody: 是否启用韵律分析
        enable_emotion: 是否启用情感分析
        yolo_model: YOLO 模型名称
        enable_yolo_colors: 是否启用 YOLO 颜色分析
        enable_yolo_materials: 是否启用 YOLO 材质分析
    """
    video_paths: List[Path] = field(default_factory=list)
    audio_paths: Optional[List[Path]] = None
    work_dir: Path = field(default_factory=lambda: Path("work"))
    modules: List[str] = field(default_factory=lambda: ["visual", "audio"])
    frame_mode: str = "edge"
    output_report: str = "style_report.docx"
    
    # ASR 配置
    asr_language: str = "en"
    asr_model_size: str = "small"
    enable_prosody: bool = True
    enable_emotion: bool = True
    
    # YOLO 配置
    yolo_model: str = "yolov8n.pt"
    enable_yolo_colors: bool = True
    enable_yolo_materials: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        # 转换路径类型
        self.video_paths = [Path(p) if isinstance(p, str) else p for p in self.video_paths]
        if self.audio_paths:
            self.audio_paths = [Path(p) if isinstance(p, str) else p for p in self.audio_paths]
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        
        # 规范化模块列表
        self.modules = self._normalize_modules(self.modules)
    
    def _normalize_modules(self, modules: List[str]) -> List[str]:
        """规范化模块列表"""
        normalized = []
        seen = set()
        for module in modules:
            m = module.strip().lower()
            if m and m not in seen:
                if m not in VALID_MODULES:
                    raise ValueError(f"无效的模块: {m}. 有效选项: {', '.join(VALID_MODULES)}")
                normalized.append(m)
                seen.add(m)
        return normalized


@dataclass
class PipelineResult:
    """
    流水线执行结果
    
    Attributes:
        video_metrics: 各视频的分析结果
        consensus: 跨视频共识
        report_path: 报告文件路径
    """
    video_metrics: List[VideoMetrics] = field(default_factory=list)
    consensus: Optional[ConsensusOutput] = None
    report_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（兼容旧版接口）"""
        return {
            "video_metrics": [vm.to_dict() for vm in self.video_metrics],
            "consensus": self.consensus.to_dict() if self.consensus else {},
            "report_path": self.report_path
        }


class ModularPipeline:
    """
    模块化流水线
    
    支持:
    - 选择性执行模块 (visual, audio, asr, yolo)
    - 步骤级调试钩子
    - 清晰的输入输出追踪
    
    使用示例 (完整流程):
        config = PipelineConfig(
            video_paths=[Path("v1.mp4"), Path("v2.mp4"), Path("v3.mp4")],
            audio_paths=[Path("a1.wav"), Path("a2.wav"), Path("a3.wav")],
            modules=["visual", "audio", "asr", "yolo"],
            output_report="report.docx"
        )
        
        pipeline = ModularPipeline(config)
        result = pipeline.run()
        print(f"报告已生成: {result.report_path}")
    
    使用示例 (单步执行):
        pipeline = ModularPipeline(config)
        
        # Step 1: 视觉分析
        visual_output = pipeline.run_visual_analysis(Path("video.mp4"))
        print(f"镜头角度: {visual_output.camera_angle}")
        
        # Step 2: 音频分析
        audio_output = pipeline.run_audio_analysis(Path("audio.wav"))
        print(f"BPM: {audio_output.tempo_bpm}")
    """
    
    def __init__(self, config: PipelineConfig):
        """
        初始化流水线
        
        Args:
            config: 流水线配置
        """
        self.config = config
        self.module_set = set(config.modules)
        
        # 验证输入
        self._validate_inputs()
        
        # 创建工作目录
        config.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化步骤实例
        self._visual_step = VisualAnalysisStep()
        self._audio_step = AudioAnalysisStep()
        self._asr_step = ASRAnalysisStep()
        self._yolo_step = YOLOAnalysisStep()
        self._consensus_step = ConsensusStep()
        self._report_step = ReportGenerationStep()
        
        logger.info(f"流水线已初始化 | 视频数: {len(config.video_paths)} | 模块: {', '.join(config.modules)}")
    
    def _validate_inputs(self) -> None:
        """验证输入"""
        config = self.config
        
        # 验证视频
        if len(config.video_paths) != 3:
            raise ValueError("必须提供恰好 3 个视频文件")
        
        for vpath in config.video_paths:
            if not vpath.exists():
                raise FileNotFoundError(f"视频文件不存在: {vpath}")
        
        # 验证音频 (如果需要)
        needs_audio = any(m in self.module_set for m in ("audio", "asr"))
        if needs_audio:
            if not config.audio_paths:
                raise ValueError("audio/asr 模块需要提供音频文件 (-a/--audios)")
            if len(config.audio_paths) != 3:
                raise ValueError("必须提供恰好 3 个音频文件")
            for apath in config.audio_paths:
                if not apath.exists():
                    raise FileNotFoundError(f"音频文件不存在: {apath}")
        elif config.audio_paths:
            logger.warning("提供了音频文件但 audio/asr 模块未启用，文件将被忽略")
    
    # =========================================================================
    # 单步执行方法 - 可独立调用
    # =========================================================================
    
    def run_visual_analysis(
        self, 
        video_path: Path, 
        frame_mode: Optional[str] = None
    ) -> VisualOutput:
        """
        执行视觉分析 (单步)
        
        输入: 视频路径
        输出: VisualOutput (色彩、镜头、构图、场景、剪辑等)
        
        Args:
            video_path: 视频文件路径
            frame_mode: 截图模式 (可选，默认使用配置值)
            
        Returns:
            VisualOutput: 视觉分析结果
        """
        input_data = VideoInput(
            video_path=video_path,
            work_dir=self.config.work_dir,
            frame_mode=frame_mode or self.config.frame_mode
        )
        return self._visual_step.run(input_data)
    
    def run_audio_analysis(self, audio_path: Path) -> AudioOutput:
        """
        执行音频分析 (单步)
        
        输入: 音频路径
        输出: AudioOutput (BPM、节拍、能量、BGM 风格、情绪等)
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            AudioOutput: 音频分析结果
        """
        input_data = AudioInput(audio_path=audio_path)
        return self._audio_step.run(input_data)
    
    def run_asr_analysis(
        self, 
        audio_path: Path,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        enable_prosody: Optional[bool] = None,
        enable_emotion: Optional[bool] = None
    ) -> ASROutput:
        """
        执行 ASR 分析 (单步)
        
        输入: 音频路径
        输出: ASROutput (转录文本、语速、口头禅、停顿、韵律、情感等)
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码 (可选，默认使用配置值)
            model_size: 模型大小 (可选，默认使用配置值)
            enable_prosody: 是否启用韵律分析 (可选，默认使用配置值)
            enable_emotion: 是否启用情感分析 (可选，默认使用配置值)
            
        Returns:
            ASROutput: ASR 分析结果
        """
        input_data = ASRInput(
            audio_path=audio_path,
            language=language or self.config.asr_language,
            model_size=model_size or self.config.asr_model_size,
            enable_prosody=enable_prosody if enable_prosody is not None else self.config.enable_prosody,
            enable_emotion=enable_emotion if enable_emotion is not None else self.config.enable_emotion
        )
        return self._asr_step.run(input_data)
    
    def run_yolo_analysis(
        self, 
        video_path: Path,
        model_name: Optional[str] = None,
        enable_colors: Optional[bool] = None,
        enable_materials: Optional[bool] = None
    ) -> YOLOOutput:
        """
        执行 YOLO 分析 (单步)
        
        输入: 视频路径
        输出: YOLOOutput (检测结果、环境分类、颜色、材质)
        
        Args:
            video_path: 视频文件路径
            model_name: YOLO 模型名称 (可选，默认使用配置值)
            enable_colors: 是否启用颜色分析 (可选，默认使用配置值)
            enable_materials: 是否启用材质分析 (可选，默认使用配置值)
            
        Returns:
            YOLOOutput: YOLO 分析结果
        """
        input_data = YOLOInput(
            video_path=video_path,
            model_name=model_name or self.config.yolo_model,
            enable_colors=enable_colors if enable_colors is not None else self.config.enable_yolo_colors,
            enable_materials=enable_materials if enable_materials is not None else self.config.enable_yolo_materials
        )
        return self._yolo_step.run(input_data)
    
    def run_consensus(self, video_metrics: List[VideoMetrics]) -> ConsensusOutput:
        """
        执行共识计算 (单步)
        
        输入: 多个视频的分析结果
        输出: ConsensusOutput (跨视频共识特征)
        
        Args:
            video_metrics: 视频分析结果列表
            
        Returns:
            ConsensusOutput: 共识计算结果
        """
        input_data = ConsensusInput(video_metrics=video_metrics)
        return self._consensus_step.run(input_data)
    
    def run_report_generation(
        self,
        video_metrics: List[VideoMetrics],
        consensus: ConsensusOutput,
        output_path: Optional[str] = None,
        show_screenshots: Optional[bool] = None
    ) -> ReportOutput:
        """
        执行报告生成 (单步)
        
        输入: 视频分析结果、共识、输出路径
        输出: ReportOutput (报告文件路径)
        
        Args:
            video_metrics: 视频分析结果列表
            consensus: 共识计算结果
            output_path: 输出路径 (可选，默认使用配置值)
            show_screenshots: 是否显示截图 (可选)
            
        Returns:
            ReportOutput: 报告生成结果
        """
        input_data = ReportInput(
            video_metrics=video_metrics,
            consensus=consensus,
            output_path=output_path or self.config.output_report,
            show_screenshots=show_screenshots if show_screenshots is not None else (self.config.frame_mode != "off")
        )
        return self._report_step.run(input_data)
    
    # =========================================================================
    # 单视频完整分析
    # =========================================================================
    
    def analyze_single_video(
        self,
        video_path: Path,
        audio_path: Optional[Path] = None,
        step_hook: Optional[StepHook] = None
    ) -> VideoMetrics:
        """
        分析单个视频的所有模块
        
        按配置的模块顺序执行:
        1. visual -> VisualOutput
        2. audio -> AudioOutput (需要音频)
        3. asr -> ASROutput (需要音频)
        4. yolo -> YOLOOutput
        
        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径 (可选)
            step_hook: 步骤完成回调
            
        Returns:
            VideoMetrics: 单个视频的完整分析结果
        """
        logger.info(f"分析视频: {video_path.name}")
        
        metrics = VideoMetrics(path=str(video_path))
        
        # 按模块顺序执行
        for i, module_name in enumerate(self.config.modules, 1):
            display_name = MODULE_DISPLAY_NAMES.get(module_name, module_name)
            logger.info(f"[{i}/{len(self.config.modules)}] 执行 {display_name}...")
            
            try:
                if module_name == "visual":
                    metrics.visual = self.run_visual_analysis(video_path)
                    
                elif module_name == "audio":
                    if not audio_path:
                        raise ValueError(f"音频模块需要音频文件")
                    metrics.audio = self.run_audio_analysis(audio_path)
                    
                elif module_name == "asr":
                    if not audio_path:
                        raise ValueError(f"ASR 模块需要音频文件")
                    metrics.asr = self.run_asr_analysis(audio_path)
                    
                elif module_name == "yolo":
                    metrics.yolo = self.run_yolo_analysis(video_path)
                
                # 调用回调
                if step_hook:
                    step_hook("module", {
                        "module": module_name,
                        "video_path": str(video_path),
                        "step_index": i,
                        "step_total": len(self.config.modules)
                    })
                    
            except Exception as e:
                logger.error(f"模块 {module_name} 执行失败: {e}")
                raise
        
        return metrics
    
    # =========================================================================
    # 完整流水线执行
    # =========================================================================
    
    @log_execution_time
    def run(self, step_hook: Optional[StepHook] = None) -> PipelineResult:
        """
        运行完整流水线
        
        执行流程:
        1. 逐视频分析 (按模块顺序)
        2. 计算跨视频共识
        3. 生成 Word 报告
        
        Args:
            step_hook: 步骤完成回调 (用于调试)
            
        Returns:
            PipelineResult: 完整流水线结果
        """
        logger.info("=" * 70)
        logger.info("开始视频风格分析流水线")
        logger.info("=" * 70)
        
        result = PipelineResult()
        
        # Step 1: 逐视频分析
        logger.info("[阶段 1/3] 视频分析")
        for i, video_path in enumerate(self.config.video_paths, 1):
            logger.info(f"[{i}/3] 处理 {video_path.name}...")
            
            audio_path = self.config.audio_paths[i-1] if self.config.audio_paths else None
            
            metrics = self.analyze_single_video(
                video_path,
                audio_path,
                step_hook
            )
            result.video_metrics.append(metrics)
            
            logger.info(f"  ✓ {video_path.name} 分析完成")
        
        # Step 2: 共识计算
        logger.info("[阶段 2/3] 共识计算")
        result.consensus = self.run_consensus(result.video_metrics)
        
        if step_hook:
            step_hook("consensus", {"consensus": result.consensus.to_dict()})
        
        # Step 3: 报告生成
        logger.info(f"[阶段 3/3] 生成报告: {self.config.output_report}")
        report_output = self.run_report_generation(
            result.video_metrics,
            result.consensus
        )
        result.report_path = report_output.report_path
        
        if step_hook:
            step_hook("report", {"report_path": result.report_path})
        
        # 输出总览
        self._log_summary(result)
        
        return result
    
    def _log_summary(self, result: PipelineResult) -> None:
        """输出总览"""
        logger.info("=" * 70)
        logger.info("分析完成!")
        logger.info("=" * 70)
        logger.info(f"  视频数: {len(result.video_metrics)}")
        logger.info(f"  执行模块: {', '.join(self.config.modules)}")
        
        if result.consensus:
            c = result.consensus
            if "visual" in self.module_set:
                cuts_per_min = c.cuts_per_minute or 0
                logger.info(f"  镜头: {c.camera_angle}")
                logger.info(f"  色彩: {c.hue_family}, {c.saturation}, {c.brightness}")
                logger.info(f"  场景: {c.scene_category}")
                logger.info(f"  剪辑: {cuts_per_min:.2f} cuts/min")
            if "audio" in self.module_set:
                logger.info(f"  BGM: {c.bgm_style} | 情绪: {c.bgm_mood} | 调式: {c.key_signature}")
        
        if result.report_path:
            logger.info(f"  报告: {result.report_path}")


# ============================================================================
# 兼容层 - 保持与原有 VideoStylePipeline 的兼容性
# ============================================================================

class VideoStylePipeline:
    """
    兼容层: 保持与原有 API 的兼容性
    
    使用方式与原有相同:
        pipeline = VideoStylePipeline(
            video_paths=['v1.mp4', 'v2.mp4', 'v3.mp4'],
            audio_paths=['a1.wav', 'a2.wav', 'a3.wav'],
            work_dir='work',
            modules=['visual', 'audio', 'asr', 'yolo']
        )
        results = pipeline.run(output_report='report.docx')
    """
    
    def __init__(
        self,
        video_paths: List[str],
        audio_paths: Optional[List[str]] = None,
        work_dir: str = "work",
        modules: Optional[Sequence[str]] = None
    ):
        """初始化 (兼容原有接口)"""
        config = PipelineConfig(
            video_paths=[Path(p) for p in video_paths],
            audio_paths=[Path(p) for p in audio_paths] if audio_paths else None,
            work_dir=Path(work_dir),
            modules=list(modules) if modules else ["visual", "audio", "asr", "yolo"]
        )
        self._pipeline = ModularPipeline(config)
    
    def run(
        self,
        frame_mode: str = "edge",
        output_report: str = "style_report.docx",
        step_hook: Optional[StepHook] = None
    ) -> Dict:
        """运行流水线 (兼容原有接口)"""
        # 更新配置
        self._pipeline.config.frame_mode = frame_mode
        self._pipeline.config.output_report = output_report
        
        # 执行
        result = self._pipeline.run(step_hook)
        
        # 转换为原有格式
        return result.to_dict()
