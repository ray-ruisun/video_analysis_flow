#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频风格分析流水线 - 主流程编排层

设计说明:
- 该文件只负责流程编排与结果汇总，不实现具体算法；
- 视觉/音频/ASR/YOLO 各模块独立，按模块顺序逐步执行；
- 每个组件完成后输出摘要日志，便于逐步定位与对比。
"""

from pathlib import Path
from typing import List, Dict, Optional, Sequence, Callable, Any
from collections import Counter

import numpy as np
from loguru import logger

from utils import setup_logger, log_execution_time, format_value
from metrics_visual import extract_visual_metrics
from metrics_audio import extract_audio_metrics, calculate_beat_alignment
from metrics_asr import extract_full_asr_metrics
from metrics_yolo import extract_full_yolo_metrics
from report_word import generate_word_report

# 初始化日志器
logger = setup_logger()

VALID_MODULES = ["visual", "audio", "asr", "yolo"]
MODULE_DISPLAY_NAMES = {
    "visual": "视觉",
    "audio": "音频",
    "asr": "ASR",
    "yolo": "YOLO"
}
StepHook = Callable[[str, Dict[str, Any]], None]


class VideoStylePipeline:
    """
    视频风格分析流水线主类。

    使用示例:
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
        """
        初始化流水线。

        设计说明:
        - 仅在此处做输入校验与目录准备，避免后续步骤重复判断；
        - 模块列表在初始化阶段归一化，保证执行顺序可控。

        Args:
            video_paths: 3 个视频路径
            audio_paths: 可选 3 个音频路径 (22.05kHz mono wav)
            work_dir: 临时工作目录
        """
        self.video_paths = [Path(p) for p in video_paths]
        self.audio_paths = [Path(p) for p in audio_paths] if audio_paths else None
        self.work_dir = Path(work_dir)
        self.modules = self._normalize_modules(modules)
        self.module_set = set(self.modules)
        
        # 校验输入
        self._validate_inputs()
        
        # 创建工作目录
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.work_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        
        logger.info(f"Pipeline initialized with {len(self.video_paths)} videos | Modules: {', '.join(self.modules)}")

    def _normalize_modules(self, modules: Optional[Sequence[str]]) -> List[str]:
        """
        归一化模块列表，输出去重且有序的模块序列。

        设计说明:
        - 保留输入顺序，确保执行顺序可预期；
        - 过滤非法模块并显式报错，避免静默忽略。
        """
        if modules:
            normalized: List[str] = []
            seen = set()
            for module in modules:
                token = module.strip().lower()
                if not token:
                    continue
                if token not in VALID_MODULES:
                    raise ValueError(f"Invalid module '{module}'. Valid options: {', '.join(VALID_MODULES)}")
                if token not in seen:
                    normalized.append(token)
                    seen.add(token)
            if not normalized:
                raise ValueError("No valid modules specified.")
            return normalized
        # 默认顺序
        return ["visual", "audio", "asr", "yolo"]
    
    def _validate_inputs(self):
        """
        校验输入文件的完整性与数量。

        设计说明:
        - 视频数量必须为 3（与报告对比逻辑一致）；
        - 只要启用 audio/asr 就强制要求音频存在。
        """
        if len(self.video_paths) != 3:
            raise ValueError("Exactly 3 video files are required")
        
        for vpath in self.video_paths:
            if not vpath.exists():
                raise FileNotFoundError(f"Video not found: {vpath}")
        
        needs_audio = any(module in self.module_set for module in ("audio", "asr"))
        if needs_audio:
            if not self.audio_paths:
                raise ValueError("Audio modules requested but no audio files provided (use -a/--audios).")
            if len(self.audio_paths) != 3:
                raise ValueError("Exactly 3 audio files are required when audio/asr modules are enabled.")
            for apath in self.audio_paths:
                if not apath.exists():
                    raise FileNotFoundError(f"Audio not found: {apath}")
        elif self.audio_paths:
            logger.warning("Audio files provided but audio/asr modules are disabled. Files will be ignored.")
    
    @log_execution_time
    def analyze_video(
        self,
        video_path: Path,
        audio_path: Optional[Path] = None,
        frame_mode: str = "edge",
        step_hook: Optional[StepHook] = None
    ) -> Dict:
        """
        分析单个视频的所有模块，按模块顺序逐步执行。

        设计说明:
        - 以 self.modules 的顺序为准，保证 CLI 中指定的模块顺序被尊重，
          这样可以更容易按“组件顺序”进行调试与比对。
        - 每个模块完成后调用 step_hook，提供统一的阶段名与上下文，
          让上层可以选择“停顿/记录/对比”而不侵入核心逻辑。
        - 对音频依赖模块进行显式校验，避免静默跳过导致的错误结论。
        - 每个组件完成后输出摘要日志，便于逐步对比与定位问题。

        Args:
            video_path: 视频文件路径
            audio_path: 可选音频路径
            frame_mode: 截图模式 (edge/mosaic/off)
            step_hook: 可选调试钩子，签名为 (stage, payload)

        Returns:
            dict: 单个视频的分析结果
        """
        logger.info(f"Analyzing: {video_path.name}")
        result: Dict = {"path": str(video_path)}

        debug_messages = {
            "visual": "Extracting visual metrics...",
            "audio": "Extracting audio metrics...",
            "asr": "Running ASR transcription...",
            "yolo": "Running YOLO object detection..."
        }
        # 组件执行清单：调试时可直接注释掉某一行来跳过该组件
        module_sequence = [
            ("visual", lambda: self._run_visual(video_path, frame_mode)),
            ("audio", lambda: self._run_audio(audio_path, video_path)),
            ("asr", lambda: self._run_asr(audio_path, video_path)),
            ("yolo", lambda: self._run_yolo(video_path))
        ]
        module_map = {name: runner for name, runner in module_sequence}

        planned_steps = []
        for module_name in self.modules:
            runner = module_map.get(module_name)
            if runner:
                planned_steps.append((module_name, runner))

        if not planned_steps:
            logger.warning("No modules scheduled for execution.")
            return result

        for step_index, (module_name, runner) in enumerate(planned_steps, 1):
            display_name = MODULE_DISPLAY_NAMES.get(module_name, module_name)
            logger.info(f"[Step {step_index}/{len(planned_steps)}] {display_name} 开始")
            logger.debug(debug_messages.get(module_name, f"Running {module_name}..."))
            module_result = runner()
            result[module_name] = module_result
            self._log_module_summary(module_name, module_result, video_path, step_index, len(planned_steps))
            if step_hook:
                step_hook(
                    "module",
                    {
                        "module": module_name,
                        "video_path": str(video_path),
                        "module_result": module_result,
                        "current_result": result,
                        "step_index": step_index,
                        "step_total": len(planned_steps)
                    }
                )

        return result

    def _require_audio(self, audio_path: Optional[Path], module_name: str, video_path: Path) -> Path:
        """
        校验音频输入并返回可用路径。

        设计说明:
        - 将音频校验统一收敛在一个函数中，避免各模块重复判断；
        - 一旦缺失直接抛错，确保调试时问题可被明确定位。
        """
        if not audio_path or not audio_path.exists():
            raise FileNotFoundError(f"Audio required for module '{module_name}' but missing for {video_path.name}")
        return audio_path

    def _run_visual(self, video_path: Path, frame_mode: str) -> Dict:
        """
        执行视觉模块的指标提取。

        设计说明:
        - 仅负责调用视觉模块并返回结果，不在此处混入其它模块逻辑；
        - 保持输入输出纯粹，便于在调试时单独复用。
        """
        return extract_visual_metrics(
            str(video_path),
            str(self.frames_dir),
            frame_mode
        )

    def _run_audio(self, audio_path: Optional[Path], video_path: Path) -> Dict:
        """
        执行音频模块的指标提取。

        设计说明:
        - 先进行音频存在性检查，再调用音频特征提取；
        - 将异常提前抛出，避免后续逻辑基于空结果继续执行。
        """
        audio_path = self._require_audio(audio_path, "audio", video_path)
        return extract_audio_metrics(str(audio_path))

    def _run_asr(self, audio_path: Optional[Path], video_path: Path) -> Dict:
        """
        执行 ASR 模块的完整分析。

        设计说明:
        - 与音频模块共用同一份音频校验逻辑；
        - 保留 enable_prosody/enable_emotion 作为调试开关入口。
        """
        audio_path = self._require_audio(audio_path, "asr", video_path)
        return extract_full_asr_metrics(
            str(audio_path),
            enable_prosody=True,
            enable_emotion=True
        )

    def _run_yolo(self, video_path: Path) -> Dict:
        """
        执行 YOLO 目标检测相关分析。

        设计说明:
        - 仅在该函数内部导入 sample_frames，减少全局导入开销；
        - 帧采样数固定为 36，保证检测样本数量与性能之间的平衡。
        """
        from metrics_visual import sample_frames
        frames, _, _, _ = sample_frames(str(video_path), target=36)
        return extract_full_yolo_metrics(frames, enable_colors=True, enable_materials=True)

    def _fmt(self, value, spec: str) -> str:
        """
        统一格式化数值，避免 None/NaN 造成日志异常。

        设计说明:
        - 使用工具函数 format_value，保持全局一致的格式化策略；
        - 所有摘要日志均通过该函数输出数值字段。
        """
        return format_value(value, spec=spec, na="N/A")

    def _build_module_summary(self, module_name: str, module_result: Dict[str, Any]) -> str:
        """
        根据模块名称构建摘要字符串。

        设计说明:
        - 摘要只输出关键字段，便于快速定位问题；
        - 若结果结构异常，回退为类型提示，避免二次异常。
        """
        if not isinstance(module_result, dict):
            return f"结果类型={type(module_result).__name__}"

        if module_name == "visual":
            scene_label = "N/A"
            scenes = module_result.get("scene_categories", [])
            if scenes:
                scene_label = scenes[0].get("label", "N/A")
            return (
                f"镜头:{module_result.get('camera_angle', 'N/A')}/"
                f"{module_result.get('focal_length_tendency', 'N/A')} | "
                f"色彩:{module_result.get('hue_family', 'N/A')}/"
                f"{module_result.get('saturation_band', 'N/A')}/"
                f"{module_result.get('brightness_band', 'N/A')}/"
                f"{module_result.get('contrast', 'N/A')} | "
                f"场景:{scene_label} | "
                f"剪辑:{module_result.get('cuts', 0)} cuts, avg "
                f"{self._fmt(module_result.get('avg_shot_length'), '.2f')}s, "
                f"{module_result.get('transition_type', 'N/A')} | "
                f"时长:{self._fmt(module_result.get('duration'), '.1f')}s, "
                f"CCT≈{self._fmt(module_result.get('cct_mean'), '.0f')}K"
            )

        if module_name == "audio":
            key_signature = module_result.get("key_signature") or "N/A"
            return (
                f"BGM:{module_result.get('bgm_style', 'N/A')} | "
                f"情绪:{module_result.get('mood', 'N/A')} | "
                f"BPM≈{self._fmt(module_result.get('tempo_bpm'), '.1f')} | "
                f"打击能量≈{self._fmt(module_result.get('percussive_ratio'), '.2f')} | "
                f"语音占比≈{self._fmt(module_result.get('speech_ratio'), '.2f')} | "
                f"调式:{key_signature}"
            )

        if module_name == "asr":
            catchphrases = module_result.get("catchphrases", [])
            catchphrase_preview = "、".join(catchphrases[:3]) if catchphrases else "无"
            return (
                f"语速:{self._fmt(module_result.get('words_per_second'), '.2f')} w/s "
                f"({self._fmt(module_result.get('words_per_minute'), '.1f')} wpm) | "
                f"节奏:{module_result.get('pace', 'N/A')} | "
                f"词数:{module_result.get('num_words', 0)} | "
                f"停顿:{module_result.get('pause_style', 'N/A')} | "
                f"口头禅:{catchphrase_preview}"
            )

        if module_name == "yolo":
            detection = module_result.get("detection", {})
            environment = module_result.get("environment", {})
            top_objects = detection.get("top_objects", [])
            if top_objects:
                top_preview = "、".join([f"{name}:{count}" for name, count in top_objects[:3]])
            else:
                top_preview = "无"
            return (
                f"场景:{environment.get('environment_type', 'N/A')}/"
                f"{environment.get('cooking_style', 'N/A')} | "
                f"物体:{detection.get('unique_objects', 0)}类/"
                f"{detection.get('total_detections', 0)}次 | "
                f"Top:{top_preview}"
            )

        return f"模块={module_name}"

    def _log_module_summary(
        self,
        module_name: str,
        module_result: Dict[str, Any],
        video_path: Path,
        step_index: int,
        step_total: int
    ) -> None:
        """
        输出组件级摘要日志。

        设计说明:
        - 摘要聚焦关键字段，避免信息淹没；
        - 日志包含步骤序号，方便手动对齐执行顺序。
        """
        display_name = MODULE_DISPLAY_NAMES.get(module_name, module_name)
        summary = self._build_module_summary(module_name, module_result)
        logger.info(
            f"[Summary {step_index}/{step_total}] "
            f"{video_path.name} | {display_name} -> {summary}"
        )
    
    @log_execution_time
    def extract_consensus(self, video_metrics: List[Dict]) -> Dict:
        """
        计算跨视频的共识特征。

        设计说明:
        - 多数票用于离散标签；
        - 中位数用于数值指标，降低极值影响；
        - 输入为空时返回 "N/A" 或 None。

        Args:
            video_metrics: 每个视频的指标字典列表

        Returns:
            dict: 跨视频共识指标
        """
        logger.info("Extracting cross-video consensus...")
        
        def majority_value(values):
            """多数票规则：需要至少 2 次出现才算稳定共识。"""
            if not values:
                return "N/A"
            counter = Counter(values)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= 2:
                return most_common[0][0]
            return "Varied"
        
        def median_value(values):
            """对数值取中位数，并过滤 NaN/Inf 等异常值。"""
            valid = [v for v in values if isinstance(v, (int, float)) 
                    and not (np.isnan(v) or np.isinf(v))]
            return float(np.median(valid)) if valid else None
        
        # 视觉共识
        visual_entries = [v["visual"] for v in video_metrics if v.get("visual")]
        camera_angles = [entry["camera_angle"] for entry in visual_entries]
        hue_families = [entry["hue_family"] for entry in visual_entries]
        saturations = [entry["saturation_band"] for entry in visual_entries]
        brightnesses = [entry["brightness_band"] for entry in visual_entries]
        contrasts = [entry["contrast"] for entry in visual_entries]
        transition_types = [entry["transition_type"] for entry in visual_entries]
        countertop_colors = [entry["countertop_color"] for entry in visual_entries]
        countertop_textures = [entry["countertop_texture"] for entry in visual_entries]
        ccts = [entry["cct_mean"] for entry in visual_entries if entry.get("cct_mean") is not None]
        
        cuts_per_min = []
        avg_shot_lengths = []
        scene_labels = []
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
            natural_ratios.append(lighting.get("natural_light_ratio"))
            artificial_ratios.append(lighting.get("artificial_light_ratio"))
            scenes = entry.get("scene_categories", [])
            if scenes:
                scene_labels.append(scenes[0]["label"])
        
        # 音频共识
        audio_entries = [v["audio"] for v in video_metrics if v.get("audio")]
        bgm_styles = [entry.get("bgm_style", "Unknown") for entry in audio_entries]
        tempos = [entry.get("tempo_bpm") for entry in audio_entries]
        percussive_ratios = [entry.get("percussive_ratio") for entry in audio_entries]
        speech_ratios = [entry.get("speech_ratio") for entry in audio_entries]
        moods = [entry.get("mood", "Unknown") for entry in audio_entries]
        key_signatures = []
        instruments_list = []
        for entry in audio_entries:
            insts = entry.get("instruments", {}).get("detected_instruments", [])
            instruments_list.extend(insts)
            key_sig = entry.get("key_signature")
            if key_sig:
                key_signatures.append(key_sig)
        
        # 音画节拍对齐
        beat_alignments = []
        for v in video_metrics:
            audio_data = v.get("audio")
            visual_data = v.get("visual")
            if not audio_data or not visual_data:
                continue
            cuts_count = visual_data.get("cuts", 0)
            duration = visual_data.get("duration")
            if audio_data.get("beat_times") and cuts_count > 0 and duration and duration > 0:
                alignment = calculate_beat_alignment(
                    duration,
                    cuts_count,
                    audio_data["beat_times"]
                )
                beat_alignments.append(alignment)
        
        # YOLO 共识
        yolo_environments = []
        yolo_styles = []
        yolo_colors = {}
        yolo_materials = {}
        
        for v in video_metrics:
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
        
        consensus_colors = {obj: majority_value(colors) for obj, colors in yolo_colors.items()}
        consensus_materials = {obj: majority_value(materials) for obj, materials in yolo_materials.items()}
        
        consensus = {
            "camera_angle": majority_value(camera_angles),
            "focal_length_tendency": majority_value([entry.get("focal_length_tendency", "Unknown") for entry in visual_entries]) if visual_entries else "N/A",
            "camera_motion": majority_value([entry.get("camera_motion", {}).get("motion_type", "Unknown") for entry in visual_entries]) if visual_entries else "N/A",
            "composition_rule_of_thirds": majority_value([entry.get("composition", {}).get("rule_of_thirds", "Unknown") for entry in visual_entries]) if visual_entries else "N/A",
            "scene_category": majority_value(scene_labels),
            "hue_family": majority_value(hue_families),
            "saturation": majority_value(saturations),
            "brightness": majority_value(brightnesses),
            "contrast": majority_value(contrasts),
            "cct": median_value(ccts),
            "natural_light_ratio": median_value([ratio for ratio in natural_ratios if ratio is not None]),
            "artificial_light_ratio": median_value([ratio for ratio in artificial_ratios if ratio is not None]),
            "cuts_per_minute": median_value(cuts_per_min),
            "avg_shot_length": median_value(avg_shot_lengths),
            "transition_type": majority_value(transition_types),
            "beat_alignment": median_value(beat_alignments),
            "bgm_style": majority_value(bgm_styles),
            "bgm_mood": majority_value(moods),
            "bgm_instruments": list(set(instruments_list)) if instruments_list else [],
            "tempo_bpm": median_value(tempos),
            "percussive_ratio": median_value(percussive_ratios),
            "speech_ratio": median_value(speech_ratios),
            "key_signature": majority_value(key_signatures),
            "countertop_color": majority_value(countertop_colors),
            "countertop_texture": majority_value(countertop_textures),
            "yolo_environment": majority_value(yolo_environments),
            "yolo_style": majority_value(yolo_styles),
            "yolo_object_colors": consensus_colors,
            "yolo_object_materials": consensus_materials
        }
        
        cuts_per_min = consensus.get('cuts_per_minute', 0) or 0
        logger.debug(f"Consensus extracted: {consensus['camera_angle']} camera, "
                    f"{cuts_per_min:.2f} cuts/min")
        
        return consensus
    
    @log_execution_time
    def run(
        self,
        frame_mode: str = "edge",
        output_report: str = "style_report.docx",
        step_hook: Optional[StepHook] = None
    ) -> Dict:
        """
        运行完整流水线，并在关键阶段触发可选的调试钩子。

        设计说明:
        - 入口只做流程编排：逐视频分析 -> 共识提取 -> 报告生成；
        - step_hook 以统一的 stage/payload 形式触发，便于在 CLI 中
          进行“暂停查看”或记录中间结果；
        - 默认行为保持不变：若不传 step_hook，则完全等价于原有流程。

        Args:
            frame_mode: 截图模式 (edge/mosaic/off)
            output_report: 输出 Word 报告路径
            step_hook: 可选调试钩子

        Returns:
            dict: 包含逐视频指标、共识与报告路径的完整结果
        """
        logger.info("=" * 70)
        logger.info("Starting Video Style Analysis Pipeline")
        logger.info("=" * 70)
        
        # 逐视频分析
        video_metrics = []
        for i, video_path in enumerate(self.video_paths, 1):
            logger.info(f"[{i}/3] Processing {video_path.name}...")
            
            audio_path = self.audio_paths[i-1] if self.audio_paths else None
            
            metrics = self.analyze_video(
                video_path,
                audio_path,
                frame_mode=frame_mode,
                step_hook=step_hook
            )
            
            video_metrics.append(metrics)
            logger.info(f"  ✓ {video_path.name} complete")
        
        # 提取跨视频共识
        consensus = self.extract_consensus(video_metrics)
        if step_hook:
            step_hook("consensus", {"consensus": consensus})
        
        # 生成报告
        logger.info(f"Generating Word report: {output_report}")
        output_path = generate_word_report(
            video_metrics,
            consensus,
            output_report,
            show_screenshots=(frame_mode != "off")
        )
        if step_hook:
            step_hook("report", {"report_path": str(output_path)})
        logger.info(f"  ✓ Report saved to: {output_path}")
        
        # 输出总览
        visual_enabled = "visual" in self.module_set
        audio_enabled = "audio" in self.module_set
        asr_enabled = "asr" in self.module_set
        yolo_enabled = "yolo" in self.module_set
        
        logger.info("=" * 70)
        logger.info("Analysis Complete!")
        logger.info("=" * 70)
        logger.info(f"  Videos analyzed: {len(video_metrics)}")
        logger.info(f"  Modules executed -> Visual: {visual_enabled} | Audio: {audio_enabled} | ASR: {asr_enabled} | YOLO: {yolo_enabled}")
        logger.info("")
        if visual_enabled:
            cuts_per_min = consensus.get('cuts_per_minute', 0) or 0
            logger.info(f"  Camera: {consensus['camera_angle']}")
            logger.info(f"  Color: {consensus['hue_family']}, {consensus['saturation']}, {consensus['brightness']}")
            logger.info(f"  Scene: {consensus['scene_category']}")
            logger.info(f"  Editing: {cuts_per_min:.2f} cuts/min")
        if audio_enabled:
            logger.info(f"  BGM: {consensus['bgm_style']} | Mood: {consensus['bgm_mood']} | Key: {consensus.get('key_signature', 'Unknown')}")
        
        return {
            "video_metrics": video_metrics,
            "consensus": consensus,
            "report_path": output_path
        }
