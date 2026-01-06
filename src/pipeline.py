#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Style Analysis Pipeline - Main Entry Point

Simplified interface to run the complete analysis pipeline for video style extraction.
Analyzes camera language, color/lighting, editing rhythm, BGM, environment, and narration.
"""

from pathlib import Path
from typing import List, Dict, Optional, Sequence
from collections import Counter

import numpy as np
from loguru import logger

from utils import setup_logger, log_execution_time
from metrics_visual import extract_visual_metrics
from metrics_audio import extract_audio_metrics, calculate_beat_alignment
from metrics_asr import extract_full_asr_metrics
from metrics_yolo import extract_full_yolo_metrics
from report_word import generate_word_report

# Setup logger
logger = setup_logger()

VALID_MODULES = ["visual", "audio", "asr", "yolo"]


class VideoStylePipeline:
    """
    Main pipeline class for video style analysis.
    
    Usage:
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
        Initialize the pipeline.
        
        Args:
            video_paths: List of 3 video file paths
            audio_paths: Optional list of 3 audio file paths (22.05kHz mono wav)
            work_dir: Working directory for temporary files
        """
        self.video_paths = [Path(p) for p in video_paths]
        self.audio_paths = [Path(p) for p in audio_paths] if audio_paths else None
        self.work_dir = Path(work_dir)
        self.modules = self._normalize_modules(modules)
        self.module_set = set(self.modules)
        
        # Validate inputs
        self._validate_inputs()
        
        # Create work directory
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.work_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        
        logger.info(f"Pipeline initialized with {len(self.video_paths)} videos | Modules: {', '.join(self.modules)}")

    def _normalize_modules(self, modules: Optional[Sequence[str]]) -> List[str]:
        """Validate and normalize module selection."""
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
        # Default order
        return ["visual", "audio", "asr", "yolo"]
    
    def _validate_inputs(self):
        """Validate input files."""
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
        frame_mode: str = "edge"
    ) -> Dict:
        """
        Analyze a single video.
        
        Args:
            video_path: Path to video file
            audio_path: Optional path to audio file
            frame_mode: Contact sheet mode (edge/mosaic/off)
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Analyzing: {video_path.name}")
        modules = self.module_set
        result: Dict = {"path": str(video_path)}
        
        visual = None
        if "visual" in modules:
            logger.debug("Extracting visual metrics...")
            visual = extract_visual_metrics(
                str(video_path),
                str(self.frames_dir),
                frame_mode
            )
            result["visual"] = visual
        
        audio = None
        if "audio" in modules:
            if not audio_path or not audio_path.exists():
                raise FileNotFoundError(f"Audio required for module 'audio' but missing for {video_path.name}")
            logger.debug("Extracting audio metrics...")
            audio = extract_audio_metrics(str(audio_path))
            result["audio"] = audio
        
        if "asr" in modules:
            if not audio_path or not audio_path.exists():
                raise FileNotFoundError(f"Audio required for module 'asr' but missing for {video_path.name}")
            logger.debug("Running ASR transcription...")
            asr = extract_full_asr_metrics(
                str(audio_path),
                enable_prosody=True,
                enable_emotion=True
            )
            result["asr"] = asr
        
        if "yolo" in modules:
            logger.debug("Running YOLO object detection...")
            from metrics_visual import sample_frames
            frames, _, _, _ = sample_frames(str(video_path), target=36)
            yolo = extract_full_yolo_metrics(frames, enable_colors=True, enable_materials=True)
            result["yolo"] = yolo
        
        return result
    
    @log_execution_time
    def extract_consensus(self, video_metrics: List[Dict]) -> Dict:
        """
        Extract cross-video consensus patterns.
        
        Args:
            video_metrics: List of per-video metrics dictionaries
            
        Returns:
            dict: Consensus metrics across all videos
        """
        logger.info("Extracting cross-video consensus...")
        
        def majority_value(values):
            """Get most common value (requires >= 2 occurrences)."""
            if not values:
                return "N/A"
            counter = Counter(values)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= 2:
                return most_common[0][0]
            return "Varied"
        
        def median_value(values):
            """Calculate median of numeric values, filtering invalid entries."""
            valid = [v for v in values if isinstance(v, (int, float)) 
                    and not (np.isnan(v) or np.isinf(v))]
            return float(np.median(valid)) if valid else None
        
        # Visual consensus
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
        
        # Audio consensus
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
        
        # Beat alignment
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
        
        # YOLO consensus
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
        output_report: str = "style_report.docx"
    ) -> Dict:
        """
        Run the complete pipeline with the configured module selection.
        
        Args:
            frame_mode: Contact sheet mode (edge/mosaic/off)
            output_report: Output Word document path
            
        Returns:
            dict: Complete analysis results including consensus
        """
        logger.info("=" * 70)
        logger.info("Starting Video Style Analysis Pipeline")
        logger.info("=" * 70)
        
        # Analyze each video
        video_metrics = []
        for i, video_path in enumerate(self.video_paths, 1):
            logger.info(f"[{i}/3] Processing {video_path.name}...")
            
            audio_path = self.audio_paths[i-1] if self.audio_paths else None
            
            metrics = self.analyze_video(
                video_path,
                audio_path,
                frame_mode=frame_mode
            )
            
            video_metrics.append(metrics)
            logger.info(f"  ✓ {video_path.name} complete")
        
        # Extract consensus
        consensus = self.extract_consensus(video_metrics)
        
        # Generate report
        logger.info(f"Generating Word report: {output_report}")
        output_path = generate_word_report(
            video_metrics,
            consensus,
            output_report,
            show_screenshots=(frame_mode != "off")
        )
        logger.info(f"  ✓ Report saved to: {output_path}")
        
        # Print summary
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



