#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Style Analysis Pipeline - Main Entry Point
Simplified interface to run the complete analysis pipeline.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

from utils import setup_logger, log_execution_time
from metrics_visual import extract_visual_metrics
from metrics_audio import extract_audio_metrics, calculate_beat_alignment
from metrics_asr import extract_full_asr_metrics
from metrics_yolo import extract_full_yolo_metrics
from report_word import generate_word_report

# Setup logger
logger = setup_logger()


class VideoStylePipeline:
    """
    Main pipeline class for video style analysis.
    
    Usage:
        pipeline = VideoStylePipeline(
            video_paths=['v1.mp4', 'v2.mp4', 'v3.mp4'],
            work_dir='work'
        )
        results = pipeline.run(
            enable_yolo=True,
            enable_asr=True,
            output_report='report.docx'
        )
    """
    
    def __init__(
        self,
        video_paths: List[str],
        audio_paths: Optional[List[str]] = None,
        work_dir: str = "work"
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
        
        # Validate inputs
        self._validate_inputs()
        
        # Create work directory
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.work_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        
        logger.info(f"Pipeline initialized with {len(self.video_paths)} videos")
    
    def _validate_inputs(self):
        """Validate input files."""
        if len(self.video_paths) != 3:
            raise ValueError("Exactly 3 video files are required")
        
        for vpath in self.video_paths:
            if not vpath.exists():
                raise FileNotFoundError(f"Video not found: {vpath}")
        
        if self.audio_paths:
            if len(self.audio_paths) != 3:
                raise ValueError("If audio_paths provided, must have exactly 3 files")
            for apath in self.audio_paths:
                if not apath.exists():
                    raise FileNotFoundError(f"Audio not found: {apath}")
    
    @log_execution_time
    def analyze_video(
        self,
        video_path: Path,
        audio_path: Optional[Path] = None,
        enable_yolo: bool = False,
        enable_asr: bool = False,
        frame_mode: str = "edge"
    ) -> Dict:
        """
        Analyze a single video.
        
        Args:
            video_path: Path to video file
            audio_path: Optional path to audio file
            enable_yolo: Enable YOLOv8 object detection
            enable_asr: Enable Whisper ASR
            frame_mode: Contact sheet mode (edge/mosaic/off)
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Analyzing: {video_path.name}")
        
        # Visual analysis
        logger.debug("Extracting visual metrics...")
        visual = extract_visual_metrics(
            str(video_path),
            str(self.frames_dir),
            frame_mode
        )
        
        # Audio analysis
        if audio_path and audio_path.exists():
            logger.debug("Extracting audio metrics...")
            audio = extract_audio_metrics(str(audio_path))
        else:
            logger.debug("Skipping audio analysis (no audio file)")
            audio = {"available": False}
        
        # ASR analysis
        if enable_asr and audio_path and audio_path.exists():
            logger.debug("Running ASR transcription...")
            asr = extract_full_asr_metrics(str(audio_path))
        else:
            asr = {"available": False}
        
        # YOLO analysis
        if enable_yolo:
            logger.debug("Running YOLO object detection...")
            from metrics_visual import sample_frames
            frames, _, _, _ = sample_frames(str(video_path), target=36)
            yolo = extract_full_yolo_metrics(frames)
        else:
            yolo = {"available": False}
        
        return {
            "path": str(video_path),
            "visual": visual,
            "audio": audio,
            "asr": asr,
            "yolo": yolo
        }
    
    @log_execution_time
    def extract_consensus(self, video_metrics: List[Dict]) -> Dict:
        """
        Extract cross-video consensus patterns.
        
        Args:
            video_metrics: List of per-video metrics
            
        Returns:
            dict: Consensus metrics
        """
        from collections import Counter
        import numpy as np
        
        logger.info("Extracting cross-video consensus...")
        
        def majority_value(values):
            """Get most common value (requires >= 2 occurrences)."""
            counter = Counter(values)
            most_common = counter.most_common(1)
            if most_common and most_common[0][1] >= 2:
                return most_common[0][0]
            return "Varied"
        
        def median_value(values):
            """Calculate median of numeric values."""
            valid = [v for v in values if isinstance(v, (int, float)) 
                    and not (np.isnan(v) or np.isinf(v))]
            return float(np.median(valid)) if valid else None
        
        # Visual consensus
        camera_angles = [v["visual"]["camera_angle"] for v in video_metrics]
        hue_families = [v["visual"]["hue_family"] for v in video_metrics]
        saturations = [v["visual"]["saturation_band"] for v in video_metrics]
        brightnesses = [v["visual"]["brightness_band"] for v in video_metrics]
        contrasts = [v["visual"]["contrast"] for v in video_metrics]
        transition_types = [v["visual"]["transition_type"] for v in video_metrics]
        countertop_colors = [v["visual"]["countertop_color"] for v in video_metrics]
        countertop_textures = [v["visual"]["countertop_texture"] for v in video_metrics]
        
        ccts = [v["visual"]["cct_mean"] for v in video_metrics]
        
        # Calculate cuts per minute
        cuts_per_min = []
        for v in video_metrics:
            duration = v["visual"]["duration"]
            cuts = v["visual"]["cuts"]
            if duration > 0:
                cuts_per_min.append(cuts / (duration / 60.0))
        
        # Calculate average shot length
        avg_shot_lengths = []
        for v in video_metrics:
            duration = v["visual"]["duration"]
            cuts = v["visual"]["cuts"]
            if cuts > 0:
                avg_shot_lengths.append(duration / (cuts + 1))
        
        # Audio consensus
        bgm_styles = []
        tempos = []
        percussive_ratios = []
        speech_ratios = []
        
        for v in video_metrics:
            if v["audio"].get("available"):
                bgm_styles.append(v["audio"]["bgm_style"])
                tempos.append(v["audio"]["tempo_bpm"])
                percussive_ratios.append(v["audio"]["percussive_ratio"])
                speech_ratios.append(v["audio"]["speech_ratio"])
        
        # Beat alignment
        beat_alignments = []
        for v in video_metrics:
            if v["audio"].get("available") and v["audio"].get("beat_times"):
                alignment = calculate_beat_alignment(
                    v["visual"]["duration"],
                    v["visual"]["cuts"],
                    v["audio"]["beat_times"]
                )
                if alignment is not None:
                    beat_alignments.append(alignment)
        
        # YOLO consensus
        yolo_available = any(v["yolo"].get("available") for v in video_metrics)
        yolo_environments = []
        yolo_styles = []
        
        if yolo_available:
            for v in video_metrics:
                if v["yolo"].get("available"):
                    env = v["yolo"].get("environment", {})
                    yolo_environments.append(env.get("environment_type", "Unknown"))
                    yolo_styles.append(env.get("cooking_style", "Unknown"))
        
        consensus = {
            "camera_angle": majority_value(camera_angles),
            "hue_family": majority_value(hue_families),
            "saturation": majority_value(saturations),
            "brightness": majority_value(brightnesses),
            "contrast": majority_value(contrasts),
            "cct": median_value(ccts),
            "cuts_per_minute": median_value(cuts_per_min),
            "avg_shot_length": median_value(avg_shot_lengths),
            "transition_type": majority_value(transition_types),
            "beat_alignment": median_value(beat_alignments),
            "bgm_style": majority_value(bgm_styles) if bgm_styles else "N/A",
            "tempo_bpm": median_value(tempos),
            "percussive_ratio": median_value(percussive_ratios),
            "speech_ratio": median_value(speech_ratios),
            "countertop_color": majority_value(countertop_colors),
            "countertop_texture": majority_value(countertop_textures),
            "yolo_available": yolo_available,
            "yolo_environment": majority_value(yolo_environments) if yolo_environments else "N/A",
            "yolo_style": majority_value(yolo_styles) if yolo_styles else "N/A"
        }
        
        logger.debug(f"Consensus extracted: {consensus['camera_angle']} camera, "
                    f"{consensus['cuts_per_minute']:.2f if consensus['cuts_per_minute'] else 0:.2f} cuts/min")
        
        return consensus
    
    @log_execution_time
    def run(
        self,
        enable_yolo: bool = False,
        enable_asr: bool = False,
        frame_mode: str = "edge",
        output_report: str = "style_report.docx"
    ) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            enable_yolo: Enable YOLOv8 object detection
            enable_asr: Enable Whisper ASR
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
                enable_yolo=enable_yolo,
                enable_asr=enable_asr,
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
        logger.info("=" * 70)
        logger.info("Analysis Complete!")
        logger.info("=" * 70)
        logger.info(f"  Videos analyzed: {len(video_metrics)}")
        logger.info(f"  Audio analyzed: {sum(1 for v in video_metrics if v['audio'].get('available'))}")
        logger.info(f"  ASR enabled: {sum(1 for v in video_metrics if v['asr'].get('available'))}")
        logger.info(f"  YOLO enabled: {sum(1 for v in video_metrics if v['yolo'].get('available'))}")
        logger.info("")
        logger.info(f"  Camera: {consensus['camera_angle']}")
        logger.info(f"  Color: {consensus['hue_family']}, {consensus['saturation']}, {consensus['brightness']}")
        logger.info(f"  Editing: {consensus['cuts_per_minute']:.2f if consensus['cuts_per_minute'] else 0:.2f} cuts/min")
        logger.info(f"  BGM: {consensus['bgm_style']}")
        
        return {
            "video_metrics": video_metrics,
            "consensus": consensus,
            "report_path": output_path
        }


# VideoStylePipeline class ends here
# Main entry point is in project root: main.py

