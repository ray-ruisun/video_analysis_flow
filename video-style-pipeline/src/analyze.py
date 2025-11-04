#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main video style analysis pipeline.
Orchestrates all metrics extraction and generates comprehensive Word report.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

# Import local modules
from metrics_visual import extract_visual_metrics, sample_frames
from metrics_audio import extract_audio_metrics, calculate_beat_alignment
from metrics_asr import extract_full_asr_metrics
from metrics_yolo import extract_full_yolo_metrics
from report_word import generate_word_report


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Style Analysis Pipeline - Extract stylistic patterns from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python analyze.py --videos v1.mp4 v2.mp4 v3.mp4 --report report.docx
  
  # With pre-extracted audio
  python analyze.py --videos v1.mp4 v2.mp4 v3.mp4 \\
                    --audios v1.wav v2.wav v3.wav \\
                    --report report.docx
  
  # Enable all features
  python analyze.py --videos v1.mp4 v2.mp4 v3.mp4 \\
                    --enable-yolo --enable-asr \\
                    --frames edge --report report.docx
        """
    )
    
    parser.add_argument(
        "--videos",
        nargs=3,
        required=True,
        metavar=("VIDEO1", "VIDEO2", "VIDEO3"),
        help="Paths to 3 human-verified video files"
    )
    
    parser.add_argument(
        "--audios",
        nargs=3,
        required=False,
        metavar=("AUDIO1", "AUDIO2", "AUDIO3"),
        help="Optional: Pre-extracted audio files (22.05kHz mono wav recommended)"
    )
    
    parser.add_argument(
        "--report",
        required=True,
        help="Output path for Word document report (.docx)"
    )
    
    parser.add_argument(
        "--frames",
        choices=["mosaic", "edge", "off"],
        default="edge",
        help="Contact sheet visualization style (default: edge)"
    )
    
    parser.add_argument(
        "--enable-yolo",
        action="store_true",
        help="Enable YOLOv8 object detection (requires: pip install ultralytics)"
    )
    
    parser.add_argument(
        "--enable-asr",
        action="store_true",
        help="Enable Whisper ASR for narration analysis (requires: pip install faster-whisper)"
    )
    
    parser.add_argument(
        "--work-dir",
        default="work",
        help="Working directory for temporary files (default: work/)"
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate input files exist."""
    errors = []
    
    for video_path in args.videos:
        if not os.path.exists(video_path):
            errors.append(f"Video not found: {video_path}")
    
    if args.audios:
        for audio_path in args.audios:
            if not os.path.exists(audio_path):
                errors.append(f"Audio not found: {audio_path}")
    
    if errors:
        print("ERROR: Input validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def extract_consensus(video_metrics):
    """
    Extract consensus patterns across all videos.
    
    Args:
        video_metrics: List of per-video metric dictionaries
        
    Returns:
        dict: Consensus metrics
    """
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
    
    return {
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


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("Video Style Analysis Pipeline")
    print("=" * 70)
    print()
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate inputs
    validate_inputs(args)
    
    # Create work directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Process each video
    video_metrics = []
    
    for i, video_path in enumerate(args.videos, 1):
        print(f"[{i}/3] Processing {Path(video_path).name}...")
        
        # Extract visual metrics
        print(f"  → Analyzing visual features...")
        visual = extract_visual_metrics(video_path, frames_dir, args.frames)
        
        # Extract audio metrics
        audio_path = args.audios[i-1] if args.audios else None
        
        if audio_path:
            print(f"  → Analyzing audio features...")
            audio = extract_audio_metrics(audio_path)
        else:
            print(f"  → Skipping audio (no audio file provided)")
            audio = {"available": False}
        
        # Extract ASR metrics
        if args.enable_asr and audio_path:
            print(f"  → Transcribing narration (this may take a while)...")
            asr = extract_full_asr_metrics(audio_path)
            if not asr.get("available"):
                print(f"     Warning: ASR failed - {asr.get('error', 'Unknown error')}")
        else:
            asr = {"available": False}
        
        # Extract YOLO metrics
        if args.enable_yolo:
            print(f"  → Detecting objects with YOLOv8...")
            # Load frames for YOLO
            frames, _, _, _ = sample_frames(video_path, target=36)
            yolo = extract_full_yolo_metrics(frames)
            if not yolo.get("available"):
                print(f"     Warning: YOLO failed - {yolo.get('error', 'Unknown error')}")
        else:
            yolo = {"available": False}
        
        # Combine metrics
        video_metrics.append({
            "path": video_path,
            "visual": visual,
            "audio": audio,
            "asr": asr,
            "yolo": yolo
        })
        
        print(f"  ✓ Complete")
        print()
    
    # Extract consensus
    print("Extracting cross-video consensus patterns...")
    consensus = extract_consensus(video_metrics)
    print("  ✓ Complete")
    print()
    
    # Generate report
    print(f"Generating Word report: {args.report}")
    output_path = generate_word_report(
        video_metrics,
        consensus,
        args.report,
        show_screenshots=(args.frames != "off")
    )
    print(f"  ✓ Report saved to: {output_path}")
    print()
    
    # Summary
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  Videos analyzed: {len(video_metrics)}")
    print(f"  Audio analyzed: {sum(1 for v in video_metrics if v['audio'].get('available'))}")
    print(f"  ASR enabled: {sum(1 for v in video_metrics if v['asr'].get('available'))}")
    print(f"  YOLO enabled: {sum(1 for v in video_metrics if v['yolo'].get('available'))}")
    print()
    print(f"  Camera angle consensus: {consensus['camera_angle']}")
    print(f"  Color palette: {consensus['hue_family']}, {consensus['saturation']}, {consensus['brightness']}")
    print(f"  Editing pace: {consensus['cuts_per_minute']:.2f if consensus['cuts_per_minute'] else 0:.2f} cuts/min")
    print(f"  BGM style: {consensus['bgm_style']}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

