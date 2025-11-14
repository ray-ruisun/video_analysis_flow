#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Style Analysis Pipeline - Main Entry Point
主入口：串联所有src模块的功能
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import VideoStylePipeline
from utils import setup_logger

# Setup logger
logger = setup_logger()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Style Analysis Pipeline / 视频风格分析流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:
  # Basic usage / 基础使用
  python main.py -v videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 -o report.docx
  
  # With audio / 带音频
  python main.py -v videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \\
                 -a work/v1.wav work/v2.wav work/v3.wav \\
                 -o report.docx
  
  # Enable all features / 启用所有功能
  python main.py -v videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \\
                 --yolo --asr -o report.docx
        """
    )
    
    parser.add_argument(
        "-v", "--videos",
        nargs=3,
        required=True,
        metavar=("V1", "V2", "V3"),
        help="3 video files / 3个视频文件"
    )
    
    parser.add_argument(
        "-a", "--audios",
        nargs=3,
        metavar=("A1", "A2", "A3"),
        help="3 audio files (22.05kHz mono wav) / 3个音频文件"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output report (.docx) / 输出报告"
    )
    
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Enable YOLOv8 object detection / 启用物体检测"
    )
    
    parser.add_argument(
        "--asr",
        action="store_true",
        help="Enable Whisper ASR / 启用语音识别"
    )
    
    parser.add_argument(
        "--frames",
        choices=["edge", "mosaic", "off"],
        default="edge",
        help="Contact sheet mode / 截图模式 (default: edge)"
    )
    
    parser.add_argument(
        "--work-dir",
        default="work",
        help="Working directory / 工作目录 (default: work)"
    )
    
    parser.add_argument(
        "--modules",
        help="Comma-separated module list (visual,audio,asr,yolo). Default honors --yolo/--asr flags. / 模块列表"
    )
    
    return parser.parse_args()


def resolve_modules(args):
    """Resolve requested modules from CLI flags."""
    if args.modules:
        tokens = [token.strip().lower() for token in args.modules.split(",")]
        modules = []
        seen = set()
        for token in tokens:
            if not token:
                continue
            if token not in ("visual", "audio", "asr", "yolo"):
                raise ValueError(f"Invalid module '{token}'. Valid options: visual,audio,asr,yolo")
            if token not in seen:
                modules.append(token)
                seen.add(token)
        if not modules:
            raise ValueError("No valid modules specified after parsing --modules.")
        return modules
    
    modules = ["visual", "audio"]
    if args.asr:
        modules.append("asr")
    if args.yolo:
        modules.append("yolo")
    # Preserve order but deduplicate
    resolved = []
    seen = set()
    for module in modules:
        if module not in seen:
            resolved.append(module)
            seen.add(module)
    return resolved


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        logger.info("Starting Video Style Analysis Pipeline")
        logger.info(f"Videos: {args.videos}")
        
        modules = resolve_modules(args)
        logger.info(f"Modules requested: {modules}")
        if args.modules and (args.asr or args.yolo):
            logger.warning("--modules specified; --asr/--yolo flags are ignored.")
        
        # Create pipeline
        pipeline = VideoStylePipeline(
            video_paths=args.videos,
            audio_paths=args.audios,
            work_dir=args.work_dir,
            modules=modules
        )
        
        # Run analysis
        results = pipeline.run(
            frame_mode=args.frames,
            output_report=args.output
        )
        
        logger.success(f"✓ Analysis complete! Report: {results['report_path']}")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user / 用户中断")
        return 1
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

