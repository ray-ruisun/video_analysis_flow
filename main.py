#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频风格分析流水线 - CLI 入口

设计逻辑:
- 将入口拆分为“参数解析 -> 模块解析 -> 构建流水线 -> 运行并输出”四段，
  让每一步的职责清晰、便于逐步调试与维护。
- 通过 --step-debug 在每个组件完成后暂停，支持“按组件逐步定位问题”。
- 所有异常在主入口统一处理，保证退出码与日志行为一致。
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# 将 src 目录加入路径，保证 CLI 入口可以直接导入内部模块
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import VideoStylePipeline
from utils import setup_logger

# 初始化日志器，统一控制台与文件输出
logger = setup_logger()


def parse_args():
    """
    解析命令行参数并返回结构化配置。

    设计说明:
    - CLI 只负责“输入表达”，不直接参与业务逻辑；
    - 可选项尽量直观，调试相关开关与业务开关分离；
    - --modules 与 --asr/--yolo 冲突时，以 --modules 为准。
    """
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

    parser.add_argument(
        "--step-debug",
        action="store_true",
        help="Pause after each component for step-by-step debugging / 组件级分步调试"
    )
    
    return parser.parse_args()


def resolve_modules(args):
    """
    解析模块列表并输出有序、去重后的模块序列。

    设计说明:
    - 优先使用 --modules 显式指定的顺序；
    - 未指定时默认启用 visual/audio，并根据 --asr/--yolo 扩展；
    - 模块顺序会影响执行与调试体验，因此必须保持可控。
    """
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
    # 保留顺序并去重，确保执行顺序可预测
    resolved = []
    seen = set()
    for module in modules:
        if module not in seen:
            resolved.append(module)
            seen.add(module)
    return resolved


def build_pipeline(args, modules):
    """
    构建并返回流水线实例。

    设计说明:
    - 将实例化逻辑独立封装，便于在调试时替换参数或注入钩子；
    - 所有路径参数保持原样传入，路径校验由流水线内部统一完成。
    """
    return VideoStylePipeline(
        video_paths=args.videos,
        audio_paths=args.audios,
        work_dir=args.work_dir,
        modules=modules
    )


def _summarize_step_payload(payload: Dict[str, Any]) -> str:
    """
    将调试 payload 约简为可读摘要，避免日志过载。

    设计说明:
    - 只展示关键字段或字典键名，帮助快速确认模块是否产出结果；
    - 保持输出简短，避免影响分步调试的节奏。
    """
    if not isinstance(payload, dict):
        return f"type={type(payload).__name__}"
    module_result = payload.get("module_result")
    if isinstance(module_result, dict):
        keys = list(module_result.keys())
        if not keys:
            return "module_result: empty"
        preview = ", ".join(keys[:6])
        suffix = "" if len(keys) <= 6 else f"...(+{len(keys) - 6})"
        return f"module_result keys: {preview}{suffix}"
    return "module_result: n/a"


def build_step_hook(args):
    """
    构建“分步调试钩子”，用于组件级暂停与日志提示。

    设计说明:
    - 仅在 --step-debug 启用时返回钩子，避免默认流程被阻塞；
    - 钩子遵循 pipeline 的 stage/payload 协议，便于统一扩展；
    - 若非交互终端，自动降级为“只记录不暂停”，避免批处理卡死。
    """
    if not args.step_debug:
        return None

    interactive = sys.stdin.isatty()
    if not interactive:
        logger.warning("--step-debug enabled but stdin is not a TTY; skip interactive pause.")

    def step_hook(stage: str, payload: Dict[str, Any]) -> None:
        if stage == "module":
            video_path = payload.get("video_path", "unknown")
            module_name = payload.get("module", "unknown")
            summary = _summarize_step_payload(payload)
            logger.info(f"[Step] {Path(video_path).name} -> {module_name} 完成 | {summary}")
        elif stage == "consensus":
            logger.info("[Step] 跨视频共识计算完成")
        elif stage == "report":
            report_path = payload.get("report_path", "unknown")
            logger.info(f"[Step] 报告生成完成: {report_path}")
        else:
            logger.info(f"[Step] 阶段完成: {stage}")

        if interactive:
            input("按回车继续下一步...")

    return step_hook


def run_pipeline(args):
    """
    执行完整的流水线流程，并返回结果字典。

    设计说明:
    - 该函数集中处理“模块解析 + 实例构建 + 运行”三个步骤；
    - 结果字典仅用于输出提示与后续扩展，不影响主流程。
    """
    logger.info("Starting Video Style Analysis Pipeline")
    logger.info(f"Videos: {args.videos}")

    modules = resolve_modules(args)
    logger.info(f"Modules requested: {modules}")
    if args.modules and (args.asr or args.yolo):
        logger.warning("--modules specified; --asr/--yolo flags are ignored.")

    pipeline = build_pipeline(args, modules)
    step_hook = build_step_hook(args)

    results = pipeline.run(
        frame_mode=args.frames,
        output_report=args.output,
        step_hook=step_hook
    )

    logger.success(f"✓ Analysis complete! Report: {results['report_path']}")
    return results


def main():
    """
    CLI 主入口。

    设计说明:
    - 入口只负责异常包装与退出码控制；
    - 业务流程统一交给 run_pipeline，确保主函数短小可控。
    """
    args = parse_args()

    try:
        run_pipeline(args)
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
