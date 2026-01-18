#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频风格分析流水线 - CLI 入口

模块化架构说明:
- 每个分析步骤都是独立的模块，有明确的输入和输出
- 支持两种使用方式:
  1. 完整流水线: 一次性执行所有步骤
  2. 单步执行: 按需调用特定分析模块

流水线步骤:
1. visual    -> VisualOutput   (色彩、镜头、构图、场景、剪辑)
2. audio     -> AudioOutput    (BPM、节拍、能量、BGM风格、情绪)
3. asr       -> ASROutput      (语音转录、语速、口头禅、韵律、情感)
4. yolo      -> YOLOOutput     (物体检测、环境分类、颜色、材质)
5. consensus -> ConsensusOutput (跨视频特征聚合)
6. report    -> ReportOutput   (Word文档)

================================================================================
单独使用每个模块 (Python API)
================================================================================

# 1. 视觉分析模块
from pathlib import Path
from steps import VisualAnalysisStep, VideoInput

step = VisualAnalysisStep()
input_data = VideoInput(video_path=Path("video.mp4"), work_dir=Path("work"))
output = step.run(input_data)
print(f"镜头角度: {output.camera_angle}")
print(f"色调: {output.hue_family}")
print(f"剪辑数: {output.cuts}")

# 2. 音频分析模块
from steps import AudioAnalysisStep, AudioInput

step = AudioAnalysisStep()
input_data = AudioInput(audio_path=Path("audio.wav"))
output = step.run(input_data)
print(f"BPM: {output.tempo_bpm}")
print(f"BGM风格: {output.bgm_style}")
print(f"情绪: {output.mood}")

# 3. ASR 分析模块
from steps import ASRAnalysisStep, ASRInput

step = ASRAnalysisStep()
input_data = ASRInput(
    audio_path=Path("audio.wav"),
    language="en",
    enable_prosody=True,
    enable_emotion=True
)
output = step.run(input_data)
print(f"转录文本: {output.text[:100]}...")
print(f"语速: {output.words_per_minute:.1f} wpm")
print(f"口头禅: {output.catchphrases}")

# 4. YOLO 检测模块
from steps import YOLOAnalysisStep, YOLOInput

step = YOLOAnalysisStep()
input_data = YOLOInput(
    video_path=Path("video.mp4"),
    enable_colors=True,
    enable_materials=True
)
output = step.run(input_data)
print(f"环境类型: {output.environment.get('environment_type')}")
print(f"检测物体数: {output.detection.get('unique_objects')}")

# 5. 共识计算模块
from steps import ConsensusStep, ConsensusInput, VideoMetrics

step = ConsensusStep()
input_data = ConsensusInput(video_metrics=[vm1, vm2, vm3])  # VideoMetrics 列表
output = step.run(input_data)
print(f"共识镜头角度: {output.camera_angle}")
print(f"共识BGM风格: {output.bgm_style}")

# 6. 报告生成模块
from steps import ReportGenerationStep, ReportInput

step = ReportGenerationStep()
input_data = ReportInput(
    video_metrics=[vm1, vm2, vm3],
    consensus=consensus_output,
    output_path="report.docx"
)
output = step.run(input_data)
print(f"报告已生成: {output.report_path}")

================================================================================
使用 ModularPipeline (单步执行)
================================================================================

from pathlib import Path
from pipeline_runner import ModularPipeline, PipelineConfig

config = PipelineConfig(
    video_paths=[Path("v1.mp4"), Path("v2.mp4"), Path("v3.mp4")],
    audio_paths=[Path("a1.wav"), Path("a2.wav"), Path("a3.wav")],
    modules=["visual", "audio"]
)
pipeline = ModularPipeline(config)

# 单独执行某个分析
visual_output = pipeline.run_visual_analysis(Path("v1.mp4"))
audio_output = pipeline.run_audio_analysis(Path("a1.wav"))
asr_output = pipeline.run_asr_analysis(Path("a1.wav"))
yolo_output = pipeline.run_yolo_analysis(Path("v1.mp4"))

================================================================================
命令行使用示例
================================================================================

  # 基础使用 (默认 visual + audio)
  python main.py -v v1.mp4 v2.mp4 v3.mp4 -a a1.wav a2.wav a3.wav -o report.docx

  # 仅视觉分析 (不需要音频)
  python main.py -v v1.mp4 v2.mp4 v3.mp4 --modules visual -o report.docx

  # 启用所有模块
  python main.py -v v1.mp4 v2.mp4 v3.mp4 -a a1.wav a2.wav a3.wav \\
                 --yolo --asr -o report.docx

  # 自定义模块顺序
  python main.py -v v1.mp4 v2.mp4 v3.mp4 -a a1.wav a2.wav a3.wav \\
                 --modules visual,audio,yolo -o report.docx

  # 分步调试模式
  python main.py -v v1.mp4 v2.mp4 v3.mp4 -a a1.wav a2.wav a3.wav \\
                 --step-debug -o report.docx
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# 将 src 目录加入路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline_runner import ModularPipeline, PipelineConfig, VideoStylePipeline
from utils import setup_logger

# 初始化日志器
logger = setup_logger()


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="Video Style Analysis Pipeline / 视频风格分析流水线 (v2.0 模块化版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模块化流水线步骤:
  1. visual   - 视觉分析 (色彩、镜头、构图、场景、剪辑)
  2. audio    - 音频分析 (BPM、节拍、能量、BGM风格、情绪) [需要音频]
  3. asr      - ASR分析 (语音转录、语速、口头禅) [需要音频]
  4. yolo     - YOLO检测 (物体检测、环境分类)
  5. consensus - 共识计算 (自动执行)
  6. report   - 报告生成 (自动执行)

Examples / 示例:
  # 基础使用 (默认 visual + audio)
  python main.py -v videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \\
                 -a work/v1.wav work/v2.wav work/v3.wav \\
                 -o report.docx

  # 仅视觉分析 (不需要音频)
  python main.py -v videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \\
                 --modules visual -o report.docx

  # 启用所有模块
  python main.py -v videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \\
                 -a work/v1.wav work/v2.wav work/v3.wav \\
                 --yolo --asr -o report.docx

  # 自定义模块顺序
  python main.py -v videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \\
                 -a work/v1.wav work/v2.wav work/v3.wav \\
                 --modules visual,yolo,audio -o report.docx

  # 分步调试模式
  python main.py -v videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \\
                 -a work/v1.wav work/v2.wav work/v3.wav \\
                 --step-debug -o report.docx
        """
    )
    
    # 输入输出
    parser.add_argument(
        "-v", "--videos",
        nargs=3,
        required=True,
        metavar=("V1", "V2", "V3"),
        help="3 个视频文件路径"
    )
    
    parser.add_argument(
        "-a", "--audios",
        nargs=3,
        metavar=("A1", "A2", "A3"),
        help="3 个音频文件路径 (22.05kHz mono wav，audio/asr 模块需要)"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出报告路径 (.docx)"
    )
    
    # 模块控制
    parser.add_argument(
        "--modules",
        help="逗号分隔的模块列表 (visual,audio,asr,yolo). 覆盖 --yolo/--asr 标志"
    )
    
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="启用 YOLO 目标检测模块"
    )
    
    parser.add_argument(
        "--asr",
        action="store_true",
        help="启用 ASR 语音识别模块"
    )
    
    # 可视化选项
    parser.add_argument(
        "--frames",
        choices=["edge", "mosaic", "off"],
        default="edge",
        help="截图模式 (default: edge)"
    )
    
    # 工作目录
    parser.add_argument(
        "--work-dir",
        default="work",
        help="工作目录 (default: work)"
    )
    
    # 调试选项
    parser.add_argument(
        "--step-debug",
        action="store_true",
        help="步骤级调试模式 - 每个模块完成后暂停"
    )
    
    return parser.parse_args()


def resolve_modules(args):
    """
    解析模块列表
    
    优先级:
    1. --modules 显式指定
    2. 默认 visual + audio，再加上 --asr/--yolo 标志
    """
    if args.modules:
        tokens = [t.strip().lower() for t in args.modules.split(",")]
        modules = []
        seen = set()
        for token in tokens:
            if not token:
                continue
            if token not in ("visual", "audio", "asr", "yolo"):
                raise ValueError(f"无效模块 '{token}'. 有效选项: visual, audio, asr, yolo")
            if token not in seen:
                modules.append(token)
                seen.add(token)
        if not modules:
            raise ValueError("--modules 解析后无有效模块")
        return modules
    
    # 默认模块
    modules = ["visual", "audio"]
    if args.asr:
        modules.append("asr")
    if args.yolo:
        modules.append("yolo")
    
    return modules


def build_step_hook(args):
    """
    构建步骤调试钩子
    """
    if not args.step_debug:
        return None
    
    interactive = sys.stdin.isatty()
    if not interactive:
        logger.warning("--step-debug 启用但 stdin 非 TTY，将跳过交互式暂停")
    
    def step_hook(stage: str, payload: Dict[str, Any]) -> None:
        if stage == "module":
            video_path = payload.get("video_path", "unknown")
            module_name = payload.get("module", "unknown")
            step_index = payload.get("step_index", 0)
            step_total = payload.get("step_total", 0)
            logger.info(f"[调试] {Path(video_path).name} -> {module_name} ({step_index}/{step_total}) 完成")
        elif stage == "consensus":
            logger.info("[调试] 共识计算完成")
        elif stage == "report":
            report_path = payload.get("report_path", "unknown")
            logger.info(f"[调试] 报告生成完成: {report_path}")
        else:
            logger.info(f"[调试] 阶段完成: {stage}")
        
        if interactive:
            input("按回车继续下一步...")
    
    return step_hook


def run_pipeline(args):
    """
    执行流水线
    
    流程:
    1. 解析模块列表
    2. 构建配置
    3. 创建流水线实例
    4. 执行并返回结果
    """
    logger.info("启动视频风格分析流水线 (v2.0 模块化版)")
    logger.info(f"视频: {args.videos}")
    
    # 解析模块
    modules = resolve_modules(args)
    logger.info(f"执行模块: {modules}")
    
    if args.modules and (args.asr or args.yolo):
        logger.warning("--modules 已指定，--asr/--yolo 标志将被忽略")
    
    # 构建配置
    config = PipelineConfig(
        video_paths=[Path(p) for p in args.videos],
        audio_paths=[Path(p) for p in args.audios] if args.audios else None,
        work_dir=Path(args.work_dir),
        modules=modules,
        frame_mode=args.frames,
        output_report=args.output
    )
    
    # 创建流水线
    pipeline = ModularPipeline(config)
    
    # 构建调试钩子
    step_hook = build_step_hook(args)
    
    # 执行
    result = pipeline.run(step_hook=step_hook)
    
    logger.success(f"✓ 分析完成! 报告: {result.report_path}")
    
    return result


def main():
    """
    CLI 主入口
    """
    args = parse_args()
    
    try:
        run_pipeline(args)
        return 0
    except KeyboardInterrupt:
        logger.warning("\n\n用户中断")
        return 1
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
