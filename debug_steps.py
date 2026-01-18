#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化流水线调试脚本

用于逐步执行和调试每个分析模块。
每个步骤执行完毕后会暂停，等待用户确认后继续。

使用方法:
  python debug_steps.py -v video1.mp4 video2.mp4 video3.mp4 \
                        -a audio1.wav audio2.wav audio3.wav \
                        -o report.docx

  # 仅执行某些模块
  python debug_steps.py -v video1.mp4 video2.mp4 video3.mp4 \
                        --modules visual,yolo -o report.docx
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# 将 src 目录加入路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from steps import (
    # 步骤类
    VisualAnalysisStep,
    AudioAnalysisStep,
    ASRAnalysisStep,
    YOLOAnalysisStep,
    ConsensusStep,
    ReportGenerationStep,
    # 输入类型
    VideoInput,
    AudioInput,
    ASRInput,
    YOLOInput,
    ConsensusInput,
    ReportInput,
    # 数据类型
    VideoMetrics,
    VisualOutput,
    AudioOutput,
    ASROutput,
    YOLOOutput,
    ConsensusOutput,
)
from utils import setup_logger

# 初始化日志器
logger = setup_logger()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="模块化流水线调试脚本 - 逐步执行每个分析模块",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-v", "--videos",
        nargs="+",
        required=True,
        help="视频文件路径 (1-3个)"
    )
    
    parser.add_argument(
        "-a", "--audios",
        nargs="+",
        help="音频文件路径 (与视频数量对应)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="debug_report.docx",
        help="输出报告路径 (default: debug_report.docx)"
    )
    
    parser.add_argument(
        "--modules",
        default="visual,audio,asr,yolo",
        help="要执行的模块 (逗号分隔, default: visual,audio,asr,yolo)"
    )
    
    parser.add_argument(
        "--work-dir",
        default="work",
        help="工作目录 (default: work)"
    )
    
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="不暂停，连续执行所有步骤"
    )
    
    return parser.parse_args()


def pause(message: str = "按回车继续下一步..."):
    """暂停等待用户确认"""
    try:
        input(f"\n{message}")
    except EOFError:
        pass


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_output_summary(name: str, output):
    """打印输出摘要"""
    print(f"\n[{name}] 输出摘要:")
    if hasattr(output, 'to_dict'):
        data = output.to_dict()
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"  {key}: {{...}}")
            elif isinstance(value, list):
                print(f"  {key}: [{len(value)} items]")
            elif isinstance(value, str) and len(str(value)) > 50:
                print(f"  {key}: {str(value)[:50]}...")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"  {output}")


def run_visual_step(
    video_path: Path, 
    work_dir: Path, 
    should_pause: bool = True
) -> Optional[VisualOutput]:
    """
    执行视觉分析步骤
    
    输入: video_path (视频文件路径)
    输出: VisualOutput (色彩、镜头、构图、场景、剪辑等)
    """
    print_separator("Step 1: 视觉分析 (VisualAnalysisStep)")
    print(f"输入: {video_path}")
    
    step = VisualAnalysisStep()
    input_data = VideoInput(
        video_path=video_path,
        work_dir=work_dir,
        frame_mode="edge"
    )
    
    print(f"\n执行中...")
    output = step.run(input_data)
    
    print_output_summary("VisualOutput", output)
    print(f"\n关键结果:")
    print(f"  - 镜头角度: {output.camera_angle}")
    print(f"  - 焦距倾向: {output.focal_length_tendency}")
    print(f"  - 色调: {output.hue_family}")
    print(f"  - 饱和度: {output.saturation_band}")
    print(f"  - 亮度: {output.brightness_band}")
    print(f"  - 剪辑数: {output.cuts}")
    print(f"  - 平均镜头时长: {output.avg_shot_length:.2f}s")
    print(f"  - 时长: {output.duration:.2f}s")
    
    if should_pause:
        pause()
    
    return output


def run_audio_step(
    audio_path: Path, 
    should_pause: bool = True
) -> Optional[AudioOutput]:
    """
    执行音频分析步骤
    
    输入: audio_path (音频文件路径)
    输出: AudioOutput (BPM、节拍、能量、BGM风格、情绪等)
    """
    print_separator("Step 2: 音频分析 (AudioAnalysisStep)")
    print(f"输入: {audio_path}")
    
    step = AudioAnalysisStep()
    input_data = AudioInput(audio_path=audio_path)
    
    print(f"\n执行中...")
    output = step.run(input_data)
    
    print_output_summary("AudioOutput", output)
    print(f"\n关键结果:")
    print(f"  - BPM: {output.tempo_bpm:.1f}")
    print(f"  - 节拍数: {output.num_beats}")
    print(f"  - 打击乐比例: {output.percussive_ratio:.2f}")
    print(f"  - BGM风格: {output.bgm_style}")
    print(f"  - 情绪: {output.mood}")
    print(f"  - 调式: {output.key_signature}")
    print(f"  - 语音比例: {output.speech_ratio:.2f}")
    
    if should_pause:
        pause()
    
    return output


def run_asr_step(
    audio_path: Path, 
    should_pause: bool = True
) -> Optional[ASROutput]:
    """
    执行 ASR 分析步骤
    
    输入: audio_path (音频文件路径)
    输出: ASROutput (转录文本、语速、口头禅、停顿、韵律、情感等)
    """
    print_separator("Step 3: ASR 分析 (ASRAnalysisStep)")
    print(f"输入: {audio_path}")
    
    step = ASRAnalysisStep()
    input_data = ASRInput(
        audio_path=audio_path,
        language="en",
        model_size="small",
        enable_prosody=True,
        enable_emotion=True
    )
    
    print(f"\n执行中 (Whisper 转录可能需要一些时间)...")
    output = step.run(input_data)
    
    print_output_summary("ASROutput", output)
    print(f"\n关键结果:")
    print(f"  - 词数: {output.num_words}")
    print(f"  - 语速: {output.words_per_second:.2f} w/s ({output.words_per_minute:.1f} wpm)")
    print(f"  - 节奏: {output.pace}")
    print(f"  - 停顿风格: {output.pause_style}")
    print(f"  - 口头禅: {output.catchphrases[:5] if output.catchphrases else '无'}")
    if output.text:
        preview = output.text[:100] + "..." if len(output.text) > 100 else output.text
        print(f"  - 转录预览: {preview}")
    
    if should_pause:
        pause()
    
    return output


def run_yolo_step(
    video_path: Path, 
    should_pause: bool = True
) -> Optional[YOLOOutput]:
    """
    执行 YOLO 检测步骤
    
    输入: video_path (视频文件路径)
    输出: YOLOOutput (检测结果、环境分类、颜色、材质)
    """
    print_separator("Step 4: YOLO 检测 (YOLOAnalysisStep)")
    print(f"输入: {video_path}")
    
    step = YOLOAnalysisStep()
    input_data = YOLOInput(
        video_path=video_path,
        target_frames=36,
        enable_colors=True,
        enable_materials=True
    )
    
    print(f"\n执行中...")
    output = step.run(input_data)
    
    print_output_summary("YOLOOutput", output)
    print(f"\n关键结果:")
    detection = output.detection
    environment = output.environment
    print(f"  - 环境类型: {environment.get('environment_type', 'N/A')}")
    print(f"  - 烹饪风格: {environment.get('cooking_style', 'N/A')}")
    print(f"  - 检测物体类数: {detection.get('unique_objects', 0)}")
    print(f"  - 总检测次数: {detection.get('total_detections', 0)}")
    top_objects = detection.get('top_objects', [])
    if top_objects:
        print(f"  - Top 物体: {', '.join([f'{name}({count})' for name, count in top_objects[:5]])}")
    
    if should_pause:
        pause()
    
    return output


def run_consensus_step(
    video_metrics: List[VideoMetrics], 
    should_pause: bool = True
) -> Optional[ConsensusOutput]:
    """
    执行共识计算步骤
    
    输入: video_metrics (多个视频的分析结果列表)
    输出: ConsensusOutput (跨视频共识特征)
    """
    print_separator("Step 5: 共识计算 (ConsensusStep)")
    print(f"输入: {len(video_metrics)} 个视频的分析结果")
    
    step = ConsensusStep()
    input_data = ConsensusInput(video_metrics=video_metrics)
    
    print(f"\n执行中...")
    output = step.run(input_data)
    
    print_output_summary("ConsensusOutput", output)
    print(f"\n关键结果:")
    print(f"  - 共识镜头角度: {output.camera_angle}")
    print(f"  - 共识色调: {output.hue_family}")
    print(f"  - 共识场景: {output.scene_category}")
    print(f"  - 共识BGM风格: {output.bgm_style}")
    print(f"  - 共识情绪: {output.bgm_mood}")
    if output.cuts_per_minute:
        print(f"  - 每分钟剪辑数: {output.cuts_per_minute:.2f}")
    
    if should_pause:
        pause()
    
    return output


def run_report_step(
    video_metrics: List[VideoMetrics],
    consensus: ConsensusOutput,
    output_path: str,
    should_pause: bool = True
):
    """
    执行报告生成步骤
    
    输入: video_metrics, consensus, output_path
    输出: ReportOutput (报告文件路径)
    """
    print_separator("Step 6: 报告生成 (ReportGenerationStep)")
    print(f"输入: {len(video_metrics)} 个视频分析结果 + 共识")
    print(f"输出路径: {output_path}")
    
    step = ReportGenerationStep()
    input_data = ReportInput(
        video_metrics=video_metrics,
        consensus=consensus,
        output_path=output_path,
        show_screenshots=True
    )
    
    print(f"\n执行中...")
    output = step.run(input_data)
    
    print(f"\n关键结果:")
    print(f"  - 报告已保存: {output.report_path}")
    
    if should_pause:
        pause()
    
    return output


def main():
    """主函数 - 逐步执行所有模块"""
    args = parse_args()
    
    # 解析参数
    video_paths = [Path(p) for p in args.videos]
    audio_paths = [Path(p) for p in args.audios] if args.audios else None
    modules = [m.strip().lower() for m in args.modules.split(",")]
    work_dir = Path(args.work_dir)
    should_pause = not args.no_pause
    
    # 创建工作目录
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  模块化流水线调试脚本")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  - 视频: {[str(p) for p in video_paths]}")
    print(f"  - 音频: {[str(p) for p in audio_paths] if audio_paths else 'None'}")
    print(f"  - 模块: {modules}")
    print(f"  - 输出: {args.output}")
    print(f"  - 暂停模式: {'否' if args.no_pause else '是'}")
    
    if should_pause:
        pause("按回车开始执行...")
    
    # 存储所有视频的分析结果
    all_video_metrics: List[VideoMetrics] = []
    
    # 逐视频分析
    for i, video_path in enumerate(video_paths):
        print_separator(f"处理视频 {i+1}/{len(video_paths)}: {video_path.name}")
        
        audio_path = audio_paths[i] if audio_paths and i < len(audio_paths) else None
        
        # 创建 VideoMetrics 实例
        metrics = VideoMetrics(path=str(video_path))
        
        # 执行各模块
        if "visual" in modules:
            metrics.visual = run_visual_step(video_path, work_dir, should_pause)
        
        if "audio" in modules:
            if audio_path:
                metrics.audio = run_audio_step(audio_path, should_pause)
            else:
                print(f"\n[跳过] 音频分析 - 未提供音频文件")
        
        if "asr" in modules:
            if audio_path:
                metrics.asr = run_asr_step(audio_path, should_pause)
            else:
                print(f"\n[跳过] ASR分析 - 未提供音频文件")
        
        if "yolo" in modules:
            metrics.yolo = run_yolo_step(video_path, should_pause)
        
        all_video_metrics.append(metrics)
        print(f"\n✓ 视频 {video_path.name} 分析完成")
    
    # 共识计算
    if len(all_video_metrics) > 0:
        consensus = run_consensus_step(all_video_metrics, should_pause)
    else:
        print("\n[跳过] 共识计算 - 无分析结果")
        consensus = ConsensusOutput()
    
    # 报告生成
    run_report_step(all_video_metrics, consensus, args.output, should_pause)
    
    # 完成
    print_separator("全部完成!")
    print(f"\n报告已生成: {args.output}")
    print(f"分析了 {len(all_video_metrics)} 个视频")
    print(f"执行的模块: {', '.join(modules)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
