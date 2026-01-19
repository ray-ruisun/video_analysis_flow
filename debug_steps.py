#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化流水线调试脚本

用于逐步执行和调试每个分析模块。
每个步骤执行完毕后会暂停，等待用户确认后继续。

================================================================================
命令行使用方法
================================================================================

# 完整流程 (自动从视频提取音频)
python debug_steps.py -v video1.mp4 video2.mp4 video3.mp4 -o report.docx

# 仅执行某些模块
python debug_steps.py -v video1.mp4 --modules visual,yolo -o report.docx

# 单个视频调试
python debug_steps.py -v video1.mp4 --modules visual -o report.docx

# 不暂停，连续执行
python debug_steps.py -v video1.mp4 video2.mp4 video3.mp4 --no-pause -o report.docx

================================================================================
单独使用每个模块 (Python API)
================================================================================

# 0. 准备工作 - 从视频提取音频
from debug_steps import extract_audio_from_video
audio_path = extract_audio_from_video(Path("video.mp4"), Path("work"))
# -> work/video_audio.wav

# 1. 视觉分析模块 (VisualAnalysisStep)
# 输入: VideoInput (video_path, work_dir, frame_mode)
# 输出: VisualOutput (camera_angle, hue_family, cuts, duration, etc.)
from steps import VisualAnalysisStep, VideoInput
step = VisualAnalysisStep()
input_data = VideoInput(video_path=Path("video.mp4"), work_dir=Path("work"))
output = step.run(input_data)
print(f"镜头角度: {output.camera_angle}")
print(f"色调: {output.hue_family}")
print(f"剪辑数: {output.cuts}")

# 2. 音频分析模块 (AudioAnalysisStep)
# 输入: AudioInput (audio_path)
# 输出: AudioOutput (tempo_bpm, bgm_style, mood, key_signature, etc.)
from steps import AudioAnalysisStep, AudioInput
step = AudioAnalysisStep()
input_data = AudioInput(audio_path=Path("audio.wav"))  # 或从视频提取的音频
output = step.run(input_data)
print(f"BPM: {output.tempo_bpm}")
print(f"BGM风格: {output.bgm_style}")
print(f"情绪: {output.mood}")

# 3. ASR 语音识别模块 (ASRAnalysisStep)
# 输入: ASRInput (audio_path, language, model_size, enable_prosody, enable_emotion)
# 输出: ASROutput (text, words_per_minute, pace, catchphrases, prosody, emotion)
from steps import ASRAnalysisStep, ASRInput
step = ASRAnalysisStep()
input_data = ASRInput(
    audio_path=Path("audio.wav"),
    language="en",            # 语言: en, zh, ja, etc.
    model_size="small",       # 模型: tiny, base, small, medium, large
    enable_prosody=True,      # 韵律分析
    enable_emotion=True       # 情感分析
)
output = step.run(input_data)
print(f"转录: {output.text[:100]}...")
print(f"语速: {output.words_per_minute:.1f} wpm")
print(f"口头禅: {output.catchphrases}")

# 4. YOLO 目标检测模块 (YOLOAnalysisStep)
# 输入: YOLOInput (video_path, target_frames, enable_colors, enable_materials)
# 输出: YOLOOutput (detection, environment, colors, materials)
from steps import YOLOAnalysisStep, YOLOInput
step = YOLOAnalysisStep()
input_data = YOLOInput(
    video_path=Path("video.mp4"),
    target_frames=36,         # 采样帧数
    enable_colors=True,       # 颜色分析
    enable_materials=True     # 材质分析
)
output = step.run(input_data)
print(f"环境类型: {output.environment.get('environment_type')}")
print(f"检测物体: {output.detection.get('unique_objects')} 类")

# 5. 共识计算模块 (ConsensusStep)
# 输入: ConsensusInput (video_metrics: List[VideoMetrics])
# 输出: ConsensusOutput (camera_angle, hue_family, bgm_style, etc. 的多数票/中位数)
from steps import ConsensusStep, ConsensusInput, VideoMetrics
step = ConsensusStep()
input_data = ConsensusInput(video_metrics=[vm1, vm2, vm3])
output = step.run(input_data)
print(f"共识镜头: {output.camera_angle}")
print(f"共识BGM: {output.bgm_style}")

# 6. 报告生成模块 (ReportGenerationStep)
# 输入: ReportInput (video_metrics, consensus, output_path, show_screenshots)
# 输出: ReportOutput (report_path)
from steps import ReportGenerationStep, ReportInput
step = ReportGenerationStep()
input_data = ReportInput(
    video_metrics=[vm1, vm2, vm3],
    consensus=consensus_output,
    output_path="report.docx",
    show_screenshots=True
)
output = step.run(input_data)
print(f"报告: {output.report_path}")

================================================================================
"""

import sys
import subprocess
import argparse
import json
import datetime
from pathlib import Path
from typing import List, Optional, Any

# 将 src 目录加入路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ============================================================================
# 日志文件配置
# ============================================================================
LOG_FILE = None
LOG_DATA = {"runs": [], "timestamp": None}

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


def extract_audio_from_video(video_path: Path, work_dir: Path) -> Path:
    """
    从视频文件中提取音频
    
    使用 ffmpeg 将视频中的音频提取为 22.05kHz mono wav 文件。
    
    Args:
        video_path: 视频文件路径
        work_dir: 工作目录 (音频文件将保存在这里)
        
    Returns:
        Path: 提取的音频文件路径
        
    Raises:
        RuntimeError: 如果 ffmpeg 不可用或提取失败
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出音频文件路径
    audio_path = work_dir / f"{video_path.stem}_audio.wav"
    
    # 如果已存在，直接返回
    if audio_path.exists():
        logger.info(f"音频文件已存在，跳过提取: {audio_path}")
        return audio_path
    
    logger.info(f"从视频提取音频: {video_path} -> {audio_path}")
    
    # 使用 ffmpeg 提取音频
    # -y: 覆盖已存在的文件
    # -i: 输入文件
    # -vn: 不处理视频
    # -acodec pcm_s16le: 16-bit PCM 编码
    # -ar 22050: 采样率 22.05kHz
    # -ac 1: 单声道
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "22050",
        "-ac", "1",
        str(audio_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"音频提取成功: {audio_path}")
        return audio_path
    except FileNotFoundError:
        logger.error("ffmpeg 未安装或不在 PATH 中")
        logger.error("请安装 ffmpeg: https://ffmpeg.org/download.html")
        logger.error("  Ubuntu: sudo apt install ffmpeg")
        logger.error("  macOS: brew install ffmpeg")
        raise RuntimeError("ffmpeg not found")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg 提取音频失败: {e.stderr}")
        raise RuntimeError(f"Audio extraction failed: {e.stderr}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="模块化流水线调试脚本 - 逐步执行每个分析模块",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整调试 (自动从视频提取音频)
  python debug_steps.py -v video1.mp4 video2.mp4 video3.mp4 -o report.docx

  # 仅视觉分析
  python debug_steps.py -v video1.mp4 --modules visual -o report.docx

  # 视觉 + YOLO (不需要音频)
  python debug_steps.py -v video1.mp4 video2.mp4 video3.mp4 --modules visual,yolo -o report.docx

  # 连续执行，不暂停
  python debug_steps.py -v video1.mp4 --no-pause -o report.docx
        """
    )
    
    parser.add_argument(
        "-v", "--videos",
        nargs="+",
        required=True,
        help="视频文件路径 (1-3个，音频将自动从视频中提取)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="debug_report.docx",
        help="输出报告路径 (default: debug_report.docx)"
    )
    
    parser.add_argument(
        "--modules",
        default="visual,audio,asr,yolo",
        help="要执行的模块，逗号分隔 (default: visual,audio,asr,yolo)"
    )
    
    parser.add_argument(
        "--work-dir",
        default="work",
        help="工作目录，用于存放提取的音频和中间文件 (default: work)"
    )
    
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="不暂停，连续执行所有步骤"
    )
    
    parser.add_argument(
        "--skip-audio-extract",
        action="store_true",
        help="跳过音频提取 (如果工作目录已有音频文件)"
    )
    
    return parser.parse_args()


def pause(message: str = "按回车继续下一步..."):
    """暂停等待用户确认"""
    try:
        input(f"\n{message}")
    except EOFError:
        pass


def init_log_file(work_dir: Path) -> Path:
    """初始化日志文件"""
    global LOG_FILE, LOG_DATA
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = work_dir / f"debug_output_{timestamp}.json"
    LOG_DATA = {
        "timestamp": timestamp,
        "start_time": datetime.datetime.now().isoformat(),
        "runs": []
    }
    
    # 创建工作目录
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📝 日志文件: {LOG_FILE}")
    return LOG_FILE


def log_step_output(step_name: str, input_data: Any, output_data: Any):
    """记录步骤输出到日志文件"""
    global LOG_DATA, LOG_FILE
    
    if LOG_FILE is None:
        return
    
    import numpy as np
    
    # 转换为可序列化的格式
    def to_serializable(obj):
        # 处理 numpy 类型
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # 处理 dataclass 和普通对象
        elif hasattr(obj, 'to_dict'):
            return to_serializable(obj.to_dict())
        elif hasattr(obj, '__dict__') and not isinstance(obj, type):
            return {k: to_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    log_entry = {
        "step": step_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "input": to_serializable(input_data),
        "output": to_serializable(output_data)
    }
    
    LOG_DATA["runs"].append(log_entry)
    
    # 实时写入文件
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(LOG_DATA, f, indent=2, ensure_ascii=False)


def print_separator(title: str):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_dict_detailed(data: dict, indent: int = 2, max_list_items: int = 20):
    """递归打印字典的详细内容"""
    prefix = " " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_dict_detailed(value, indent + 2)
        elif isinstance(value, list):
            if len(value) == 0:
                print(f"{prefix}{key}: []")
            elif len(value) <= max_list_items:
                if all(isinstance(item, (int, float, str)) for item in value):
                    # 简单类型列表，显示所有
                    print(f"{prefix}{key}: {value}")
                else:
                    # 复杂类型列表，逐个显示
                    print(f"{prefix}{key}: [{len(value)} items]")
                    for i, item in enumerate(value[:max_list_items]):
                        if isinstance(item, dict):
                            print(f"{prefix}  [{i}]:")
                            print_dict_detailed(item, indent + 4)
                        else:
                            print(f"{prefix}  [{i}]: {item}")
            else:
                print(f"{prefix}{key}: [{len(value)} items, showing first {max_list_items}]")
                for i, item in enumerate(value[:max_list_items]):
                    if isinstance(item, dict):
                        print(f"{prefix}  [{i}]:")
                        print_dict_detailed(item, indent + 4)
                    else:
                        print(f"{prefix}  [{i}]: {item}")
        elif isinstance(value, str) and len(str(value)) > 200:
            print(f"{prefix}{key}: {str(value)[:200]}... ({len(value)} chars)")
        else:
            print(f"{prefix}{key}: {value}")


def print_output_summary(name: str, output, show_full: bool = True):
    """打印输出摘要 (完整详细版)"""
    print(f"\n[{name}] 完整输出:")
    print("-" * 60)
    
    if hasattr(output, 'to_dict'):
        data = output.to_dict()
        print_dict_detailed(data)
    elif isinstance(output, dict):
        print_dict_detailed(output)
    else:
        print(f"  {output}")
    
    print("-" * 60)


def run_visual_step(
    video_path: Path, 
    work_dir: Path, 
    should_pause: bool = True
) -> Optional[VisualOutput]:
    """
    执行视觉分析步骤
    
    输入: VideoInput (video_path, work_dir, frame_mode)
    输出: VisualOutput (camera_angle, hue_family, cuts, duration, etc.)
    """
    print_separator("Step: 视觉分析 (VisualAnalysisStep)")
    print(f"输入: VideoInput(video_path={video_path})")
    
    step = VisualAnalysisStep()
    input_data = VideoInput(
        video_path=video_path,
        work_dir=work_dir,
        frame_mode="edge"
    )
    
    print(f"\n执行中...")
    output = step.run(input_data)
    
    # 记录到日志文件
    log_step_output("visual", {"video_path": str(video_path)}, output)
    
    print_output_summary("VisualOutput", output)
    
    # 显示详细分布
    print(f"\n📊 详细分布:")
    print(f"\n  镜头角度分布:")
    if hasattr(output, 'camera_angle_detail') and output.camera_angle_detail:
        for item in output.camera_angle_detail.get('distribution', []):
            print(f"    - {item['value']}: {item['count']}次 ({item['percentage']}%)")
    
    print(f"\n  色调分布:")
    if hasattr(output, 'hue_detail') and output.hue_detail:
        for item in output.hue_detail.get('distribution', []):
            print(f"    - {item['value']}: {item['count']}次 ({item['percentage']}%)")
    
    print(f"\n  饱和度分布:")
    if hasattr(output, 'saturation_detail') and output.saturation_detail:
        for item in output.saturation_detail.get('distribution', []):
            print(f"    - {item['value']}: {item['count']}次 ({item['percentage']}%)")
    
    print(f"\n  亮度分布:")
    if hasattr(output, 'brightness_detail') and output.brightness_detail:
        for item in output.brightness_detail.get('distribution', []):
            print(f"    - {item['value']}: {item['count']}次 ({item['percentage']}%)")
    
    print(f"\n  对比度分布:")
    if hasattr(output, 'contrast_detail') and output.contrast_detail:
        for item in output.contrast_detail.get('distribution', []):
            print(f"    - {item['value']}: {item['count']}次 ({item['percentage']}%)")
    
    print(f"\n  光线类型:")
    if output.lighting and output.lighting.get('type_detail'):
        for item in output.lighting['type_detail'].get('distribution', []):
            print(f"    - {item['value']}: {item['count']}次 ({item['percentage']}%)")
    
    print(f"\n📈 关键数值:")
    print(f"  - 总时长: {output.duration:.2f}s")
    print(f"  - 采样帧数: {output.sampled_frames}")
    print(f"  - 剪辑数: {output.cuts}")
    print(f"  - 平均镜头时长: {output.avg_shot_length:.2f}s")
    if output.cct_mean:
        print(f"  - 色温: {output.cct_mean:.0f}K (±{output.cct_std:.0f})")
    
    if should_pause:
        pause()
    
    return output


def run_audio_step(
    audio_path: Path, 
    should_pause: bool = True
) -> Optional[AudioOutput]:
    """
    执行音频分析步骤
    
    输入: AudioInput (audio_path)
    输出: AudioOutput (tempo_bpm, bgm_style, mood, key_signature, etc.)
    """
    print_separator("Step: 音频分析 (AudioAnalysisStep)")
    print(f"输入: AudioInput(audio_path={audio_path})")
    
    step = AudioAnalysisStep()
    input_data = AudioInput(audio_path=audio_path)
    
    print(f"\n执行中...")
    output = step.run(input_data)
    
    # 记录到日志文件
    log_step_output("audio", {"audio_path": str(audio_path)}, output)
    
    print_output_summary("AudioOutput", output)
    
    # 显示详细分类结果
    print(f"\n📊 CLAP 分类详情:")
    print(f"\n  BGM 风格:")
    print(f"    - 主要风格: {output.bgm_style}")
    if hasattr(output, 'bgm_style_detail') and output.bgm_style_detail:
        top3 = output.bgm_style_detail.get('top_3', [])
        if top3:
            print(f"    - Top 3 风格:")
            for item in top3:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    print(f"        {item[0]}: {item[1]:.1%}")
    
    print(f"\n  情绪分析:")
    print(f"    - 主要情绪: {output.mood}")
    if hasattr(output, 'mood_detail') and output.mood_detail:
        top3 = output.mood_detail.get('top_3', [])
        if top3:
            print(f"    - Top 3 情绪:")
            for item in top3:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    print(f"        {item[0]}: {item[1]:.1%}")
    
    print(f"\n📈 基础指标:")
    print(f"  - BPM: {output.tempo_bpm:.1f}")
    print(f"  - 节拍数: {output.num_beats}")
    print(f"  - 打击乐比例: {output.percussive_ratio:.2f}")
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
    
    输入: ASRInput (audio_path, language, model_size, enable_prosody, enable_emotion)
    输出: ASROutput (text, words_per_minute, pace, catchphrases, prosody, emotion)
    """
    print_separator("Step: ASR 语音识别 (ASRAnalysisStep)")
    print(f"输入: ASRInput(audio_path={audio_path}, language='en', model_size='large-v3-turbo')")
    
    step = ASRAnalysisStep()
    input_data = ASRInput(
        audio_path=audio_path,
        language="en",
        model_size="large-v3-turbo",  # 使用最新最强模型
        enable_prosody=True,
        enable_emotion=True
    )
    
    print(f"\n执行中 (Whisper large-v3-turbo 转录中)...")
    output = step.run(input_data)
    
    # 记录到日志文件
    log_step_output("asr", {"audio_path": str(audio_path)}, output)
    
    print_output_summary("ASROutput", output)
    
    print(f"\n📊 ASR 详情:")
    print(f"\n  转录统计:")
    print(f"    - 词数: {output.num_words}")
    print(f"    - 语速: {output.words_per_second:.2f} w/s ({output.words_per_minute:.1f} wpm)")
    print(f"    - 节奏: {output.pace}")
    print(f"    - 停顿数: {output.num_pauses}")
    print(f"    - 停顿风格: {output.pause_style}")
    
    print(f"\n  口头禅 (高频短语):")
    if output.catchphrases:
        for phrase in output.catchphrases[:10]:
            print(f"    - {phrase}")
    else:
        print(f"    - 无")
    
    if output.prosody:
        print(f"\n  韵律分析:")
        print(f"    - 平均音高: {output.prosody.get('mean_pitch_hz', 0):.1f} Hz")
        print(f"    - 音高变化: {output.prosody.get('pitch_std', 0):.1f}")
        print(f"    - 音调: {output.prosody.get('tone', 'N/A')}")
        print(f"    - 韵律风格: {output.prosody.get('prosody_style', 'N/A')}")
    
    if output.emotion:
        print(f"\n  情感分析 (HuBERT):")
        print(f"    - 主要情感: {output.emotion.get('dominant_emotion', 'N/A')}")
        print(f"    - 置信度: {output.emotion.get('confidence', 0):.1%}")
        emotion_scores = output.emotion.get('emotion_scores', {})
        if emotion_scores:
            print(f"    - 情感分布:")
            for emotion, score in list(emotion_scores.items())[:5]:
                print(f"        {emotion}: {score:.1%}")
    
    if output.text:
        print(f"\n  转录文本 (前500字):")
        preview = output.text[:500] + "..." if len(output.text) > 500 else output.text
        print(f"    {preview}")
    
    if should_pause:
        pause()
    
    return output


def run_yolo_step(
    video_path: Path, 
    should_pause: bool = True
) -> Optional[YOLOOutput]:
    """
    执行 YOLO 检测步骤 (YOLO11)
    
    输入: YOLOInput (video_path, target_frames, enable_colors, enable_materials)
    输出: YOLOOutput (detection, environment, colors, materials)
    """
    print_separator("Step: YOLO11 目标检测 (YOLOAnalysisStep)")
    print(f"输入: YOLOInput(video_path={video_path}, target_frames=36, model=yolo11s.pt)")
    
    step = YOLOAnalysisStep()
    input_data = YOLOInput(
        video_path=video_path,
        target_frames=36,
        enable_colors=True,
        enable_materials=True
    )
    
    print(f"\n执行中 (YOLO11 检测中)...")
    output = step.run(input_data)
    
    # 记录到日志文件
    log_step_output("yolo", {"video_path": str(video_path)}, output)
    
    print_output_summary("YOLOOutput", output)
    
    print(f"\n📊 YOLO11 检测详情:")
    detection = output.detection
    environment = output.environment
    
    print(f"\n  环境分析:")
    print(f"    - 环境类型: {environment.get('environment_type', 'N/A')}")
    print(f"    - 烹饪风格: {environment.get('cooking_style', 'N/A')}")
    print(f"    - 设备档次: {environment.get('appliance_tier', 'N/A')}")
    
    print(f"\n  检测统计:")
    print(f"    - 检测物体类数: {detection.get('unique_objects', 0)}")
    print(f"    - 总检测次数: {detection.get('total_detections', 0)}")
    print(f"    - 处理帧数: {detection.get('frames_processed', 0)}")
    
    print(f"\n  检测到的物体:")
    object_counts = detection.get('object_counts', {})
    for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
        avg_conf = detection.get('avg_confidence', {}).get(obj, 0)
        print(f"    - {obj}: {count}次 (置信度: {avg_conf:.1%})")
    
    # 颜色分析
    colors = output.colors
    if colors and colors.get('detailed_analysis'):
        print(f"\n  物体颜色分析:")
        for obj, analysis in colors.get('detailed_analysis', {}).items():
            print(f"    {obj}:")
            for item in analysis.get('distribution', []):
                print(f"      - {item['color']}: {item['count']}次 ({item['percentage']}%)")
    
    # 材质分析
    materials = output.materials
    if materials and materials.get('detailed_analysis'):
        print(f"\n  物体材质分析:")
        for obj, analysis in materials.get('detailed_analysis', {}).items():
            print(f"    {obj}:")
            for item in analysis.get('distribution', []):
                print(f"      - {item['material']}: {item['count']}次 ({item['percentage']}%)")
    
    if should_pause:
        pause()
    
    return output


def run_consensus_step(
    video_metrics: List[VideoMetrics], 
    should_pause: bool = True
) -> Optional[ConsensusOutput]:
    """
    执行共识计算步骤
    
    输入: ConsensusInput (video_metrics: List[VideoMetrics])
    输出: ConsensusOutput (camera_angle, hue_family, bgm_style 的多数票/中位数)
    """
    print_separator("Step: 共识计算 (ConsensusStep)")
    print(f"输入: ConsensusInput(video_metrics=[{len(video_metrics)} 个视频])")
    
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
    
    输入: ReportInput (video_metrics, consensus, output_path, show_screenshots)
    输出: ReportOutput (report_path)
    """
    print_separator("Step: 报告生成 (ReportGenerationStep)")
    print(f"输入: ReportInput(video_metrics=[{len(video_metrics)} 个], output_path={output_path})")
    
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
    modules = [m.strip().lower() for m in args.modules.split(",")]
    work_dir = Path(args.work_dir)
    should_pause = not args.no_pause
    
    # 检查是否需要音频
    needs_audio = any(m in modules for m in ("audio", "asr"))
    
    # 创建工作目录
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志文件
    log_file = init_log_file(work_dir)
    
    print("\n" + "=" * 70)
    print("  模块化流水线调试脚本 (SOTA 2025/2026)")
    print("=" * 70)
    print(f"\n📋 配置:")
    print(f"  - 视频: {[str(p) for p in video_paths]}")
    print(f"  - 模块: {modules}")
    print(f"  - 输出报告: {args.output}")
    print(f"  - 工作目录: {work_dir}")
    print(f"  - 日志文件: {log_file}")
    print(f"  - 暂停模式: {'否' if args.no_pause else '是'}")
    print(f"  - 需要音频: {'是' if needs_audio else '否'}")
    print(f"\n🔧 使用的模型:")
    print(f"  - 场景分类: CLIP (openai/clip-vit-large-patch14)")
    print(f"  - 音频分类: CLAP (laion/larger_clap_music_and_speech)")
    print(f"  - 语音情感: HuBERT (superb/hubert-large-superb-er)")
    print(f"  - ASR: Whisper large-v3-turbo")
    print(f"  - 目标检测: YOLO11 (yolo11s.pt)")
    
    # 提取音频 (如果需要)
    audio_paths: List[Optional[Path]] = []
    if needs_audio:
        print_separator("预处理: 从视频提取音频")
        for video_path in video_paths:
            try:
                audio_path = extract_audio_from_video(video_path, work_dir)
                audio_paths.append(audio_path)
            except Exception as e:
                logger.error(f"无法从 {video_path} 提取音频: {e}")
                audio_paths.append(None)
    else:
        audio_paths = [None] * len(video_paths)
    
    if should_pause:
        pause("按回车开始执行模块...")
    
    # 存储所有视频的分析结果
    all_video_metrics: List[VideoMetrics] = []
    
    # 逐视频分析
    for i, video_path in enumerate(video_paths):
        print_separator(f"处理视频 {i+1}/{len(video_paths)}: {video_path.name}")
        
        audio_path = audio_paths[i]
        
        # 创建 VideoMetrics 实例
        metrics = VideoMetrics(path=str(video_path))
        
        # 执行各模块
        if "visual" in modules:
            metrics.visual = run_visual_step(video_path, work_dir, should_pause)
        
        if "audio" in modules:
            if audio_path and audio_path.exists():
                metrics.audio = run_audio_step(audio_path, should_pause)
            else:
                print(f"\n[跳过] 音频分析 - 音频提取失败或不存在")
        
        if "asr" in modules:
            if audio_path and audio_path.exists():
                metrics.asr = run_asr_step(audio_path, should_pause)
            else:
                print(f"\n[跳过] ASR分析 - 音频提取失败或不存在")
        
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
    if needs_audio:
        print(f"提取的音频文件保存在: {work_dir}/")


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
