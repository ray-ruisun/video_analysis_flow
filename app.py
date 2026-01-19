#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Style Analysis - Gradio Web Interface
视频风格分析 - Gradio 网页界面

SOTA Models: CLIP | CLAP | HuBERT | Whisper | YOLO11
"""

import sys
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import gradio as gr
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))

from steps import (
    VisualAnalysisStep, AudioAnalysisStep, ASRAnalysisStep,
    YOLOAnalysisStep, ConsensusStep,
    VideoInput, AudioInput, ASRInput, YOLOInput, ConsensusInput,
    VideoMetrics, VisualOutput, AudioOutput, ASROutput, YOLOOutput, ConsensusOutput,
)
from report_word import generate_word_report

# =============================================================================
# 国际化 (i18n)
# =============================================================================
I18N = {
    "zh": {
        "title": "🎬 视频风格分析系统",
        "subtitle": "SOTA 2025/2026 | PyTorch + HuggingFace",
        "tech_stack": "CLIP (场景) | CLAP (音频) | HuBERT (情感) | Whisper (ASR) | YOLO11 (检测)",
        "tab_upload": "📤 上传与预览",
        "tab_analysis": "🔬 分析",
        "tab_results": "📊 结果与导出",
        "upload_video": "上传视频",
        "upload_label": "选择视频文件 (mp4, avi, mov, mkv)",
        "upload_status": "状态",
        "video_preview": "视频预览",
        "audio_preview": "音频预览",
        "audio_label": "提取的音频",
        "frame_preview": "关键帧预览",
        "frame_label": "均匀采样的关键帧",
        "settings": "设置",
        "asr_lang": "ASR 语言",
        "ui_lang": "界面语言",
        "analysis_ctrl": "分析控制",
        "run_all": "🎯 一键分析全部",
        "gen_report": "📄 生成报告",
        "export_json": "💾 导出 JSON",
        "step_exec": "分步执行",
        "visual_btn": "📹 视觉分析",
        "audio_btn": "🎵 音频分析",
        "asr_btn": "🎤 语音分析",
        "yolo_btn": "🔍 目标检测",
        "consensus_btn": "🎯 共识计算",
        "tab_visual": "📹 视觉",
        "tab_audio": "🎵 音频",
        "tab_asr": "🎤 语音",
        "tab_yolo": "🔍 检测",
        "tab_consensus": "🎯 共识",
        "report_gen": "报告生成",
        "report_status": "报告状态",
        "download": "下载",
        "word_report": "Word 报告",
        "pdf_report": "PDF 报告",
        "json_status": "JSON 状态",
        "json_data": "JSON 数据",
        "summary": "分析摘要",
        "summary_label": "快速预览",
        "summary_placeholder": "分析完成后显示摘要...",
        "hint_upload": "请先上传视频",
        "hint_consensus": "运行分析后自动生成",
        "uploaded": "✅ 视频已上传",
        "work_dir": "工作目录",
        "frames_extracted": "已提取 {n} 个关键帧",
        "audio_extracted": "✅ 音频已提取",
        "audio_failed": "⚠️ 音频提取失败",
        "please_upload": "❌ 请先上传视频",
        "analysis_failed": "分析失败",
        "visual_init": "⏳ 加载 CLIP 模型...",
        "visual_analyzing": "🔄 视觉分析中...",
        "audio_init": "⏳ 加载 CLAP 模型...",
        "audio_analyzing": "🔄 音频分析中...",
        "asr_init": "⏳ 加载 Whisper 模型...",
        "asr_analyzing": "🔄 语音识别中...",
        "yolo_init": "⏳ 加载 YOLO11 模型...",
        "yolo_analyzing": "🔄 目标检测中...",
        "consensus_calc": "🔄 计算共识...",
        "report_gen_word": "🔄 生成 Word 报告...",
        "report_gen_pdf": "🔄 转换为 PDF...",
        "done": "✅ 完成!",
        "report_done": "✅ 报告已生成",
        "pdf_failed": "⚠️ PDF 转换失败",
        "json_exported": "✅ JSON 已导出",
        "summary_title": "=== 分析摘要 ===",
        "footer": "Video Style Analysis | SOTA 2025/2026 | PyTorch + HuggingFace",
        # Results formatting
        "visual_result": "# 📹 视觉分析结果",
        "audio_result": "# 🎵 音频分析结果",
        "asr_result": "# 🎤 语音分析结果",
        "yolo_result": "# 🔍 目标检测结果",
        "consensus_result": "# 🎯 共识分析结果",
        "duration": "时长",
        "sampled_frames": "采样帧数",
        "cam_angle": "镜头角度",
        "hue": "色调",
        "saturation": "饱和度",
        "brightness": "亮度",
        "contrast": "对比度",
        "cct": "色温",
        "cuts": "剪辑数",
        "avg_shot_len": "平均镜头时长",
        "transition": "转场类型",
        "bpm": "BPM",
        "bgm_style": "BGM 风格",
        "main_mood": "主要情绪",
        "speech_ratio": "语音比例",
        "word_count": "词数",
        "speech_rate": "语速",
        "pace": "节奏",
        "env_type": "环境类型",
        "unique_obj": "检测物体",
        "confidence": "置信度",
    },
    "en": {
        "title": "🎬 Video Style Analysis",
        "subtitle": "SOTA 2025/2026 | PyTorch + HuggingFace",
        "tech_stack": "CLIP (Scene) | CLAP (Audio) | HuBERT (Emotion) | Whisper (ASR) | YOLO11 (Detection)",
        "tab_upload": "📤 Upload & Preview",
        "tab_analysis": "🔬 Analysis",
        "tab_results": "📊 Results & Export",
        "upload_video": "Upload Video",
        "upload_label": "Select video file (mp4, avi, mov, mkv)",
        "upload_status": "Status",
        "video_preview": "Video Preview",
        "audio_preview": "Audio Preview",
        "audio_label": "Extracted Audio",
        "frame_preview": "Key Frames Preview",
        "frame_label": "Uniformly Sampled Frames",
        "settings": "Settings",
        "asr_lang": "ASR Language",
        "ui_lang": "UI Language",
        "analysis_ctrl": "Analysis Control",
        "run_all": "🎯 Analyze All",
        "gen_report": "📄 Generate Report",
        "export_json": "💾 Export JSON",
        "step_exec": "Step by Step",
        "visual_btn": "📹 Visual",
        "audio_btn": "🎵 Audio",
        "asr_btn": "🎤 ASR",
        "yolo_btn": "🔍 YOLO",
        "consensus_btn": "🎯 Consensus",
        "tab_visual": "📹 Visual",
        "tab_audio": "🎵 Audio",
        "tab_asr": "🎤 ASR",
        "tab_yolo": "🔍 YOLO",
        "tab_consensus": "🎯 Consensus",
        "report_gen": "Report Generation",
        "report_status": "Report Status",
        "download": "Download",
        "word_report": "Word Report",
        "pdf_report": "PDF Report",
        "json_status": "JSON Status",
        "json_data": "JSON Data",
        "summary": "Analysis Summary",
        "summary_label": "Quick Preview",
        "summary_placeholder": "Summary will appear after analysis...",
        "hint_upload": "Please upload a video first",
        "hint_consensus": "Generated after analysis",
        "uploaded": "✅ Video uploaded",
        "work_dir": "Work directory",
        "frames_extracted": "Extracted {n} key frames",
        "audio_extracted": "✅ Audio extracted",
        "audio_failed": "⚠️ Audio extraction failed",
        "please_upload": "❌ Please upload a video first",
        "analysis_failed": "Analysis failed",
        "visual_init": "⏳ Loading CLIP model...",
        "visual_analyzing": "🔄 Visual analysis...",
        "audio_init": "⏳ Loading CLAP model...",
        "audio_analyzing": "🔄 Audio analysis...",
        "asr_init": "⏳ Loading Whisper model...",
        "asr_analyzing": "🔄 Speech recognition...",
        "yolo_init": "⏳ Loading YOLO11 model...",
        "yolo_analyzing": "🔄 Object detection...",
        "consensus_calc": "🔄 Calculating consensus...",
        "report_gen_word": "🔄 Generating Word report...",
        "report_gen_pdf": "🔄 Converting to PDF...",
        "done": "✅ Done!",
        "report_done": "✅ Report generated",
        "pdf_failed": "⚠️ PDF conversion failed",
        "json_exported": "✅ JSON exported",
        "summary_title": "=== Analysis Summary ===",
        "footer": "Video Style Analysis | SOTA 2025/2026 | PyTorch + HuggingFace",
        "visual_result": "# 📹 Visual Analysis Results",
        "audio_result": "# 🎵 Audio Analysis Results",
        "asr_result": "# 🎤 Speech Analysis Results",
        "yolo_result": "# 🔍 Object Detection Results",
        "consensus_result": "# 🎯 Consensus Analysis Results",
        "duration": "Duration",
        "sampled_frames": "Sampled frames",
        "cam_angle": "Camera angle",
        "hue": "Hue",
        "saturation": "Saturation",
        "brightness": "Brightness",
        "contrast": "Contrast",
        "cct": "CCT",
        "cuts": "Cuts",
        "avg_shot_len": "Avg shot length",
        "transition": "Transition",
        "bpm": "BPM",
        "bgm_style": "BGM style",
        "main_mood": "Main mood",
        "speech_ratio": "Speech ratio",
        "word_count": "Word count",
        "speech_rate": "Speech rate",
        "pace": "Pace",
        "env_type": "Environment",
        "unique_obj": "Objects detected",
        "confidence": "confidence",
    }
}

CURRENT_LANG = "zh"

def t(key: str) -> str:
    return I18N.get(CURRENT_LANG, I18N["zh"]).get(key, key)

def set_lang(lang: str):
    global CURRENT_LANG
    CURRENT_LANG = lang if lang in I18N else "zh"

# =============================================================================
# 全局状态
# =============================================================================
class AnalysisState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.video_path: Optional[Path] = None
        self.audio_path: Optional[Path] = None
        self.work_dir: Optional[Path] = None
        self.visual_output: Optional[VisualOutput] = None
        self.audio_output: Optional[AudioOutput] = None
        self.asr_output: Optional[ASROutput] = None
        self.yolo_output: Optional[YOLOOutput] = None
        self.consensus_output: Optional[ConsensusOutput] = None
        self.report_path: Optional[str] = None
        self.pdf_path: Optional[str] = None

STATE = AnalysisState()

# =============================================================================
# 工具函数
# =============================================================================
def extract_audio_from_video(video_path: Path, output_dir: Path) -> Optional[Path]:
    output_path = output_dir / f"{video_path.stem}_audio.wav"
    if output_path.exists():
        return output_path
    try:
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
               "-ar", "22050", "-ac", "1", str(output_path)]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except Exception:
        return None


def extract_frames_for_gallery(video_path: Path, output_dir: Path, num_frames: int = 12) -> List[str]:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        cap.release()
        return []
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frame_paths = []
    
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            timestamp = idx / fps if fps > 0 else 0
            frame_path = frames_dir / f"frame_{i:03d}_{timestamp:.1f}s.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
    
    cap.release()
    return frame_paths


def convert_docx_to_pdf(docx_path: str) -> Optional[str]:
    pdf_path = docx_path.replace('.docx', '.pdf')
    try:
        cmd = ["libreoffice", "--headless", "--convert-to", "pdf",
               "--outdir", str(Path(docx_path).parent), docx_path]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        if Path(pdf_path).exists():
            return pdf_path
    except Exception:
        pass
    return None


def format_distribution(detail: Dict) -> str:
    if not detail or 'distribution' not in detail:
        return ""
    lines = []
    for item in detail.get('distribution', [])[:5]:
        value = item.get('value', 'Unknown')
        count = item.get('count', 0)
        pct = item.get('percentage', 0)
        lines.append(f"  • {value}: {count}× ({pct}%)")
    return "\n".join(lines)


# =============================================================================
# 结果格式化
# =============================================================================
def format_visual_output(output: VisualOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    lines = [
        f"{t('visual_result')}\n",
        f"**{t('duration')}**: {output.duration:.2f}s | **FPS**: {output.fps:.1f} | **{t('sampled_frames')}**: {output.sampled_frames}\n",
        f"### 📷 镜头\n**{t('cam_angle')}**: {output.camera_angle}",
        format_distribution(output.camera_angle_detail),
        f"\n### 🎨 色彩\n**{t('hue')}**: {output.hue_family} | **{t('saturation')}**: {output.saturation_band} | **{t('brightness')}**: {output.brightness_band}",
    ]
    if output.cct_mean:
        lines.append(f"**{t('cct')}**: {output.cct_mean:.0f}K")
    
    lines.extend([
        f"\n### ✂️ 剪辑\n**{t('cuts')}**: {output.cuts} | **{t('avg_shot_len')}**: {output.avg_shot_length:.2f}s | **{t('transition')}**: {output.transition_type}",
        f"\n### 🏠 场景 (CLIP)",
    ])
    
    for scene in output.scene_categories[:3]:
        label = scene.get('label', 'Unknown')
        prob = scene.get('probability', 0)
        lines.append(f"  • {label}: {prob:.1%}")
    
    return "\n".join(lines)


def format_audio_output(output: AudioOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    lines = [
        f"{t('audio_result')}\n",
        f"### 🎵 节奏\n**{t('bpm')}**: {output.tempo_bpm:.1f} | **节拍**: {output.num_beats} | **打击乐比例**: {output.percussive_ratio:.2f}",
        f"\n### 🎸 BGM\n**{t('bgm_style')}**: {output.bgm_style} ({output.bgm_style_confidence:.1%})",
        f"\n### 😊 情绪\n**{t('main_mood')}**: {output.mood} ({output.mood_confidence:.1%})",
        f"\n### 其他\n**调式**: {output.key_signature} | **{t('speech_ratio')}**: {output.speech_ratio:.2f}",
    ]
    
    instruments = output.instruments.get('detected_instruments', [])
    if instruments:
        lines.append(f"**乐器**: {', '.join(instruments)}")
    
    return "\n".join(lines)


def format_asr_output(output: ASROutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    lines = [
        f"{t('asr_result')}\n",
        f"### 📝 统计\n**{t('word_count')}**: {output.num_words} | **{t('speech_rate')}**: {output.words_per_minute:.1f} wpm | **{t('pace')}**: {output.pace}",
    ]
    
    if output.catchphrases:
        lines.append(f"\n### 🔁 口头禅\n" + " | ".join([f'"{p}"' for p in output.catchphrases[:5]]))
    
    if output.prosody:
        lines.append(f"\n### 🎼 韵律\n**音高**: {output.prosody.get('mean_pitch_hz', 0):.1f} Hz | **风格**: {output.prosody.get('prosody_style', 'N/A')}")
    
    if output.emotion:
        lines.append(f"\n### 😊 情感\n**主要**: {output.emotion.get('dominant_emotion', 'N/A')} ({output.emotion.get('confidence', 0):.1%})")
    
    if output.text:
        text_preview = output.text[:300] + ('...' if len(output.text) > 300 else '')
        lines.append(f"\n### 📜 转录\n```\n{text_preview}\n```")
    
    return "\n".join(lines)


def format_yolo_output(output: YOLOOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    detection = output.detection
    environment = output.environment
    
    lines = [
        f"{t('yolo_result')}\n",
        f"### 🏠 环境\n**{t('env_type')}**: {environment.get('environment_type', 'N/A')} | **风格**: {environment.get('cooking_style', 'N/A')}",
        f"\n### 📊 统计\n**{t('unique_obj')}**: {detection.get('unique_objects', 0)} 种 | **总检测**: {detection.get('total_detections', 0)} 次",
        f"\n### 🎯 检测到的物体",
    ]
    
    object_counts = detection.get('object_counts', {})
    avg_conf = detection.get('avg_confidence', {})
    for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        conf = avg_conf.get(obj, 0)
        lines.append(f"  • **{obj}**: {count}× ({t('confidence')}: {conf:.1%})")
    
    return "\n".join(lines)


def format_consensus_output(output: ConsensusOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    lines = [
        f"{t('consensus_result')}\n",
        f"### 📷 镜头\n**角度**: {output.camera_angle} | **焦距**: {output.focal_length_tendency} | **运动**: {output.camera_motion}",
        f"\n### 🎨 色彩\n**{t('hue')}**: {output.hue_family} | **{t('saturation')}**: {output.saturation} | **{t('brightness')}**: {output.brightness}",
    ]
    if output.cct:
        lines.append(f"**{t('cct')}**: {output.cct:.0f}K")
    
    lines.append(f"\n### ✂️ 剪辑\n**{t('transition')}**: {output.transition_type}")
    if output.avg_shot_length:
        lines.append(f"**{t('avg_shot_len')}**: {output.avg_shot_length:.2f}s")
    
    lines.append(f"\n### 🎵 音频\n**{t('bgm_style')}**: {output.bgm_style} | **情绪**: {output.bgm_mood}")
    if output.tempo_bpm:
        lines.append(f"**{t('bpm')}**: {output.tempo_bpm:.1f}")
    
    if output.yolo_available:
        lines.append(f"\n### 🔍 YOLO\n**环境**: {output.yolo_environment} | **风格**: {output.yolo_style}")
    
    return "\n".join(lines)


# =============================================================================
# 处理函数
# =============================================================================
def upload_video(video_file):
    """处理视频上传 - 不返回到 video_input 避免刷新循环"""
    if video_file is None:
        return "", None, []
    
    STATE.reset()
    STATE.work_dir = Path(tempfile.mkdtemp(prefix="video_analysis_"))
    
    video_path = Path(video_file)
    STATE.video_path = STATE.work_dir / video_path.name
    
    import shutil
    shutil.copy(video_file, STATE.video_path)
    
    STATE.audio_path = extract_audio_from_video(STATE.video_path, STATE.work_dir)
    frame_paths = extract_frames_for_gallery(STATE.video_path, STATE.work_dir, num_frames=12)
    
    status = f"{t('uploaded')}: {video_path.name}\n"
    status += f"{t('work_dir')}: {STATE.work_dir}\n"
    status += t('frames_extracted').format(n=len(frame_paths)) + "\n"
    status += t('audio_extracted') if STATE.audio_path else t('audio_failed')
    
    audio_path = str(STATE.audio_path) if STATE.audio_path else None
    
    return status, audio_path, frame_paths


def run_visual_analysis(progress=gr.Progress()) -> Tuple[str, str]:
    if STATE.video_path is None:
        return t('please_upload'), None
    
    progress(0.1, desc=t('visual_init'))
    
    try:
        step = VisualAnalysisStep()
        input_data = VideoInput(video_path=STATE.video_path, work_dir=STATE.work_dir, frame_mode="edge")
        
        progress(0.4, desc=t('visual_analyzing'))
        STATE.visual_output = step.run(input_data)
        
        progress(1.0, desc=t('done'))
        result = format_visual_output(STATE.visual_output)
        contact_sheet = STATE.visual_output.contact_sheet if STATE.visual_output else None
        
        return result, contact_sheet
    except Exception as e:
        return f"❌ {t('analysis_failed')}: {str(e)}", None


def run_audio_analysis(progress=gr.Progress()) -> str:
    if STATE.audio_path is None:
        return t('please_upload')
    
    progress(0.1, desc=t('audio_init'))
    
    try:
        step = AudioAnalysisStep()
        input_data = AudioInput(audio_path=STATE.audio_path)
        
        progress(0.4, desc=t('audio_analyzing'))
        STATE.audio_output = step.run(input_data)
        
        progress(1.0, desc=t('done'))
        return format_audio_output(STATE.audio_output)
    except Exception as e:
        return f"❌ {t('analysis_failed')}: {str(e)}"


def run_asr_analysis(language: str, progress=gr.Progress()) -> str:
    if STATE.audio_path is None:
        return t('please_upload')
    
    progress(0.1, desc=t('asr_init'))
    
    try:
        step = ASRAnalysisStep()
        input_data = ASRInput(
            audio_path=STATE.audio_path,
            language=language,
            model_size="large-v3-turbo",
            enable_prosody=True,
            enable_emotion=True
        )
        
        progress(0.4, desc=t('asr_analyzing'))
        STATE.asr_output = step.run(input_data)
        
        progress(1.0, desc=t('done'))
        return format_asr_output(STATE.asr_output)
    except Exception as e:
        return f"❌ {t('analysis_failed')}: {str(e)}"


def run_yolo_analysis(progress=gr.Progress()) -> str:
    if STATE.video_path is None:
        return t('please_upload')
    
    progress(0.1, desc=t('yolo_init'))
    
    try:
        step = YOLOAnalysisStep()
        input_data = YOLOInput(video_path=STATE.video_path, target_frames=36,
                               enable_colors=True, enable_materials=True)
        
        progress(0.4, desc=t('yolo_analyzing'))
        STATE.yolo_output = step.run(input_data)
        
        progress(1.0, desc=t('done'))
        return format_yolo_output(STATE.yolo_output)
    except Exception as e:
        return f"❌ {t('analysis_failed')}: {str(e)}"


def run_consensus_analysis() -> str:
    metrics = VideoMetrics(path=str(STATE.video_path) if STATE.video_path else "")
    metrics.visual = STATE.visual_output
    metrics.audio = STATE.audio_output
    metrics.asr = STATE.asr_output
    metrics.yolo = STATE.yolo_output
    
    try:
        step = ConsensusStep()
        input_data = ConsensusInput(video_metrics=[metrics])
        STATE.consensus_output = step.run(input_data)
        return format_consensus_output(STATE.consensus_output)
    except Exception as e:
        return f"❌ {t('analysis_failed')}: {str(e)}"


def run_all_analysis(language: str, progress=gr.Progress()) -> Tuple:
    progress(0.05, desc=t('visual_init'))
    visual_result, contact_sheet = run_visual_analysis()
    
    progress(0.25, desc=t('audio_init'))
    audio_result = run_audio_analysis()
    
    progress(0.45, desc=t('asr_init'))
    asr_result = run_asr_analysis(language)
    
    progress(0.65, desc=t('yolo_init'))
    yolo_result = run_yolo_analysis()
    
    progress(0.85, desc=t('consensus_calc'))
    consensus_result = run_consensus_analysis()
    
    progress(1.0, desc=t('done'))
    
    summary_lines = [t('summary_title') + "\n"]
    if STATE.visual_output:
        summary_lines.append(f"📹 {t('cam_angle')}: {STATE.visual_output.camera_angle}")
        summary_lines.append(f"🎨 {t('hue')}: {STATE.visual_output.hue_family}")
        summary_lines.append(f"✂️ {t('cuts')}: {STATE.visual_output.cuts}")
    if STATE.audio_output:
        summary_lines.append(f"🎵 {t('bpm')}: {STATE.audio_output.tempo_bpm:.1f}")
        summary_lines.append(f"🎸 {t('bgm_style')}: {STATE.audio_output.bgm_style}")
    if STATE.asr_output:
        summary_lines.append(f"🎤 {t('speech_rate')}: {STATE.asr_output.words_per_minute:.1f} wpm")
    if STATE.yolo_output:
        obj_count = STATE.yolo_output.detection.get('unique_objects', 0)
        summary_lines.append(f"🔍 {t('unique_obj')}: {obj_count}")
    
    summary = "\n".join(summary_lines)
    return visual_result, contact_sheet, audio_result, asr_result, yolo_result, consensus_result, summary


def generate_report(progress=gr.Progress()) -> Tuple[str, str, str]:
    if STATE.video_path is None:
        return t('please_upload'), None, None
    
    progress(0.2, desc=t('report_gen_word'))
    
    try:
        metrics = VideoMetrics(path=str(STATE.video_path))
        metrics.visual = STATE.visual_output
        metrics.audio = STATE.audio_output
        metrics.asr = STATE.asr_output
        metrics.yolo = STATE.yolo_output
        
        if STATE.consensus_output is None:
            run_consensus_analysis()
        
        metrics_dict = metrics.to_dict()
        if STATE.visual_output:
            metrics_dict["visual"]["available"] = True
        if STATE.audio_output:
            metrics_dict["audio"]["available"] = True
        if STATE.asr_output:
            metrics_dict["asr"]["available"] = True
        if STATE.yolo_output:
            metrics_dict["yolo"]["available"] = True
        
        consensus_dict = STATE.consensus_output.to_dict() if STATE.consensus_output else {}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = STATE.work_dir / f"report_{timestamp}.docx"
        
        generate_word_report(
            video_metrics=[metrics_dict],
            consensus=consensus_dict,
            output_path=str(report_path),
            show_screenshots=True
        )
        
        STATE.report_path = str(report_path)
        
        progress(0.6, desc=t('report_gen_pdf'))
        STATE.pdf_path = convert_docx_to_pdf(STATE.report_path)
        
        progress(1.0, desc=t('done'))
        
        status = f"{t('report_done')}\n"
        status += f"📄 Word: {report_path.name}\n"
        status += f"📕 PDF: {Path(STATE.pdf_path).name}" if STATE.pdf_path else t('pdf_failed')
        
        return status, STATE.report_path, STATE.pdf_path
    except Exception as e:
        return f"❌ {t('analysis_failed')}: {str(e)}", None, None


def export_json() -> Tuple[str, str]:
    if STATE.video_path is None:
        return t('please_upload'), None
    
    try:
        data = {
            "timestamp": datetime.now().isoformat(),
            "video_path": str(STATE.video_path),
            "visual": STATE.visual_output.to_dict() if STATE.visual_output else None,
            "audio": STATE.audio_output.to_dict() if STATE.audio_output else None,
            "asr": STATE.asr_output.to_dict() if STATE.asr_output else None,
            "yolo": STATE.yolo_output.to_dict() if STATE.yolo_output else None,
            "consensus": STATE.consensus_output.to_dict() if STATE.consensus_output else None,
        }
        
        json_path = STATE.work_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return f"{t('json_exported')}: {json_path.name}", str(json_path)
    except Exception as e:
        return f"❌ {t('analysis_failed')}: {str(e)}", None


# =============================================================================
# Gradio 界面 - 使用顶级 Tabs 布局
# =============================================================================
CUSTOM_CSS = """
.main-container { max-width: 1400px; margin: auto; padding: 20px; }
.result-box { min-height: 400px; }
.step-btn { min-width: 100px; }
.upload-box { border: 2px dashed #ccc; border-radius: 10px; padding: 20px; }
"""

def create_ui():
    with gr.Blocks(title="Video Style Analysis") as demo:
        
        # 标题栏
        with gr.Row():
            with gr.Column(scale=8):
                gr.Markdown(f"# {t('title')}\n**{t('subtitle')}** | {t('tech_stack')}")
            with gr.Column(scale=2):
                lang_btn = gr.Radio(
                    choices=[("中文", "zh"), ("English", "en")],
                    value="zh",
                    label=t('ui_lang'),
                    interactive=True
                )
        
        # 主要内容 - 顶级 Tabs
        with gr.Tabs() as main_tabs:
            
            # ========== Tab 1: 上传与预览 ==========
            with gr.TabItem(t('tab_upload'), id="upload"):
                with gr.Row():
                    # 左侧：上传
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {t('upload_video')}")
                        video_input = gr.Video(
                            label=t('upload_label'),
                            height=350
                        )
                        upload_status = gr.Textbox(
                            label=t('upload_status'),
                            lines=4,
                            interactive=False
                        )
                        
                        gr.Markdown(f"### {t('settings')}")
                        language_select = gr.Dropdown(
                            choices=[("English", "en"), ("中文", "zh"), ("日本語", "ja"), ("한국어", "ko"), ("Auto", "auto")],
                            value="en",
                            label=t('asr_lang')
                        )
                    
                    # 右侧：预览
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem(t('audio_preview')):
                                audio_player = gr.Audio(
                                    label=t('audio_label'),
                                    type="filepath"
                                )
                            
                            with gr.TabItem(t('frame_preview')):
                                frame_gallery = gr.Gallery(
                                    label=t('frame_label'),
                                    columns=4,
                                    rows=3,
                                    height=400,
                                    object_fit="contain"
                                )
            
            # ========== Tab 2: 分析 ==========
            with gr.TabItem(t('tab_analysis'), id="analysis"):
                # 控制栏
                gr.Markdown(f"### {t('analysis_ctrl')}")
                with gr.Row():
                    run_all_btn = gr.Button(t('run_all'), variant="primary", size="lg", scale=2)
                    run_visual_btn = gr.Button(t('visual_btn'), variant="secondary")
                    run_audio_btn = gr.Button(t('audio_btn'), variant="secondary")
                    run_asr_btn = gr.Button(t('asr_btn'), variant="secondary")
                    run_yolo_btn = gr.Button(t('yolo_btn'), variant="secondary")
                    run_consensus_btn = gr.Button(t('consensus_btn'), variant="secondary")
                
                gr.Markdown("---")
                
                # 结果展示 - 子 Tabs
                with gr.Tabs() as result_tabs:
                    with gr.TabItem(t('tab_visual'), id="visual"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                visual_result = gr.Markdown(value=f"*{t('hint_upload')}*")
                            with gr.Column(scale=1):
                                contact_sheet_img = gr.Image(label="Contact Sheet", height=300)
                    
                    with gr.TabItem(t('tab_audio'), id="audio"):
                        audio_result = gr.Markdown(value=f"*{t('hint_upload')}*")
                    
                    with gr.TabItem(t('tab_asr'), id="asr"):
                        asr_result = gr.Markdown(value=f"*{t('hint_upload')}*")
                    
                    with gr.TabItem(t('tab_yolo'), id="yolo"):
                        yolo_result = gr.Markdown(value=f"*{t('hint_upload')}*")
                    
                    with gr.TabItem(t('tab_consensus'), id="consensus"):
                        consensus_result = gr.Markdown(value=f"*{t('hint_consensus')}*")
            
            # ========== Tab 3: 结果与导出 ==========
            with gr.TabItem(t('tab_results'), id="results"):
                with gr.Row():
                    # 摘要
                    with gr.Column(scale=2):
                        gr.Markdown(f"### {t('summary')}")
                        summary_box = gr.Textbox(
                            label=t('summary_label'),
                            lines=15,
                            interactive=False,
                            placeholder=t('summary_placeholder')
                        )
                    
                    # 报告生成
                    with gr.Column(scale=1):
                        gr.Markdown(f"### {t('report_gen')}")
                        
                        with gr.Row():
                            generate_report_btn = gr.Button(t('gen_report'), variant="primary", size="lg")
                            export_json_btn = gr.Button(t('export_json'), size="lg")
                        
                        report_status = gr.Textbox(label=t('report_status'), lines=4, interactive=False)
                        
                        gr.Markdown(f"### {t('download')}")
                        report_download = gr.File(label=t('word_report'))
                        pdf_download = gr.File(label=t('pdf_report'))
                        
                        gr.Markdown("---")
                        json_status = gr.Textbox(label=t('json_status'), lines=2, interactive=False)
                        json_download = gr.File(label=t('json_data'))
        
        # 页脚
        gr.Markdown(f"---\n{t('footer')}")
        
        # ========== 事件绑定 ==========
        
        # 上传视频 - 不返回到 video_input
        video_input.change(
            fn=upload_video,
            inputs=[video_input],
            outputs=[upload_status, audio_player, frame_gallery]
        )
        
        # 分析按钮
        run_visual_btn.click(fn=run_visual_analysis, outputs=[visual_result, contact_sheet_img])
        run_audio_btn.click(fn=run_audio_analysis, outputs=[audio_result])
        run_asr_btn.click(fn=run_asr_analysis, inputs=[language_select], outputs=[asr_result])
        run_yolo_btn.click(fn=run_yolo_analysis, outputs=[yolo_result])
        run_consensus_btn.click(fn=run_consensus_analysis, outputs=[consensus_result])
        
        # 一键分析
        run_all_btn.click(
            fn=run_all_analysis,
            inputs=[language_select],
            outputs=[visual_result, contact_sheet_img, audio_result, asr_result, yolo_result, consensus_result, summary_box]
        )
        
        # 报告和导出
        generate_report_btn.click(fn=generate_report, outputs=[report_status, report_download, pdf_download])
        export_json_btn.click(fn=export_json, outputs=[json_status, json_download])
        
        # 语言切换 (简化版 - 刷新页面后生效)
        def switch_lang(lang):
            set_lang(lang)
            return lang
        
        lang_btn.change(fn=switch_lang, inputs=[lang_btn], outputs=[lang_btn])
    
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video Style Analysis Web UI")
    parser.add_argument("--port", type=int, default=8088, help="Server port (default: 8088)")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=CUSTOM_CSS
    )
