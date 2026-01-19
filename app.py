#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Style Analysis - Gradio Web Interface
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
# Internationalization (i18n)
# =============================================================================
TRANSLATIONS = {
    "en": {
        # Header
        "title": "🎬 Video Style Analysis",
        "subtitle": "SOTA 2025/2026 | PyTorch + HuggingFace",
        "models": "CLIP (Scene) · CLAP (Audio) · HuBERT (Emotion) · Whisper (ASR) · YOLO11 (Detection)",
        
        # Sections
        "upload_section": "📤 Upload",
        "settings_section": "⚙️ Settings",
        "preview_section": "🎬 Preview",
        "control_section": "🚀 Analysis",
        "results_section": "📊 Results",
        "export_section": "📥 Export",
        
        # Upload
        "select_video": "Select Video (mp4, avi, mov, mkv)",
        "status": "Status",
        "asr_language": "ASR Language",
        "audio_preview": "Extracted Audio",
        "keyframes": "Key Frames",
        
        # Buttons
        "analyze_all": "🎯 Analyze All",
        "btn_visual": "📹 Visual",
        "btn_audio": "🎵 Audio", 
        "btn_asr": "🎤 ASR",
        "btn_yolo": "🔍 YOLO",
        "btn_consensus": "📊 Summary",
        "gen_report": "📄 Generate Report",
        "export_json": "💾 Export JSON",
        
        # Tabs
        "tab_visual": "📹 Visual",
        "tab_audio": "🎵 Audio",
        "tab_asr": "🎤 ASR",
        "tab_yolo": "🔍 YOLO",
        "tab_summary": "📊 Summary",
        
        # Export
        "report_status": "Report Status",
        "word_report": "Word Report (.docx)",
        "pdf_report": "PDF Report (.pdf)",
        "json_data": "JSON Data",
        "json_status": "JSON Status",
        "quick_summary": "Quick Summary",
        
        # Messages
        "upload_first": "Please upload a video first",
        "run_analysis_first": "Please run visual or audio analysis first",
        "uploaded": "✅ Uploaded",
        "workdir": "Work Directory",
        "frames_extracted": "Extracted {n} frames",
        "audio_extracted": "✅ Audio extracted",
        "audio_failed": "⚠️ Audio extraction failed",
        "analysis_failed": "Analysis failed",
        "report_generated": "✅ Report generated",
        "pdf_needs_libreoffice": "⚠️ PDF conversion requires LibreOffice",
        "json_exported": "✅ JSON exported",
        
        # Progress
        "loading_clip": "⏳ Loading CLIP model...",
        "analyzing_visual": "🔄 Visual analysis...",
        "loading_clap": "⏳ Loading CLAP model...",
        "analyzing_audio": "🔄 Audio analysis...",
        "loading_whisper": "⏳ Loading Whisper model...",
        "analyzing_asr": "🔄 Speech recognition...",
        "loading_yolo": "⏳ Loading YOLO11 model...",
        "analyzing_yolo": "🔄 Object detection...",
        "calculating_consensus": "🔄 Calculating summary...",
        "generating_word": "📄 Generating Word report...",
        "converting_pdf": "📕 Converting to PDF...",
        "done": "✅ Done",
        
        # Results - Visual
        "visual_results": "Visual Analysis Results",
        "basic_info": "Basic Info",
        "duration": "Duration",
        "fps": "FPS",
        "sampled": "Sampled",
        "frames": "frames",
        "camera": "Camera",
        "angle": "Angle",
        "focal": "Focal",
        "color": "Color",
        "hue": "Hue",
        "saturation": "Saturation",
        "brightness": "Brightness",
        "contrast": "Contrast",
        "cct": "CCT",
        "editing": "Editing",
        "cuts": "Cuts",
        "avg_shot": "Avg Shot",
        "transition": "Transition",
        "scene_clip": "Scene (CLIP)",
        
        # Results - Audio
        "audio_results": "Audio Analysis Results (CLAP)",
        "rhythm": "Rhythm",
        "bpm": "BPM",
        "beats": "Beats",
        "percussive": "Percussive",
        "bgm_style": "BGM Style",
        "mood": "Mood",
        "key": "Key",
        "speech_ratio": "Speech Ratio",
        "instruments": "Instruments",
        
        # Results - ASR
        "asr_results": "Speech Analysis Results (Whisper + HuBERT)",
        "statistics": "Statistics",
        "words": "Words",
        "wpm": "WPM",
        "pace": "Pace",
        "catchphrases": "Catchphrases",
        "prosody": "Prosody",
        "pitch": "Pitch",
        "style": "Style",
        "emotion": "Emotion",
        "transcript": "Transcript",
        
        # Results - YOLO
        "yolo_results": "Object Detection Results (YOLO11)",
        "environment": "Environment",
        "env_type": "Type",
        "cook_style": "Style",
        "detection_stats": "Detection Stats",
        "unique_objects": "Unique Objects",
        "total_detections": "Total Detections",
        "detected_objects": "Detected Objects",
        "confidence": "confidence",
        
        # Results - Summary
        "summary_results": "Cross-Video Summary",
        "na": "N/A",
        
        # Footer
        "footer": "Video Style Analysis | SOTA 2025/2026 | PyTorch + HuggingFace",
    },
    "zh": {
        # Header
        "title": "🎬 视频风格分析系统",
        "subtitle": "SOTA 2025/2026 | PyTorch + HuggingFace",
        "models": "CLIP (场景) · CLAP (音频) · HuBERT (情感) · Whisper (语音) · YOLO11 (检测)",
        
        # Sections
        "upload_section": "📤 上传",
        "settings_section": "⚙️ 设置",
        "preview_section": "🎬 预览",
        "control_section": "🚀 分析",
        "results_section": "📊 结果",
        "export_section": "📥 导出",
        
        # Upload
        "select_video": "选择视频 (mp4, avi, mov, mkv)",
        "status": "状态",
        "asr_language": "语音识别语言",
        "audio_preview": "提取的音频",
        "keyframes": "关键帧",
        
        # Buttons
        "analyze_all": "🎯 一键分析全部",
        "btn_visual": "📹 视觉",
        "btn_audio": "🎵 音频",
        "btn_asr": "🎤 语音",
        "btn_yolo": "🔍 检测",
        "btn_consensus": "📊 汇总",
        "gen_report": "📄 生成报告",
        "export_json": "💾 导出 JSON",
        
        # Tabs
        "tab_visual": "📹 视觉",
        "tab_audio": "🎵 音频",
        "tab_asr": "🎤 语音",
        "tab_yolo": "🔍 检测",
        "tab_summary": "📊 汇总",
        
        # Export
        "report_status": "报告状态",
        "word_report": "Word 报告 (.docx)",
        "pdf_report": "PDF 报告 (.pdf)",
        "json_data": "JSON 数据",
        "json_status": "JSON 状态",
        "quick_summary": "快速摘要",
        
        # Messages
        "upload_first": "请先上传视频",
        "run_analysis_first": "请先运行视觉或音频分析",
        "uploaded": "✅ 已上传",
        "workdir": "工作目录",
        "frames_extracted": "已提取 {n} 帧",
        "audio_extracted": "✅ 音频已提取",
        "audio_failed": "⚠️ 音频提取失败",
        "analysis_failed": "分析失败",
        "report_generated": "✅ 报告已生成",
        "pdf_needs_libreoffice": "⚠️ PDF 转换需要 LibreOffice",
        "json_exported": "✅ JSON 已导出",
        
        # Progress
        "loading_clip": "⏳ 加载 CLIP 模型...",
        "analyzing_visual": "🔄 视觉分析中...",
        "loading_clap": "⏳ 加载 CLAP 模型...",
        "analyzing_audio": "🔄 音频分析中...",
        "loading_whisper": "⏳ 加载 Whisper 模型...",
        "analyzing_asr": "🔄 语音识别中...",
        "loading_yolo": "⏳ 加载 YOLO11 模型...",
        "analyzing_yolo": "🔄 目标检测中...",
        "calculating_consensus": "🔄 计算汇总...",
        "generating_word": "📄 生成 Word 报告...",
        "converting_pdf": "📕 转换为 PDF...",
        "done": "✅ 完成",
        
        # Results - Visual
        "visual_results": "视觉分析结果",
        "basic_info": "基本信息",
        "duration": "时长",
        "fps": "帧率",
        "sampled": "采样",
        "frames": "帧",
        "camera": "镜头",
        "angle": "角度",
        "focal": "焦距",
        "color": "色彩",
        "hue": "色调",
        "saturation": "饱和度",
        "brightness": "亮度",
        "contrast": "对比度",
        "cct": "色温",
        "editing": "剪辑",
        "cuts": "剪辑次数",
        "avg_shot": "平均镜头",
        "transition": "转场",
        "scene_clip": "场景 (CLIP)",
        
        # Results - Audio
        "audio_results": "音频分析结果 (CLAP)",
        "rhythm": "节奏",
        "bpm": "BPM",
        "beats": "节拍数",
        "percussive": "打击乐比例",
        "bgm_style": "BGM 风格",
        "mood": "情绪",
        "key": "调式",
        "speech_ratio": "语音比例",
        "instruments": "乐器",
        
        # Results - ASR
        "asr_results": "语音分析结果 (Whisper + HuBERT)",
        "statistics": "统计",
        "words": "词数",
        "wpm": "语速",
        "pace": "节奏",
        "catchphrases": "口头禅",
        "prosody": "韵律",
        "pitch": "音高",
        "style": "风格",
        "emotion": "情感",
        "transcript": "转录文本",
        
        # Results - YOLO
        "yolo_results": "目标检测结果 (YOLO11)",
        "environment": "环境",
        "env_type": "类型",
        "cook_style": "风格",
        "detection_stats": "检测统计",
        "unique_objects": "物体种类",
        "total_detections": "检测总数",
        "detected_objects": "检测到的物体",
        "confidence": "置信度",
        
        # Results - Summary
        "summary_results": "跨视频汇总",
        "na": "N/A",
        
        # Footer
        "footer": "视频风格分析 | SOTA 2025/2026 | PyTorch + HuggingFace",
    }
}

# Current language state
LANG = "en"

def t(key: str) -> str:
    """Get translated string"""
    return TRANSLATIONS.get(LANG, TRANSLATIONS["en"]).get(key, key)

def set_language(lang: str):
    """Set current language"""
    global LANG
    LANG = lang if lang in TRANSLATIONS else "en"


# =============================================================================
# Global State
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
# Utility Functions
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


# =============================================================================
# Result Formatters
# =============================================================================
def format_visual(output: VisualOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    scenes = "\n".join([
        f"  • {s.get('label', '?')}: **{s.get('probability', 0):.1%}**" 
        for s in output.scene_categories[:3]
    ])
    
    return f"""## 📹 {t('visual_results')}

### 📊 {t('basic_info')}
| {t('duration')} | {t('fps')} | {t('sampled')} |
|:---:|:---:|:---:|
| **{output.duration:.2f}s** | **{output.fps:.1f}** | **{output.sampled_frames}** {t('frames')} |

### 📷 {t('camera')}
| {t('angle')} | {t('focal')} |
|:---:|:---:|
| **{output.camera_angle}** | **{output.focal_length_tendency}** |

### 🎨 {t('color')}
| {t('hue')} | {t('saturation')} | {t('brightness')} | {t('contrast')} |
|:---:|:---:|:---:|:---:|
| **{output.hue_family}** | **{output.saturation_band}** | **{output.brightness_band}** | **{output.contrast}** |

{t('cct')}: **{output.cct_mean:.0f}K** | {t('cuts')}: **{output.cuts}** | {t('avg_shot')}: **{output.avg_shot_length:.2f}s** | {t('transition')}: **{output.transition_type}**

### 🏠 {t('scene_clip')}
{scenes}
"""


def format_audio(output: AudioOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    instruments = output.instruments.get('detected_instruments', [])
    inst_str = ", ".join(instruments) if instruments else t('na')
    
    return f"""## 🎵 {t('audio_results')}

### 🥁 {t('rhythm')}
| {t('bpm')} | {t('beats')} | {t('percussive')} |
|:---:|:---:|:---:|
| **{output.tempo_bpm:.1f}** | **{output.num_beats}** | **{output.percussive_ratio:.2f}** |

### 🎸 {t('bgm_style')}
**{output.bgm_style}** ({output.bgm_style_confidence:.1%})

### 😊 {t('mood')}
**{output.mood}** ({output.mood_confidence:.1%})

### 🎹 {t('key')}: **{output.key_signature}** | {t('speech_ratio')}: **{output.speech_ratio:.2f}**

### 🎺 {t('instruments')}
{inst_str}
"""


def format_asr(output: ASROutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    text_preview = output.text[:500] + '...' if len(output.text) > 500 else output.text
    
    emotion_str = ""
    if output.emotion:
        emotion_str = f"\n### 😊 {t('emotion')}\n**{output.emotion.get('dominant_emotion', t('na'))}** ({output.emotion.get('confidence', 0):.1%})"
    
    prosody_str = ""
    if output.prosody:
        prosody_str = f"\n### 🎼 {t('prosody')}\n{t('pitch')}: **{output.prosody.get('mean_pitch_hz', 0):.1f}Hz** | {t('style')}: **{output.prosody.get('prosody_style', t('na'))}**"
    
    catchphrases_str = ""
    if output.catchphrases:
        catchphrases_str = f"\n### 🔁 {t('catchphrases')}\n" + " · ".join([f'"{p}"' for p in output.catchphrases[:5]])
    
    return f"""## 🎤 {t('asr_results')}

### 📊 {t('statistics')}
| {t('words')} | {t('wpm')} | {t('pace')} |
|:---:|:---:|:---:|
| **{output.num_words}** | **{output.words_per_minute:.1f}** | **{output.pace}** |
{catchphrases_str}{prosody_str}{emotion_str}

### 📜 {t('transcript')}
```
{text_preview}
```
"""


def format_yolo(output: YOLOOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    detection = output.detection
    environment = output.environment
    object_counts = detection.get('object_counts', {})
    avg_conf = detection.get('avg_confidence', {})
    
    objects_str = "\n".join([
        f"| {obj} | {cnt} | {avg_conf.get(obj, 0):.1%} |"
        for obj, cnt in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ])
    
    return f"""## 🔍 {t('yolo_results')}

### 🏠 {t('environment')}
| {t('env_type')} | {t('cook_style')} |
|:---:|:---:|
| **{environment.get('environment_type', t('na'))}** | **{environment.get('cooking_style', t('na'))}** |

### 📊 {t('detection_stats')}
| {t('unique_objects')} | {t('total_detections')} |
|:---:|:---:|
| **{detection.get('unique_objects', 0)}** | **{detection.get('total_detections', 0)}** |

### 🎯 {t('detected_objects')}
| Object | Count | {t('confidence')} |
|:---|:---:|:---:|
{objects_str}
"""


def format_consensus(output: ConsensusOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    cct_str = f"{output.cct:.0f}K" if output.cct else t('na')
    shot_str = f"{output.avg_shot_length:.2f}s" if output.avg_shot_length else t('na')
    bpm_str = f"{output.tempo_bpm:.1f}" if output.tempo_bpm else t('na')
    
    return f"""## 📊 {t('summary_results')}

### 📷 {t('camera')}
| {t('angle')} | {t('focal')} | Motion |
|:---:|:---:|:---:|
| **{output.camera_angle}** | **{output.focal_length_tendency}** | **{output.camera_motion}** |

### 🎨 {t('color')}
| {t('hue')} | {t('saturation')} | {t('brightness')} |
|:---:|:---:|:---:|
| **{output.hue_family}** | **{output.saturation}** | **{output.brightness}** |

{t('cct')}: **{cct_str}** | {t('avg_shot')}: **{shot_str}** | {t('transition')}: **{output.transition_type}**

### 🎵 Audio
| {t('bgm_style')} | {t('mood')} | {t('bpm')} |
|:---:|:---:|:---:|
| **{output.bgm_style}** | **{output.bgm_mood}** | **{bpm_str}** |

### 🏠 Scene: **{output.scene_category}**
"""


# =============================================================================
# Processing Functions
# =============================================================================
def upload_video(video_file):
    if video_file is None:
        return t('upload_first'), None, []
    
    STATE.reset()
    STATE.work_dir = Path(tempfile.mkdtemp(prefix="video_analysis_"))
    
    video_path = Path(video_file)
    STATE.video_path = STATE.work_dir / video_path.name
    
    import shutil
    shutil.copy(video_file, STATE.video_path)
    
    STATE.audio_path = extract_audio_from_video(STATE.video_path, STATE.work_dir)
    frame_paths = extract_frames_for_gallery(STATE.video_path, STATE.work_dir, num_frames=12)
    
    status = f"{t('uploaded')}: {video_path.name}\n"
    status += f"{t('workdir')}: {STATE.work_dir}\n"
    status += t('frames_extracted').format(n=len(frame_paths)) + "\n"
    status += t('audio_extracted') if STATE.audio_path else t('audio_failed')
    
    audio_path = str(STATE.audio_path) if STATE.audio_path else None
    return status, audio_path, frame_paths


def run_visual(progress=gr.Progress()):
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}", None
    
    progress(0.1, desc=t('loading_clip'))
    step = VisualAnalysisStep()
    input_data = VideoInput(video_path=STATE.video_path, work_dir=STATE.work_dir, frame_mode="edge")
    
    progress(0.4, desc=t('analyzing_visual'))
    STATE.visual_output = step.run(input_data)
    
    progress(1.0, desc=t('done'))
    contact = STATE.visual_output.contact_sheet if STATE.visual_output else None
    return format_visual(STATE.visual_output), contact


def run_audio(progress=gr.Progress()):
    if STATE.audio_path is None:
        return f"❌ {t('upload_first')}"
    
    progress(0.1, desc=t('loading_clap'))
    step = AudioAnalysisStep()
    input_data = AudioInput(audio_path=STATE.audio_path)
    
    progress(0.4, desc=t('analyzing_audio'))
    STATE.audio_output = step.run(input_data)
    
    progress(1.0, desc=t('done'))
    return format_audio(STATE.audio_output)


def run_asr(language: str, progress=gr.Progress()):
    if STATE.audio_path is None:
        return f"❌ {t('upload_first')}"
    
    progress(0.1, desc=t('loading_whisper'))
    step = ASRAnalysisStep()
    input_data = ASRInput(audio_path=STATE.audio_path, language=language,
                          model_size="large-v3-turbo", enable_prosody=True, enable_emotion=True)
    
    progress(0.4, desc=t('analyzing_asr'))
    STATE.asr_output = step.run(input_data)
    
    progress(1.0, desc=t('done'))
    return format_asr(STATE.asr_output)


def run_yolo(progress=gr.Progress()):
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}"
    
    progress(0.1, desc=t('loading_yolo'))
    step = YOLOAnalysisStep()
    input_data = YOLOInput(video_path=STATE.video_path, target_frames=36,
                           enable_colors=True, enable_materials=True)
    
    progress(0.4, desc=t('analyzing_yolo'))
    STATE.yolo_output = step.run(input_data)
    
    progress(1.0, desc=t('done'))
    return format_yolo(STATE.yolo_output)


def run_consensus():
    if STATE.visual_output is None and STATE.audio_output is None:
        return f"❌ {t('run_analysis_first')}"
    
    metrics = VideoMetrics(path=str(STATE.video_path) if STATE.video_path else "")
    metrics.visual = STATE.visual_output
    metrics.audio = STATE.audio_output
    metrics.asr = STATE.asr_output
    metrics.yolo = STATE.yolo_output
    
    step = ConsensusStep()
    input_data = ConsensusInput(video_metrics=[metrics])
    STATE.consensus_output = step.run(input_data)
    
    return format_consensus(STATE.consensus_output)


def run_all(language: str, progress=gr.Progress()):
    progress(0.1, desc=t('analyzing_visual'))
    visual_result, contact = run_visual()
    
    progress(0.3, desc=t('analyzing_audio'))
    audio_result = run_audio()
    
    progress(0.5, desc=t('analyzing_asr'))
    asr_result = run_asr(language)
    
    progress(0.7, desc=t('analyzing_yolo'))
    yolo_result = run_yolo()
    
    progress(0.9, desc=t('calculating_consensus'))
    consensus_result = run_consensus()
    
    progress(1.0, desc=t('done'))
    
    # Generate summary
    lines = ["=" * 30, t('quick_summary'), "=" * 30, ""]
    if STATE.visual_output:
        lines.append(f"📹 {t('angle')}: {STATE.visual_output.camera_angle}")
        lines.append(f"🎨 {t('hue')}: {STATE.visual_output.hue_family}")
        lines.append(f"✂️ {t('cuts')}: {STATE.visual_output.cuts}")
    if STATE.audio_output:
        lines.append(f"🎵 {t('bpm')}: {STATE.audio_output.tempo_bpm:.1f}")
        lines.append(f"🎸 {t('bgm_style')}: {STATE.audio_output.bgm_style}")
    if STATE.asr_output:
        lines.append(f"🎤 {t('wpm')}: {STATE.asr_output.words_per_minute:.1f}")
    if STATE.yolo_output:
        lines.append(f"🔍 Objects: {STATE.yolo_output.detection.get('unique_objects', 0)}")
    
    summary = "\n".join(lines)
    return visual_result, contact, audio_result, asr_result, yolo_result, consensus_result, summary


def gen_report(progress=gr.Progress()):
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}", None, None
    
    if STATE.visual_output is None and STATE.audio_output is None:
        return f"❌ {t('run_analysis_first')}", None, None
    
    progress(0.2, desc=t('generating_word'))
    
    metrics = VideoMetrics(path=str(STATE.video_path))
    metrics.visual = STATE.visual_output
    metrics.audio = STATE.audio_output
    metrics.asr = STATE.asr_output
    metrics.yolo = STATE.yolo_output
    
    if STATE.consensus_output is None:
        run_consensus()
    
    metrics_dict = metrics.to_dict()
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
    
    progress(0.7, desc=t('converting_pdf'))
    STATE.pdf_path = convert_docx_to_pdf(STATE.report_path)
    
    progress(1.0, desc=t('done'))
    
    status = f"{t('report_generated')}\n📄 {report_path.name}"
    if STATE.pdf_path:
        status += f"\n📕 {Path(STATE.pdf_path).name}"
    else:
        status += f"\n{t('pdf_needs_libreoffice')}"
    
    return status, STATE.report_path, STATE.pdf_path


def export_json():
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}", None
    
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


def switch_language(lang: str):
    """Switch UI language and return updated labels"""
    set_language(lang)
    return (
        f"# {t('title')}\n**{t('subtitle')}**\n\n{t('models')}",
        t('analyze_all'),
        t('btn_visual'),
        t('btn_audio'),
        t('btn_asr'),
        t('btn_yolo'),
        t('btn_consensus'),
        t('gen_report'),
        t('export_json'),
        t('footer'),
    )


# =============================================================================
# Gradio UI
# =============================================================================
def create_ui():
    with gr.Blocks(
        title="Video Style Analysis",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
        .markdown-text { font-size: 14px; }
        .result-markdown { min-height: 400px; }
        """
    ) as demo:
        
        # Header
        header_md = gr.Markdown(f"# {t('title')}\n**{t('subtitle')}**\n\n{t('models')}")
        
        # Language selector
        with gr.Row():
            lang_radio = gr.Radio(
                choices=[("English", "en"), ("中文", "zh")],
                value="en",
                label="Language / 语言",
                scale=1
            )
        
        gr.Markdown("---")
        
        with gr.Row():
            # ========== Left Column: Upload & Settings ==========
            with gr.Column(scale=1, min_width=300):
                gr.Markdown(f"### {t('upload_section')}")
                video_input = gr.Video(label=t('select_video'), height=200)
                upload_status = gr.Textbox(label=t('status'), lines=4, interactive=False)
                
                gr.Markdown(f"### {t('settings_section')}")
                language_select = gr.Dropdown(
                    choices=[("English", "en"), ("中文", "zh"), ("日本語", "ja"), ("Auto", "auto")],
                    value="en",
                    label=t('asr_language')
                )
                
                gr.Markdown(f"### {t('preview_section')}")
                audio_player = gr.Audio(label=t('audio_preview'), type="filepath")
                frame_gallery = gr.Gallery(label=t('keyframes'), columns=3, height=150, object_fit="contain")
            
            # ========== Middle Column: Analysis & Results ==========
            with gr.Column(scale=2, min_width=500):
                gr.Markdown(f"### {t('control_section')}")
                
                with gr.Row():
                    run_all_btn = gr.Button(t('analyze_all'), variant="primary", size="lg", scale=2)
                
                with gr.Row():
                    run_visual_btn = gr.Button(t('btn_visual'), size="sm")
                    run_audio_btn = gr.Button(t('btn_audio'), size="sm")
                    run_asr_btn = gr.Button(t('btn_asr'), size="sm")
                    run_yolo_btn = gr.Button(t('btn_yolo'), size="sm")
                    run_consensus_btn = gr.Button(t('btn_consensus'), size="sm")
                
                gr.Markdown(f"### {t('results_section')}")
                
                with gr.Tabs():
                    with gr.Tab(t('tab_visual')):
                        visual_result = gr.Markdown(f"*{t('upload_first')}*", elem_classes="result-markdown")
                        contact_img = gr.Image(label="Contact Sheet", height=150)
                    
                    with gr.Tab(t('tab_audio')):
                        audio_result = gr.Markdown(f"*{t('upload_first')}*", elem_classes="result-markdown")
                    
                    with gr.Tab(t('tab_asr')):
                        asr_result = gr.Markdown(f"*{t('upload_first')}*", elem_classes="result-markdown")
                    
                    with gr.Tab(t('tab_yolo')):
                        yolo_result = gr.Markdown(f"*{t('upload_first')}*", elem_classes="result-markdown")
                    
                    with gr.Tab(t('tab_summary')):
                        consensus_result = gr.Markdown(f"*{t('run_analysis_first')}*", elem_classes="result-markdown")
            
            # ========== Right Column: Export ==========
            with gr.Column(scale=1, min_width=280):
                gr.Markdown(f"### {t('export_section')}")
                
                with gr.Row():
                    gen_report_btn = gr.Button(t('gen_report'), variant="secondary", size="sm")
                    export_json_btn = gr.Button(t('export_json'), size="sm")
                
                report_status = gr.Textbox(label=t('report_status'), lines=3, interactive=False)
                
                report_file = gr.File(label=t('word_report'))
                pdf_file = gr.File(label=t('pdf_report'))
                
                gr.Markdown("---")
                
                json_status = gr.Textbox(label=t('json_status'), lines=2, interactive=False)
                json_file = gr.File(label=t('json_data'))
                
                gr.Markdown("---")
                
                summary_box = gr.Textbox(label=t('quick_summary'), lines=10, interactive=False)
        
        # Footer
        footer_md = gr.Markdown(f"---\n{t('footer')}")
        
        # ========== Event Handlers ==========
        video_input.change(
            fn=upload_video,
            inputs=[video_input],
            outputs=[upload_status, audio_player, frame_gallery]
        )
        
        run_visual_btn.click(fn=run_visual, outputs=[visual_result, contact_img])
        run_audio_btn.click(fn=run_audio, outputs=[audio_result])
        run_asr_btn.click(fn=run_asr, inputs=[language_select], outputs=[asr_result])
        run_yolo_btn.click(fn=run_yolo, outputs=[yolo_result])
        run_consensus_btn.click(fn=run_consensus, outputs=[consensus_result])
        
        run_all_btn.click(
            fn=run_all,
            inputs=[language_select],
            outputs=[visual_result, contact_img, audio_result, asr_result, 
                     yolo_result, consensus_result, summary_box]
        )
        
        gen_report_btn.click(fn=gen_report, outputs=[report_status, report_file, pdf_file])
        export_json_btn.click(fn=export_json, outputs=[json_status, json_file])
        
        # Language switch
        lang_radio.change(
            fn=switch_language,
            inputs=[lang_radio],
            outputs=[
                header_md,
                run_all_btn,
                run_visual_btn,
                run_audio_btn,
                run_asr_btn,
                run_yolo_btn,
                run_consensus_btn,
                gen_report_btn,
                export_json_btn,
                footer_md,
            ]
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video Style Analysis Web UI")
    parser.add_argument("--port", type=int, default=8088, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "zh"], help="Default language")
    args = parser.parse_args()
    
    set_language(args.lang)
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )
