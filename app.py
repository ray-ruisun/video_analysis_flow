#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Style Analysis - Gradio Web Interface
SOTA Models: CLIP | CLAP | HuBERT | Whisper | YOLO11 | Deep-Fake-Detector-v2
"""

import sys
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

import gradio as gr
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import PipelineConfig, get_default_config
from steps import (
    VisualAnalysisStep, AudioAnalysisStep, ASRAnalysisStep,
    YOLOAnalysisStep, ConsensusStep, AIDetectionStep,
    VideoInput, AudioInput, ASRInput, YOLOInput, ConsensusInput, AIDetectionInput,
    VideoMetrics, VisualOutput, AudioOutput, ASROutput, YOLOOutput, ConsensusOutput, AIDetectionOutput,
)
from report_word import generate_word_report

# =============================================================================
# Internationalization (i18n)
# =============================================================================
TRANSLATIONS = {
    "en": {
        "title": "🎬 Video Style Analysis",
        "subtitle": "SOTA 2025/2026 | PyTorch + HuggingFace",
        "models": "CLIP · CLAP · HuBERT · Whisper · YOLO11 · DeepFake-v2",
        "upload_section": "📤 Upload",
        "settings_section": "⚙️ Settings",
        "preview_section": "🎬 Preview",
        "control_section": "🚀 Analysis",
        "results_section": "📊 Results",
        "export_section": "📥 Export",
        "config_section": "🔧 Configuration",
        "select_video": "Select Video (mp4, avi, mov, mkv)",
        "status": "Status",
        "asr_language": "ASR Language",
        "audio_preview": "Extracted Audio",
        "keyframes": "Key Frames",
        "analyze_all": "🎯 Analyze All",
        "analyze_current": "🎯 Analyze Current Video",
        "analyze_batch": "📈 Analyze All Videos (Cross-Video)",
        "batch_hint": "*Use 'Analyze Current' for single video, 'Analyze All' for multi-video comparison*",
        "btn_visual": "📹 Camera & Color",
        "btn_audio": "🎵 BGM & Tempo",
        "btn_asr": "🎤 Speech & Emotion",
        "btn_yolo": "🔍 Objects",
        "btn_consensus": "📊 Summary",
        "btn_ai_detect": "🤖 AI Detect",
        "gen_report": "📄 Report",
        "export_json": "💾 JSON",
        "tab_upload": "📤 Upload & Preview",
        "tab_analysis": "🚀 Run Analysis",
        "tab_export": "📥 Export & Reports",
        "tab_config": "⚙️ Settings & Weights",
        "tab_visual": "📹 Camera & Color",
        "tab_audio": "🎵 BGM & Tempo",
        "tab_asr": "🎤 Speech & Emotion",
        "tab_yolo": "🔍 Objects & Materials",
        "tab_summary": "📊 Analysis Summary",
        "video_list": "Video List",
        "add_video": "➕ Add Video",
        "clear_all": "🗑️ Clear All",
        "delete_video": "🗑️ Delete",
        "single_video_mode": "Single Video Analysis",
        "multi_video_mode": "Cross-Video Comparison ({n} videos)",
        "no_videos": "No videos added",
        "video_n": "Video {n}",
        "select_video": "Select video to view",
        "comparison_chart": "Comparison Chart",
        "tab_ai": "🤖 AI/Deepfake Detection",
        "report_status": "Report Status",
        "word_report": "Word Report",
        "pdf_report": "PDF Report",
        "json_data": "JSON Data",
        "json_status": "JSON Status",
        "quick_summary": "Quick Summary",
        "upload_first": "Please upload a video first",
        "run_analysis_first": "Please run analysis first",
        "uploaded": "✅ Uploaded",
        "workdir": "Work Directory",
        "frames_extracted": "Extracted {n} frames",
        "audio_extracted": "✅ Audio extracted",
        "audio_failed": "⚠️ Audio extraction failed",
        "analysis_failed": "Analysis failed",
        "report_generated": "✅ Report generated",
        "json_exported": "✅ JSON exported",
        "loading": "Loading",
        "analyzing": "Analyzing",
        "done": "✅ Done",
        "footer": "Video Style Analysis | SOTA 2025/2026",
        # Config labels
        "visual_frames": "Visual: Target Frames",
        "visual_scene_threshold": "Visual: Scene Threshold",
        "audio_sample_rate": "Audio: Sample Rate",
        "asr_model": "ASR: Whisper Model",
        "asr_beam_size": "ASR: Beam Size",
        "yolo_model": "YOLO: Model",
        "yolo_conf": "YOLO: Confidence",
        "yolo_frames": "YOLO: Target Frames",
        "ai_enabled": "AI Detection: Enabled",
        "ai_video_model": "AI: Video Model",
        "ai_video_threshold": "AI: Video Threshold",
        "ai_frame_enabled": "AI: Frame Detection",
        "ai_frame_threshold": "AI: Frame Threshold",
        "ai_face_enabled": "AI: Face Detection",
        "ai_video_weight": "AI: Video Weight",
        "ai_frame_weight": "AI: Frame Weight",
        "ai_face_weight": "AI: Face Weight",
        # Results
        "visual_results": "Visual Analysis Results",
        "audio_results": "Audio Analysis Results",
        "asr_results": "Speech Analysis Results",
        "yolo_results": "Object Detection Results",
        "summary_results": "Summary Results",
        "ai_results": "AI Detection Results",
        "verdict": "Verdict",
        "confidence": "Confidence",
        "real": "Real",
        "deepfake": "Deepfake",
        "synthetic": "Synthetic",
        "suspicious": "Suspicious",
        "already_added": "Already in list",
        "add_more_videos": "➕ Add More Videos",
    },
    "zh": {
        "title": "🎬 视频风格分析系统",
        "subtitle": "SOTA 2025/2026 | PyTorch + HuggingFace",
        "models": "CLIP · CLAP · HuBERT · Whisper · YOLO11 · DeepFake-v2",
        "upload_section": "📤 上传",
        "settings_section": "⚙️ 设置",
        "preview_section": "🎬 预览",
        "control_section": "🚀 分析",
        "results_section": "📊 结果",
        "export_section": "📥 导出",
        "config_section": "🔧 参数配置",
        "select_video": "选择视频 (mp4, avi, mov, mkv)",
        "status": "状态",
        "asr_language": "语音识别语言",
        "audio_preview": "提取的音频",
        "keyframes": "关键帧",
        "analyze_all": "🎯 一键分析",
        "analyze_current": "🎯 分析当前视频",
        "analyze_batch": "📈 分析全部视频 (跨视频对比)",
        "batch_hint": "*单视频分析使用'分析当前'，多视频对比使用'分析全部'*",
        "btn_visual": "📹 镜头色彩",
        "btn_audio": "🎵 背景音乐",
        "btn_asr": "🎤 语音情感",
        "btn_yolo": "🔍 物体检测",
        "btn_consensus": "📊 综合汇总",
        "btn_ai_detect": "🤖 AI检测",
        "gen_report": "📄 生成报告",
        "export_json": "💾 导出JSON",
        "tab_upload": "📤 上传与预览",
        "tab_analysis": "🚀 运行分析",
        "tab_export": "📥 导出与报告",
        "tab_config": "⚙️ 参数设置",
        "tab_visual": "📹 镜头与色彩",
        "tab_audio": "🎵 背景音乐与节奏",
        "tab_asr": "🎤 语音与情感",
        "tab_yolo": "🔍 物体与材质",
        "tab_summary": "📊 综合汇总",
        "video_list": "视频列表",
        "add_video": "➕ 添加视频",
        "clear_all": "🗑️ 清空全部",
        "delete_video": "🗑️ 删除",
        "single_video_mode": "单视频分析",
        "multi_video_mode": "跨视频对比 ({n} 个视频)",
        "no_videos": "未添加视频",
        "video_n": "视频 {n}",
        "select_video": "选择视频查看",
        "comparison_chart": "对比图表",
        "tab_ai": "🤖 AI生成检测",
        "report_status": "报告状态",
        "word_report": "Word 报告",
        "pdf_report": "PDF 报告",
        "json_data": "JSON 数据",
        "json_status": "JSON 状态",
        "quick_summary": "快速摘要",
        "upload_first": "请先上传视频",
        "run_analysis_first": "请先运行分析",
        "uploaded": "✅ 已上传",
        "workdir": "工作目录",
        "frames_extracted": "已提取 {n} 帧",
        "audio_extracted": "✅ 音频已提取",
        "audio_failed": "⚠️ 音频提取失败",
        "analysis_failed": "分析失败",
        "report_generated": "✅ 报告已生成",
        "json_exported": "✅ JSON 已导出",
        "loading": "加载中",
        "analyzing": "分析中",
        "done": "✅ 完成",
        "footer": "视频风格分析 | SOTA 2025/2026",
        # Config labels
        "visual_frames": "视觉: 目标帧数",
        "visual_scene_threshold": "视觉: 场景阈值",
        "audio_sample_rate": "音频: 采样率",
        "asr_model": "ASR: Whisper 模型",
        "asr_beam_size": "ASR: Beam Size",
        "yolo_model": "YOLO: 模型",
        "yolo_conf": "YOLO: 置信度",
        "yolo_frames": "YOLO: 目标帧数",
        "ai_enabled": "AI检测: 启用",
        "ai_video_model": "AI: 视频模型",
        "ai_video_threshold": "AI: 视频阈值",
        "ai_frame_enabled": "AI: 帧检测",
        "ai_frame_threshold": "AI: 帧阈值",
        "ai_face_enabled": "AI: 人脸检测",
        "ai_video_weight": "AI: 视频权重",
        "ai_frame_weight": "AI: 帧权重",
        "ai_face_weight": "AI: 人脸权重",
        # Results
        "visual_results": "视觉分析结果",
        "audio_results": "音频分析结果",
        "asr_results": "语音分析结果",
        "yolo_results": "目标检测结果",
        "summary_results": "汇总结果",
        "ai_results": "AI生成检测结果",
        "verdict": "判定",
        "confidence": "置信度",
        "real": "真实",
        "deepfake": "深度伪造",
        "synthetic": "合成",
        "suspicious": "可疑",
        "already_added": "已在列表中",
        "add_more_videos": "➕ 添加更多视频",
    }
}

LANG = "en"

def t(key: str) -> str:
    return TRANSLATIONS.get(LANG, TRANSLATIONS["en"]).get(key, key)

def set_language(lang: str):
    global LANG
    LANG = lang if lang in TRANSLATIONS else "en"


# =============================================================================
# Global State
# =============================================================================
@dataclass
class VideoAnalysis:
    """Single video analysis result container"""
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    work_dir: Optional[Path] = None
    visual_output: Optional[VisualOutput] = None
    audio_output: Optional[AudioOutput] = None
    asr_output: Optional[ASROutput] = None
    yolo_output: Optional[YOLOOutput] = None
    ai_output: Optional[AIDetectionOutput] = None
    
    def to_metrics(self) -> VideoMetrics:
        """Convert to VideoMetrics for consensus calculation"""
        metrics = VideoMetrics(path=str(self.video_path) if self.video_path else "")
        metrics.visual = self.visual_output
        metrics.audio = self.audio_output
        metrics.asr = self.asr_output
        metrics.yolo = self.yolo_output
        return metrics


class AnalysisState:
    def __init__(self):
        self.reset()
        self.config = get_default_config()
    
    def reset(self):
        # Multi-video support
        self.videos: List[VideoAnalysis] = []
        self.current_index: int = 0  # Currently selected video
        
        # Legacy single-video properties (for backward compatibility)
        self.video_path: Optional[Path] = None
        self.audio_path: Optional[Path] = None
        self.work_dir: Optional[Path] = None
        self.visual_output: Optional[VisualOutput] = None
        self.audio_output: Optional[AudioOutput] = None
        self.asr_output: Optional[ASROutput] = None
        self.yolo_output: Optional[YOLOOutput] = None
        self.consensus_output: Optional[ConsensusOutput] = None
        self.ai_output: Optional[AIDetectionOutput] = None
        self.report_path: Optional[str] = None
        self.pdf_path: Optional[str] = None
    
    def add_video(self, video_path: Path, work_dir: Path) -> int:
        """Add a new video for analysis, returns index"""
        analysis = VideoAnalysis(video_path=video_path, work_dir=work_dir)
        self.videos.append(analysis)
        return len(self.videos) - 1
    
    def get_current(self) -> Optional[VideoAnalysis]:
        """Get current video analysis"""
        if 0 <= self.current_index < len(self.videos):
            return self.videos[self.current_index]
        return None
    
    def sync_current_to_legacy(self):
        """Sync current video to legacy single-video properties"""
        current = self.get_current()
        if current:
            self.video_path = current.video_path
            self.audio_path = current.audio_path
            self.work_dir = current.work_dir
            self.visual_output = current.visual_output
            self.audio_output = current.audio_output
            self.asr_output = current.asr_output
            self.yolo_output = current.yolo_output
            self.ai_output = current.ai_output
    
    def sync_legacy_to_current(self):
        """Sync legacy properties back to current video"""
        current = self.get_current()
        if current:
            current.visual_output = self.visual_output
            current.audio_output = self.audio_output
            current.asr_output = self.asr_output
            current.yolo_output = self.yolo_output
            current.ai_output = self.ai_output
    
    def get_all_metrics(self) -> List[VideoMetrics]:
        """Get all video metrics for consensus"""
        return [v.to_metrics() for v in self.videos if v.visual_output or v.audio_output]
    
    def get_video_count(self) -> int:
        """Get number of videos"""
        return len(self.videos)
    
    def get_analyzed_count(self) -> int:
        """Get number of analyzed videos"""
        return sum(1 for v in self.videos if v.visual_output or v.audio_output)

STATE = AnalysisState()


# =============================================================================
# Utility Functions
# =============================================================================
def extract_audio_from_video(video_path: Path, output_dir: Path) -> Optional[Path]:
    output_path = output_dir / f"{video_path.stem}_audio.wav"
    if output_path.exists():
        return output_path
    try:
        sr = STATE.config.audio.sample_rate
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
               "-ar", str(sr), "-ac", "1", str(output_path)]
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
        timeout = STATE.config.report.pdf_timeout
        cmd = ["libreoffice", "--headless", "--convert-to", "pdf",
               "--outdir", str(Path(docx_path).parent), docx_path]
        subprocess.run(cmd, capture_output=True, check=True, timeout=timeout)
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
        for s in output.scene_categories[:5]
    ])
    
    # CCT interpretation
    cct = output.cct_mean
    if cct < 3500:
        cct_desc = "Warm (incandescent/sunset)"
    elif cct < 5500:
        cct_desc = "Neutral (daylight balanced)"
    else:
        cct_desc = "Cool (overcast/blue hour)"
    
    # Cut rate interpretation
    cut_rate = output.cuts / max(output.duration, 1) * 60  # cuts per minute
    if cut_rate > 30:
        pace_desc = "Very fast editing"
    elif cut_rate > 15:
        pace_desc = "Dynamic editing"
    elif cut_rate > 5:
        pace_desc = "Moderate pacing"
    else:
        pace_desc = "Slow, contemplative"
    
    return f"""## 📹 Visual Analysis Results

### 📊 Video Info

| Metric | Value | Description |
|:-------|:-----:|:------------|
| **Duration** | {output.duration:.2f}s | Total video length |
| **FPS** | {output.fps:.1f} | Frames per second |
| **Frames Analyzed** | {output.sampled_frames} | Sample size for analysis |

### 📷 Camera & Composition

| Metric | Value | Description |
|:-------|:------|:------------|
| **Camera Angle** | {output.camera_angle} | Viewer perspective (eye-level, overhead, low) |
| **Focal Length** | {output.focal_length_tendency} | Wide-angle, normal, or telephoto |

### 🎨 Color Analysis

| Metric | Value | Description |
|:-------|:------|:------------|
| **Dominant Hue** | {output.hue_family} | Primary color family |
| **Saturation** | {output.saturation_band} | Color intensity (vivid/muted) |
| **Brightness** | {output.brightness_band} | Light/dark overall |
| **Contrast** | {output.contrast} | Dynamic range |
| **Color Temp (CCT)** | {output.cct_mean:.0f}K | {cct_desc} |

### ✂️ Editing Pace

| Metric | Value | Description |
|:-------|:-----:|:------------|
| **Total Cuts** | {output.cuts} | Scene transitions detected |
| **Cuts/Minute** | {cut_rate:.1f} | {pace_desc} |

### 🏠 Scene Classification (CLIP)
*Top detected scene types:*
{scenes}

---
*Scene classification powered by CLIP (openai/clip-vit-large-patch14)*
"""


def format_audio(output: AudioOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    instruments = output.instruments.get('detected_instruments', [])
    inst_str = ", ".join(instruments[:5]) if instruments else "N/A"
    
    # Add explanations based on values
    tempo_desc = "Fast-paced" if output.tempo_bpm > 120 else "Medium tempo" if output.tempo_bpm > 80 else "Slow, relaxed"
    percussive_desc = "Heavy drums" if output.percussive_ratio > 0.5 else "Moderate beats" if output.percussive_ratio > 0.2 else "Light rhythm"
    
    return f"""## 🎵 Audio Analysis Results

### 💓 Rhythm & Tempo

| Metric | Value | Interpretation |
|:-------|:-----:|:---------------|
| **BPM** | {output.tempo_bpm:.1f} | {tempo_desc} |
| **Beat Count** | {output.num_beats} | Total rhythmic beats detected |
| **Percussive Ratio** | {output.percussive_ratio:.2f} | {percussive_desc} |

### 🎸 Music Classification (CLAP Model)

| Metric | Value | Description |
|:-------|:------|:------------|
| **BGM Style** | {output.bgm_style} | Genre/style of background music |
| **Mood** | {output.mood} | Emotional tone of the audio |
| **Key** | {output.key_signature} | Musical key (if detected) |

### 🎹 Instruments Detected
{inst_str}

---
*Analysis powered by CLAP (laion/larger_clap_music_and_speech)*
"""


def format_asr(output: ASROutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    text_preview = output.text[:500] + '...' if len(output.text) > 500 else output.text
    
    # WPM interpretation
    wpm = output.words_per_minute
    if wpm > 160:
        pace_desc = "Very fast (energetic/urgent)"
    elif wpm > 130:
        pace_desc = "Fast (conversational+)"
    elif wpm > 100:
        pace_desc = "Normal conversation"
    elif wpm > 60:
        pace_desc = "Slow (deliberate/clear)"
    else:
        pace_desc = "Very slow (emphatic)"
    
    # Emotion section
    emotion_section = ""
    if output.emotion:
        emo = output.emotion.get('dominant_emotion', 'N/A')
        conf = output.emotion.get('confidence', 0)
        emotion_section = f"""
### 🎭 Emotion Analysis (HuBERT Model)
| Detected Emotion | Confidence |
|:----------------:|:----------:|
| **{emo}** | {conf:.1%} |
"""
    
    # Prosody section
    prosody_section = ""
    if output.prosody:
        pitch = output.prosody.get('mean_pitch_hz', 0)
        style = output.prosody.get('prosody_style', 'N/A')
        intensity = output.prosody.get('mean_intensity', 0)
        prosody_section = f"""
### 📊 Prosody Analysis (Librosa)
| Metric | Value | Description |
|:-------|:-----:|:------------|
| **Pitch** | {pitch:.1f} Hz | Average fundamental frequency |
| **Style** | {style} | Speaking manner |
| **Intensity** | {intensity:.1f} dB | Volume level |
"""
    
    return f"""## 🎤 Speech Analysis Results

### 🗣️ Speech Rate

| Metric | Value | Interpretation |
|:-------|:-----:|:---------------|
| **Total Words** | {output.num_words} | Words transcribed |
| **Words/Min (WPM)** | {wpm:.1f} | {pace_desc} |
| **Pace Category** | {output.pace} | Overall speaking speed |

{emotion_section}
{prosody_section}

### 📝 Transcript (Whisper large-v3-turbo)
```
{text_preview}
```

---
*ASR powered by faster-whisper, Emotion by HuBERT*
"""


def format_yolo(output: YOLOOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    detection = output.detection
    environment = output.environment
    object_counts = detection.get('object_counts', {})
    
    objects_str = "\n".join([
        f"| {obj} | {cnt} |"
        for obj, cnt in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ])
    
    # Colors section
    colors_section = ""
    if output.colors:
        colors = output.colors
        if isinstance(colors, dict):
            dom_colors = colors.get('dominant_colors', colors.get('all_colors', []))
            if dom_colors and isinstance(dom_colors, list):
                colors_str = ", ".join(dom_colors[:5]) if dom_colors else "N/A"
                colors_section = f"""
### 🎨 Object Colors
**Dominant Colors**: {colors_str}
"""
    
    # Materials section
    materials_section = ""
    if output.materials:
        mats = output.materials
        if isinstance(mats, dict):
            dom_mats = mats.get('dominant_materials', mats.get('all_materials', []))
            if dom_mats and isinstance(dom_mats, list):
                mats_str = ", ".join(dom_mats[:5]) if dom_mats else "N/A"
                materials_section = f"""
### 🧱 Materials Detected
**Dominant Materials**: {mats_str}
"""
    
    return f"""## 🔍 Object Detection Results

### 🏠 Environment Classification

| Metric | Value | Description |
|:-------|:------|:------------|
| **Environment Type** | {environment.get('environment_type', 'N/A')} | Primary scene category |
| **Activity Style** | {environment.get('cooking_style', 'N/A')} | Detected activity type |

### 📦 Object Detection Statistics

| Metric | Value | Description |
|:-------|:-----:|:------------|
| **Unique Objects** | {detection.get('unique_objects', 0)} | Different object types found |
| **Total Detections** | {detection.get('total_detections', 0)} | Total instances across frames |

### 📋 Detected Objects (Top 10)

| Object | Count |
|:-------|:-----:|
{objects_str}

{colors_section}
{materials_section}

---
*Detection powered by YOLO11 (ultralytics)*
"""


def format_consensus(output: ConsensusOutput, video_count: int = 1) -> str:
    """Format consensus output - adapts based on single vs multi-video mode"""
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    cct_str = f"{output.cct:.0f}K" if output.cct else "N/A"
    bpm_str = f"{output.tempo_bpm:.1f}" if output.tempo_bpm else "N/A"
    cuts_str = f"{output.cuts_per_minute:.1f}" if output.cuts_per_minute else "N/A"
    shot_str = f"{output.avg_shot_length:.2f}s" if output.avg_shot_length else "N/A"
    
    # Determine mode
    if video_count <= 1:
        # Single video mode - Analysis Summary
        mode_title = t('single_video_mode')
        mode_icon = "📊"
        comparison_note = ""
    else:
        # Multi-video mode - Cross-Video Comparison  
        mode_title = t('multi_video_mode').format(n=video_count)
        mode_icon = "📈"
        comparison_note = f"\n> 📊 *Aggregated from {video_count} videos using majority voting (categorical) and median (numerical)*\n"
    
    # Build distribution details for categorical metrics
    def format_distribution(detail: Dict) -> str:
        if not detail or not detail.get('distribution'):
            return ""
        dist = detail.get('distribution', [])
        if len(dist) <= 1:
            return ""
        items = [f"`{d['value']}` ({d['percentage']:.0f}%)" for d in dist[:3]]
        return " | ".join(items)
    
    # Camera distribution
    camera_dist = format_distribution(getattr(output, 'camera_angle_detail', None))
    camera_row = f"| **Camera Angle** | {output.camera_angle} | {camera_dist if camera_dist else '—'} |"
    
    # Hue distribution
    hue_dist = format_distribution(getattr(output, 'hue_detail', None))
    hue_row = f"| **Hue Family** | {output.hue_family} | {hue_dist if hue_dist else '—'} |"
    
    # Saturation distribution
    sat_dist = format_distribution(getattr(output, 'saturation_detail', None))
    sat_row = f"| **Saturation** | {output.saturation} | {sat_dist if sat_dist else '—'} |"
    
    # Brightness distribution
    bright_dist = format_distribution(getattr(output, 'brightness_detail', None))
    bright_row = f"| **Brightness** | {output.brightness} | {bright_dist if bright_dist else '—'} |"
    
    # Scene distribution
    scene_dist = format_distribution(getattr(output, 'scene_category_detail', None))
    scene_row = f"| **Scene** | {output.scene_category} | {scene_dist if scene_dist else '—'} |"
    
    # BGM Style distribution
    bgm_dist = format_distribution(getattr(output, 'bgm_style_detail', None))
    bgm_row = f"| **BGM Style** | {output.bgm_style} | {bgm_dist if bgm_dist else '—'} |"
    
    # Mood distribution
    mood_dist = format_distribution(getattr(output, 'bgm_mood_detail', None))
    mood_row = f"| **Mood** | {output.bgm_mood} | {mood_dist if mood_dist else '—'} |"
    
    # YOLO section
    yolo_section = ""
    if getattr(output, 'yolo_available', False):
        yolo_env = getattr(output, 'yolo_environment', 'N/A')
        yolo_style = getattr(output, 'yolo_style', 'N/A')
        yolo_section = f"""
### 🔍 Object Detection Summary
| Metric | Value |
|:-------|:------|
| Environment | {yolo_env} |
| Activity | {yolo_style} |
"""
    
    # Beat alignment
    beat_section = ""
    if output.beat_alignment is not None:
        beat_pct = output.beat_alignment * 100
        beat_icon = "🎯" if beat_pct > 50 else "📍"
        beat_section = f"""
### 🎵 Audio-Visual Sync
| Metric | Value | Interpretation |
|:-------|:-----:|:---------------|
| **Beat Alignment** | {beat_pct:.1f}% | {beat_icon} {'Good' if beat_pct > 50 else 'Moderate'} sync between cuts & beats |
"""
    
    return f"""## {mode_icon} {mode_title}
{comparison_note}
### 🎬 Visual Characteristics

| Metric | Dominant Value | Distribution |
|:-------|:-------------:|:-------------|
{camera_row}
{hue_row}
{sat_row}
{bright_row}
{scene_row}

### 📊 Technical Metrics

| Metric | Value | Description |
|:-------|:-----:|:------------|
| **Color Temperature** | {cct_str} | Average CCT |
| **Cuts per Minute** | {cuts_str} | Editing pace |
| **Avg Shot Length** | {shot_str} | Shot duration |
| **BPM** | {bpm_str} | Music tempo |

### 🎵 Audio Characteristics

| Metric | Dominant Value | Distribution |
|:-------|:-------------:|:-------------|
{bgm_row}
{mood_row}
| **Key Signature** | {output.key_signature or 'N/A'} | — |
{yolo_section}{beat_section}
---
*Summary based on {'single video analysis' if video_count <= 1 else f'cross-video consensus from {video_count} videos'}*
"""


def format_ai_detection(output: AIDetectionOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    verdict_emoji = {
        "Real": "✅", "Suspicious": "⚠️", "Deepfake": "🎭",
        "AIGC": "🎨", "Audio-Deepfake": "🔊",
        "Synthetic": "🤖", "AI-Generated": "🤖", "Unknown": "❓"
    }
    emoji = verdict_emoji.get(output.verdict, "❓")
    
    models_str = ", ".join(output.models_used) if output.models_used else "None"
    
    # Get scores and availability flags with fallback
    aigc_score = getattr(output, 'aigc_score', 0.0)
    aigc_available = getattr(output, 'aigc_available', False)
    audio_score = getattr(output, 'audio_deepfake_score', 0.0)
    audio_available = getattr(output, 'audio_deepfake_available', False)
    temporal_available = getattr(output, 'temporal_available', False)
    face_available = getattr(output, 'face_available', False)
    
    # Get weights from analysis details
    weights = output.analysis_details.get("weights", {})
    deepfake_w = weights.get("deepfake", 0.30)
    clip_w = weights.get("clip", 0.20)
    temporal_w = weights.get("temporal", 0.15)
    aigc_w = weights.get("aigc", 0.20)
    audio_w = weights.get("audio_deepfake", 0.10)
    face_w = weights.get("face", 0.05)
    
    # Calculate weighted contribution
    def weighted_contrib(score, weight, available):
        if not available:
            return "—"
        contrib = score * weight
        return f"{contrib:.1%}"
    
    return f"""## 🤖 AI Detection Results

### {emoji} Verdict: **{output.verdict}**
### Confidence: **{output.confidence:.1%}** (weighted average)

---

### 📊 Detection Models & Weights

| Model | Weight | Score | Contribution | Status | Description |
|:------|:------:|:-----:|:------------:|:------:|:------------|
| 🎭 **DeepFake-v2** | `{deepfake_w:.0%}` | {output.deepfake_score:.1%} | {weighted_contrib(output.deepfake_score, deepfake_w, output.deepfake_available)} | {'✅' if output.deepfake_available else '❌'} | *HuggingFace ViT model (92% acc), detects face swaps* |
| 🔍 **CLIP Synthetic** | `{clip_w:.0%}` | {output.clip_synthetic_score:.1%} | {weighted_contrib(output.clip_synthetic_score, clip_w, output.clip_available)} | {'✅' if output.clip_available else '❌'} | *Zero-shot detection using CLIP embeddings* |
| ⏱️ **CLIP-Temporal** | `{temporal_w:.0%}` | {output.temporal_score:.1%} | {weighted_contrib(output.temporal_score, temporal_w, temporal_available)} | {'✅' if temporal_available else '❌'} | *Semantic consistency between frames (CLIP-based)* |
| 🎨 **AIGC Detector** | `{aigc_w:.0%}` | {aigc_score:.1%} | {weighted_contrib(aigc_score, aigc_w, aigc_available)} | {'✅' if aigc_available else '❌'} | *Detects Stable Diffusion, DALL-E, Midjourney* |
| 🔊 **Audio Deepfake** | `{audio_w:.0%}` | {audio_score:.1%} | {weighted_contrib(audio_score, audio_w, audio_available)} | {'✅' if audio_available else '❌'} | *Detects voice cloning & TTS synthesis* |
| 👤 **Face Analysis** | `{face_w:.0%}` | {output.no_face_ratio:.1%} | — | {'✅' if face_available else '❌'} | *No-face ratio analysis (>90% suspicious)* |

---

### 👤 Face Detection Details

| Metric | Value | Explanation |
|:-------|:-----:|:------------|
| **Faces Detected** | {output.faces_detected} | Total faces found across all frames |
| **Frames with Faces** | {output.frames_with_faces}/{output.frames_analyzed} | Ratio of frames containing faces |
| **No-Face Ratio** | {output.no_face_ratio:.1%} | Higher = more suspicious for face videos |
| **Temporal Anomalies** | {output.temporal_anomalies} | Sudden changes in frame consistency |

---

### ℹ️ How Scoring Works

- **Final Confidence** = Σ (Model Score × Weight) / Σ Weights
- Models with ⭐ use HuggingFace pretrained models (higher reliability)
- Models with ✅ use computed features (good reliability)
- **Verdict Thresholds**: Real <40% | Suspicious 40-70% | AI-Generated ≥70%

**Active Models**: {models_str}
"""


# =============================================================================
# Processing Functions
# =============================================================================
def upload_video(video_file):
    """Upload and add a video to the list (doesn't reset existing videos)"""
    if video_file is None:
        # Close button clicked - just clear preview, keep video list intact
        if STATE.videos:
            # Keep current state, just return current info without changes
            choices = get_video_list_choices()
            video = STATE.videos[STATE.current_index] if STATE.current_index < len(STATE.videos) else None
            if video and video.video_path:
                num_frames = STATE.config.ui.gallery_frames
                frame_paths = extract_frames_for_gallery(video.video_path, video.work_dir, num_frames)
                audio_str = str(video.audio_path) if video.audio_path else None
                status = f"📹 {t('video_n').format(n=STATE.current_index+1)}: {video.video_path.name}"
                return status, audio_str, frame_paths, get_video_list_header(), gr.update(choices=choices, value=STATE.current_index), gr.update(choices=choices, value=STATE.current_index)
        cur_val = STATE.current_index if STATE.videos else None
        choices = get_video_list_choices()
        return t('upload_first'), None, [], get_video_list_header(), gr.update(choices=choices, value=cur_val), gr.update(choices=choices, value=cur_val)
    
    import shutil
    video_path = Path(video_file)
    
    # Check if video already exists in list (by filename)
    for i, v in enumerate(STATE.videos):
        if v.video_path and v.video_path.name == video_path.name:
            # Video already exists, just switch to it
            STATE.current_index = i
            STATE.sync_current_to_legacy()
            num_frames = STATE.config.ui.gallery_frames
            frame_paths = extract_frames_for_gallery(v.video_path, v.work_dir, num_frames)
            audio_str = str(v.audio_path) if v.audio_path else None
            status = f"📹 {t('video_n').format(n=i+1)}: {v.video_path.name} ({t('already_added')})"
            choices = get_video_list_choices()
            return status, audio_str, frame_paths, get_video_list_header(), gr.update(choices=choices, value=i), gr.update(choices=choices, value=i)
    
    # Create main work directory if not exists
    if STATE.work_dir is None:
        STATE.work_dir = Path(tempfile.mkdtemp(prefix="video_analysis_"))
    
    # Create unique subdirectory for this video
    video_idx = len(STATE.videos)
    video_work_dir = STATE.work_dir / f"video_{video_idx}"
    video_work_dir.mkdir(exist_ok=True)
    
    # Copy video to work directory
    dest_path = video_work_dir / video_path.name
    shutil.copy(video_file, dest_path)
    
    # Add to video list
    idx = STATE.add_video(dest_path, video_work_dir)
    
    # Extract audio
    audio_path = extract_audio_from_video(dest_path, video_work_dir)
    STATE.videos[idx].audio_path = audio_path
    
    # Set as current video
    STATE.current_index = idx
    STATE.sync_current_to_legacy()
    
    # Extract frames for gallery
    num_frames = STATE.config.ui.gallery_frames
    frame_paths = extract_frames_for_gallery(dest_path, video_work_dir, num_frames)
    
    status = f"✅ {t('uploaded')}: {video_path.name}\n"
    status += f"📁 {t('video_n').format(n=idx+1)} of {len(STATE.videos)}\n"
    status += t('frames_extracted').format(n=len(frame_paths)) + "\n"
    status += t('audio_extracted') if audio_path else t('audio_failed')
    
    audio_str = str(audio_path) if audio_path else None
    
    # Get updated choices for radio
    choices = get_video_list_choices()
    
    return status, audio_str, frame_paths, get_video_list_header(), gr.update(choices=choices, value=idx), gr.update(choices=choices, value=idx)


def add_more_videos(video_files):
    """Add multiple videos to the list"""
    if video_files is None or len(video_files) == 0:
        choices = get_video_list_choices()
        cur_val = STATE.current_index if STATE.videos else None
        return get_video_list_header(), gr.update(choices=choices, value=cur_val), gr.update(choices=choices, value=cur_val)
    
    import shutil
    
    # Handle single file or list of files
    if not isinstance(video_files, list):
        video_files = [video_files]
    
    # Create main work directory if not exists
    if STATE.work_dir is None:
        STATE.work_dir = Path(tempfile.mkdtemp(prefix="video_analysis_"))
    
    for video_file in video_files:
        video_path = Path(video_file)
        
        # Check if video already exists
        exists = False
        for v in STATE.videos:
            if v.video_path and v.video_path.name == video_path.name:
                exists = True
                break
        if exists:
            continue
        
        # Create unique subdirectory for this video
        video_idx = len(STATE.videos)
        video_work_dir = STATE.work_dir / f"video_{video_idx}"
        video_work_dir.mkdir(exist_ok=True)
        
        # Copy video to work directory
        dest_path = video_work_dir / video_path.name
        shutil.copy(video_file, dest_path)
        
        # Add to video list
        idx = STATE.add_video(dest_path, video_work_dir)
        
        # Extract audio
        audio_path = extract_audio_from_video(dest_path, video_work_dir)
        STATE.videos[idx].audio_path = audio_path
    
    # Don't change current selection - user can switch manually
    choices = get_video_list_choices()
    cur_val = STATE.current_index if STATE.videos else None
    
    return get_video_list_header(), gr.update(choices=choices, value=cur_val), gr.update(choices=choices, value=cur_val)



def clear_all_videos():
    """Clear all videos"""
    STATE.reset()
    return get_video_list_header(), gr.update(choices=[], value=None), t('no_videos'), None, [], gr.update(choices=[], value=None)


def select_video_from_list(index):
    """Select a video from the radio list (index is passed directly)"""
    choices = get_video_list_choices()
    if index is None or not STATE.videos:
        return t('no_videos'), None, [], None, gr.update(choices=choices, value=None)
    
    # Convert to int if needed
    if isinstance(index, str):
        try:
            index = int(index)
        except:
            return t('no_videos'), None, [], None, gr.update(choices=choices, value=None)
    
    if 0 <= index < len(STATE.videos):
        STATE.current_index = index
        STATE.sync_current_to_legacy()
        
        video = STATE.videos[index]
        if video.video_path:
            # Extract frames for gallery
            num_frames = STATE.config.ui.gallery_frames
            frame_paths = extract_frames_for_gallery(video.video_path, video.work_dir, num_frames)
            audio_path = str(video.audio_path) if video.audio_path else None
            video_path_str = str(video.video_path)
            
            status = f"📹 {t('video_n').format(n=index+1)}: {video.video_path.name}"
            return status, audio_path, frame_paths, video_path_str, gr.update(choices=choices, value=index)
    
    return t('no_videos'), None, [], None, gr.update(choices=choices, value=None)


def load_video_results(index):
    """Load and display results for the selected video"""
    if index is None or not STATE.videos:
        no_result = f"*{t('upload_first')}*"
        return no_result, None, no_result, no_result, no_result, no_result
    
    # Convert to int if needed
    if isinstance(index, str):
        try:
            index = int(index)
        except:
            no_result = f"*{t('upload_first')}*"
            return no_result, None, no_result, no_result, no_result, no_result
    
    if 0 <= index < len(STATE.videos):
        STATE.current_index = index
        video = STATE.videos[index]
        
        # Format results for this video
        visual_result = format_visual(video.visual_output) if video.visual_output else f"*{t('run_analysis_first')}*"
        contact = video.visual_output.contact_sheet if video.visual_output else None
        audio_result = format_audio(video.audio_output) if video.audio_output else f"*{t('run_analysis_first')}*"
        asr_result = format_asr(video.asr_output) if video.asr_output else f"*{t('run_analysis_first')}*"
        yolo_result = format_yolo(video.yolo_output) if video.yolo_output else f"*{t('run_analysis_first')}*"
        ai_result = format_ai_detection(video.ai_output) if video.ai_output else f"*{t('run_analysis_first')}*"
        
        return visual_result, contact, audio_result, asr_result, yolo_result, ai_result
    
    no_result = f"*{t('upload_first')}*"
    return no_result, None, no_result, no_result, no_result, no_result


def delete_current_video():
    """Delete the currently selected video"""
    if not STATE.videos:
        return get_video_list_header(), gr.update(choices=[], value=None), t('no_videos'), None, [], gr.update(choices=[], value=None)
    
    idx = STATE.current_index
    if 0 <= idx < len(STATE.videos):
        STATE.videos.pop(idx)
        
        # Adjust current index
        if STATE.current_index >= len(STATE.videos):
            STATE.current_index = max(0, len(STATE.videos) - 1)
        
        # Sync to legacy state
        if STATE.videos:
            STATE.sync_current_to_legacy()
            video = STATE.videos[STATE.current_index]
            num_frames = STATE.config.ui.gallery_frames
            frame_paths = extract_frames_for_gallery(video.video_path, video.work_dir, num_frames) if video.video_path else []
            audio_path = str(video.audio_path) if video.audio_path else None
            status = f"📹 {t('video_n').format(n=STATE.current_index+1)}: {video.video_path.name if video.video_path else 'N/A'}"
            choices = get_video_list_choices()
            return get_video_list_header(), gr.update(choices=choices, value=STATE.current_index), status, audio_path, frame_paths, gr.update(choices=choices, value=STATE.current_index)
        else:
            STATE.video_path = None
            STATE.audio_path = None
            return get_video_list_header(), gr.update(choices=[], value=None), t('no_videos'), None, [], gr.update(choices=[], value=None)
    
    return get_video_list_header(), gr.update(choices=[], value=None), t('no_videos'), None, [], gr.update(choices=[], value=None)



def get_video_list_choices() -> List[Tuple[str, int]]:
    """Generate choices for video list radio buttons"""
    if not STATE.videos:
        return []
    
    choices = []
    for i, video in enumerate(STATE.videos):
        name = video.video_path.name if video.video_path else f"Video {i+1}"
        
        # Only show X marks for modules NOT analyzed
        missing = []
        if not video.visual_output:
            missing.append("✗V")
        if not video.audio_output:
            missing.append("✗A")
        if not video.asr_output:
            missing.append("✗S")
        if not video.yolo_output:
            missing.append("✗Y")
        if not video.ai_output:
            missing.append("✗AI")
        
        # Show missing modules or nothing if all done
        status_str = f" [{' '.join(missing)}]" if missing else ""
        label = f"📹 {i+1}. {name}{status_str}"
        choices.append((label, i))
    
    return choices


def get_video_list_header() -> str:
    """Get header text for video list"""
    if not STATE.videos:
        return f"### 📋 {t('video_list')}\n*{t('no_videos')}*"
    
    if len(STATE.videos) == 1:
        mode = t('single_video_mode')
    else:
        mode = t('multi_video_mode').format(n=len(STATE.videos))
    
    return f"### 📋 {t('video_list')} - {mode}\n*Click to switch videos*"


# Internal analysis functions (no progress tracking)
def _run_visual_internal():
    """Internal visual analysis without progress tracking"""
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}", None
    
    cfg = STATE.config.visual
    step = VisualAnalysisStep()
    input_data = VideoInput(
        video_path=STATE.video_path,
        work_dir=STATE.work_dir,
        frame_mode=cfg.frame_mode,
        target_frames=cfg.target_frames,
        scene_threshold=cfg.scene_threshold
    )
    
    STATE.visual_output = step.run(input_data)
    contact = STATE.visual_output.contact_sheet if STATE.visual_output else None
    return format_visual(STATE.visual_output), contact


def _run_audio_internal():
    """Internal audio analysis without progress tracking"""
    if STATE.audio_path is None:
        return f"❌ {t('upload_first')}"
    
    step = AudioAnalysisStep()
    input_data = AudioInput(audio_path=STATE.audio_path)
    STATE.audio_output = step.run(input_data)
    return format_audio(STATE.audio_output)


def _run_asr_internal(language: str):
    """Internal ASR analysis without progress tracking"""
    if STATE.audio_path is None:
        return f"❌ {t('upload_first')}"
    
    cfg = STATE.config.asr
    step = ASRAnalysisStep()
    input_data = ASRInput(
        audio_path=STATE.audio_path,
        language=language,
        model_size=cfg.whisper_model,
        beam_size=cfg.whisper_beam_size,
        enable_prosody=True,
        enable_emotion=True
    )
    STATE.asr_output = step.run(input_data)
    return format_asr(STATE.asr_output)


def _run_yolo_internal():
    """Internal YOLO analysis without progress tracking"""
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}"
    
    cfg = STATE.config.yolo
    step = YOLOAnalysisStep()
    input_data = YOLOInput(
        video_path=STATE.video_path,
        target_frames=cfg.target_frames,
        model_name=cfg.model_name,
        confidence_threshold=cfg.confidence_threshold,
        enable_colors=cfg.enable_colors,
        enable_materials=cfg.enable_materials
    )
    STATE.yolo_output = step.run(input_data)
    return format_yolo(STATE.yolo_output)


def _run_ai_detection_internal():
    """Internal AI detection without progress tracking"""
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}"
    
    cfg = STATE.config.ai_detection
    if not cfg.enabled:
        return "❌ AI Detection is disabled"
    
    step = AIDetectionStep()
    input_data = AIDetectionInput(
        video_path=STATE.video_path,
        audio_path=STATE.audio_path,
        use_deepfake=cfg.use_deepfake,
        use_clip=cfg.use_clip,
        use_temporal=cfg.use_temporal,
        use_face_detection=cfg.use_face_detection,
        use_aigc=cfg.use_aigc,
        use_audio_deepfake=cfg.use_audio_deepfake,
        num_frames=cfg.num_frames,
        temporal_frames=cfg.temporal_frames,
        fake_threshold=cfg.fake_threshold,
        no_face_threshold=cfg.no_face_threshold,
        deepfake_weight=cfg.deepfake_weight,
        clip_weight=cfg.clip_weight,
        temporal_weight=cfg.temporal_weight,
        aigc_weight=cfg.aigc_weight,
        audio_deepfake_weight=cfg.audio_deepfake_weight,
        face_weight=cfg.face_weight,
    )
    STATE.ai_output = step.run(input_data)
    return format_ai_detection(STATE.ai_output)


# Public analysis functions with progress tracking (for standalone button clicks)
def run_visual(progress=gr.Progress()):
    progress(0.1, desc="📹 Loading CLIP...")
    progress(0.3, desc="📹 Analyzing visual...")
    result = _run_visual_internal()
    STATE.sync_legacy_to_current()  # Sync to multi-video state
    progress(1.0, desc="✅ Visual done")
    return result


def run_audio(progress=gr.Progress()):
    progress(0.1, desc="🎵 Loading CLAP...")
    progress(0.3, desc="🎵 Analyzing audio...")
    result = _run_audio_internal()
    STATE.sync_legacy_to_current()  # Sync to multi-video state
    progress(1.0, desc="✅ Audio done")
    return result


def run_asr(language: str, progress=gr.Progress()):
    progress(0.1, desc="🎤 Loading Whisper...")
    progress(0.3, desc="🎤 Transcribing...")
    result = _run_asr_internal(language)
    STATE.sync_legacy_to_current()  # Sync to multi-video state
    progress(1.0, desc="✅ ASR done")
    return result


def run_yolo(progress=gr.Progress()):
    progress(0.1, desc="🔍 Loading YOLO...")
    progress(0.3, desc="🔍 Detecting objects...")
    result = _run_yolo_internal()
    STATE.sync_legacy_to_current()  # Sync to multi-video state
    progress(1.0, desc="✅ YOLO done")
    return result


def run_ai_detection(progress=gr.Progress()):
    progress(0.1, desc="🤖 Loading AI models...")
    progress(0.3, desc="🤖 Detecting AI content...")
    result = _run_ai_detection_internal()
    STATE.sync_legacy_to_current()  # Sync to multi-video state
    progress(1.0, desc="✅ AI detection done")
    return result


def run_batch_analysis(language: str, progress=gr.Progress()):
    """Analyze all videos in the list for cross-video comparison"""
    if not STATE.videos:
        empty_update = gr.update(choices=[], value=None)
        return (f"❌ {t('no_videos')}", None, "", "", "", "", "", "", get_video_list_header(), empty_update, empty_update)
    
    total_videos = len(STATE.videos)
    results = []
    
    for i, video in enumerate(STATE.videos):
        # Switch to this video
        STATE.current_index = i
        STATE.sync_current_to_legacy()
        
        # Progress for this video
        base_progress = i / total_videos
        progress(base_progress + 0.02, desc=f"📹 Video {i+1}/{total_videos}: Visual...")
        
        _run_visual_internal()
        progress(base_progress + 0.03, desc=f"🎵 Video {i+1}/{total_videos}: Audio...")
        
        _run_audio_internal()
        progress(base_progress + 0.05, desc=f"🎤 Video {i+1}/{total_videos}: ASR...")
        
        _run_asr_internal(language)
        progress(base_progress + 0.07, desc=f"🔍 Video {i+1}/{total_videos}: YOLO...")
        
        _run_yolo_internal()
        
        progress(base_progress + 0.09, desc=f"🤖 Video {i+1}/{total_videos}: AI Detection...")
        if STATE.config.ai_detection.enabled:
            _run_ai_detection_internal()
        
        # Sync back to video list
        STATE.sync_legacy_to_current()
        results.append(video.video_path.name if video.video_path else f"Video {i+1}")
    
    # Generate cross-video consensus
    progress(0.95, desc="📊 Generating cross-video summary...")
    consensus_result = run_consensus()
    
    progress(1.0, desc=t('done'))
    
    # Summary
    summary = f"✅ Analyzed {total_videos} videos:\n"
    for r in results:
        summary += f"  • {r}\n"
    summary += f"\n📊 Cross-video consensus generated"
    
    # Return the last video's results for display
    STATE.current_index = total_videos - 1
    STATE.sync_current_to_legacy()
    
    visual_result = format_visual(STATE.visual_output) if STATE.visual_output else ""
    audio_result = format_audio(STATE.audio_output) if STATE.audio_output else ""
    asr_result = format_asr(STATE.asr_output) if STATE.asr_output else ""
    yolo_result = format_yolo(STATE.yolo_output) if STATE.yolo_output else ""
    ai_result = format_ai_detection(STATE.ai_output) if STATE.ai_output else "*Not run*"
    contact = STATE.visual_output.contact_sheet if STATE.visual_output else None
    
    choices = get_video_list_choices()
    radio_update = gr.update(choices=choices, value=STATE.current_index)
    return (visual_result, contact, audio_result, asr_result, yolo_result, 
            ai_result, consensus_result, summary, get_video_list_header(), radio_update, radio_update)


def run_consensus():
    """Run consensus analysis - supports both single and multi-video mode"""
    # Sync current analysis to legacy state
    STATE.sync_legacy_to_current()
    
    # Get all metrics from all videos
    all_metrics = STATE.get_all_metrics()
    
    # Fallback to legacy single-video mode if no multi-video data
    if not all_metrics:
        if STATE.visual_output is None and STATE.audio_output is None:
            return f"❌ {t('run_analysis_first')}"
        
        # Use legacy single video
        metrics = VideoMetrics(path=str(STATE.video_path) if STATE.video_path else "")
        metrics.visual = STATE.visual_output
        metrics.audio = STATE.audio_output
        metrics.asr = STATE.asr_output
        metrics.yolo = STATE.yolo_output
        all_metrics = [metrics]
    
    step = ConsensusStep()
    input_data = ConsensusInput(video_metrics=all_metrics)
    STATE.consensus_output = step.run(input_data)
    
    video_count = len(all_metrics)
    return format_consensus(STATE.consensus_output, video_count)


def run_all(language: str, progress=gr.Progress()):
    # Use internal functions to avoid duplicate progress bars
    
    progress(0.05, desc="📹 Step 1/6: Visual Analysis...")
    visual_result, contact = _run_visual_internal()
    
    progress(0.20, desc="🎵 Step 2/6: Audio Analysis...")
    audio_result = _run_audio_internal()
    
    progress(0.35, desc="🎤 Step 3/6: Speech Recognition...")
    asr_result = _run_asr_internal(language)
    
    progress(0.50, desc="🔍 Step 4/6: Object Detection...")
    yolo_result = _run_yolo_internal()
    
    progress(0.65, desc="🤖 Step 5/6: AI Detection...")
    ai_result = _run_ai_detection_internal() if STATE.config.ai_detection.enabled else "*Disabled*"
    
    # Sync results back to multi-video state
    STATE.sync_legacy_to_current()
    
    progress(0.85, desc="📊 Step 6/6: Generating Summary...")
    consensus_result = run_consensus()
    
    progress(1.0, desc=t('done'))
    
    # Generate summary
    video_count = STATE.get_video_count()
    lines = ["=" * 25, t('quick_summary'), "=" * 25, ""]
    
    if video_count > 1:
        lines.append(f"📈 Mode: Cross-Video Comparison ({video_count} videos)")
        lines.append(f"✅ Analyzed: {STATE.get_analyzed_count()} videos")
    else:
        lines.append(f"📊 Mode: Single Video Analysis")
    
    lines.append("")
    
    if STATE.visual_output:
        lines.append(f"📹 Camera: {STATE.visual_output.camera_angle}")
        lines.append(f"🎨 Color: {STATE.visual_output.hue_family}")
    if STATE.audio_output:
        lines.append(f"🎵 BPM: {STATE.audio_output.tempo_bpm:.1f}")
        lines.append(f"🎸 BGM: {STATE.audio_output.bgm_style}")
    if STATE.asr_output:
        lines.append(f"🎤 WPM: {STATE.asr_output.words_per_minute:.1f}")
    if STATE.yolo_output:
        lines.append(f"🔍 Objects: {STATE.yolo_output.detection.get('unique_objects', 0)}")
    if STATE.ai_output:
        lines.append(f"🤖 AI: {STATE.ai_output.verdict} ({STATE.ai_output.confidence:.0%})")
    
    summary = "\n".join(lines)
    choices = get_video_list_choices()
    
    radio_update = gr.update(choices=choices, value=STATE.current_index)
    return visual_result, contact, audio_result, asr_result, yolo_result, ai_result, consensus_result, summary, get_video_list_header(), radio_update, radio_update


def gen_report(progress=gr.Progress()):
    if STATE.video_path is None:
        error_html = "<div style='text-align:center; padding:40px; background:#fee; border-radius:8px;'><p>❌ Please upload a video first</p></div>"
        return f"❌ {t('upload_first')}", None, None, error_html
    
    if STATE.visual_output is None and STATE.audio_output is None:
        error_html = "<div style='text-align:center; padding:40px; background:#fee; border-radius:8px;'><p>❌ Please run analysis first</p></div>"
        return f"❌ {t('run_analysis_first')}", None, None, error_html
    
    progress(0.2, desc="📄 Generating Word...")
    
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
        show_screenshots=STATE.config.report.include_screenshots
    )
    
    STATE.report_path = str(report_path)
    
    progress(0.7, desc="📕 Converting PDF...")
    STATE.pdf_path = convert_docx_to_pdf(STATE.report_path)
    
    progress(1.0, desc=t('done'))
    
    status = f"{t('report_generated')}\n📄 {report_path.name}"
    
    # Generate PDF preview HTML
    pdf_preview_html = "<div style='text-align:center; padding:40px; background:#f5f5f5; border-radius:8px;'><p>📄 PDF conversion not available (requires LibreOffice)</p></div>"
    if STATE.pdf_path:
        status += f"\n📕 {Path(STATE.pdf_path).name}"
        # Create embedded PDF viewer
        pdf_preview_html = f'''
        <div style="width:100%; height:500px; border:1px solid #ddd; border-radius:8px; overflow:hidden;">
            <iframe src="file://{STATE.pdf_path}" width="100%" height="100%" style="border:none;">
                <p>PDF preview not supported. <a href="file://{STATE.pdf_path}" download>Download PDF</a></p>
            </iframe>
        </div>
        <p style="text-align:center; margin-top:10px; color:#666;">
            ⬆️ If preview doesn't load, download the PDF file above
        </p>
        '''
    
    return status, STATE.report_path, STATE.pdf_path, pdf_preview_html


def export_json():
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}", None, "// Please upload a video first"
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "video_path": str(STATE.video_path),
        "config": STATE.config.to_dict(),
        "visual": STATE.visual_output.to_dict() if STATE.visual_output else None,
        "audio": STATE.audio_output.to_dict() if STATE.audio_output else None,
        "asr": STATE.asr_output.to_dict() if STATE.asr_output else None,
        "yolo": STATE.yolo_output.to_dict() if STATE.yolo_output else None,
        "ai_detection": STATE.ai_output.to_dict() if STATE.ai_output else None,
        "consensus": STATE.consensus_output.to_dict() if STATE.consensus_output else None,
    }
    
    indent = STATE.config.report.json_indent
    json_path = STATE.work_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    
    # Generate preview (truncated for large data)
    preview_data = {
        "timestamp": data["timestamp"],
        "video_path": data["video_path"],
        "visual": {"...": "see full JSON"} if data["visual"] else None,
        "audio": {"...": "see full JSON"} if data["audio"] else None,
        "asr": {"text_preview": data["asr"]["text"][:200] + "..." if data["asr"] and data["asr"].get("text") else None} if data["asr"] else None,
        "yolo": {"...": "see full JSON"} if data["yolo"] else None,
        "ai_detection": data["ai_detection"] if data["ai_detection"] else None,
        "consensus": {"...": "see full JSON"} if data["consensus"] else None,
    }
    json_preview = json.dumps(preview_data, indent=2, ensure_ascii=False, default=str)
    
    return f"{t('json_exported')}: {json_path.name}", str(json_path), json_preview


def update_config(
    visual_frames, visual_threshold,
    yolo_model, yolo_conf, yolo_frames,
    asr_model, asr_beam,
    ai_enabled, ai_deepfake, ai_clip, ai_temporal, ai_aigc, ai_audio, ai_face,
    ai_threshold, ai_deepfake_weight, ai_clip_weight, ai_temporal_weight, ai_aigc_weight, ai_audio_weight, ai_face_weight
):
    """Update configuration from UI controls"""
    # Visual
    STATE.config.visual.target_frames = int(visual_frames)
    STATE.config.visual.scene_threshold = float(visual_threshold)
    
    # YOLO
    STATE.config.yolo.model_name = yolo_model
    STATE.config.yolo.confidence_threshold = float(yolo_conf)
    STATE.config.yolo.target_frames = int(yolo_frames)
    
    # ASR
    STATE.config.asr.whisper_model = asr_model
    STATE.config.asr.whisper_beam_size = int(asr_beam)
    
    # AI Detection (SOTA 2025/2026)
    STATE.config.ai_detection.enabled = ai_enabled
    STATE.config.ai_detection.use_deepfake = ai_deepfake
    STATE.config.ai_detection.use_clip = ai_clip
    STATE.config.ai_detection.use_temporal = ai_temporal
    STATE.config.ai_detection.use_aigc = ai_aigc
    STATE.config.ai_detection.use_audio_deepfake = ai_audio
    STATE.config.ai_detection.use_face_detection = ai_face
    STATE.config.ai_detection.fake_threshold = float(ai_threshold)
    STATE.config.ai_detection.deepfake_weight = float(ai_deepfake_weight)
    STATE.config.ai_detection.clip_weight = float(ai_clip_weight)
    STATE.config.ai_detection.temporal_weight = float(ai_temporal_weight)
    STATE.config.ai_detection.aigc_weight = float(ai_aigc_weight)
    STATE.config.ai_detection.audio_deepfake_weight = float(ai_audio_weight)
    STATE.config.ai_detection.face_weight = float(ai_face_weight)
    
    # Calculate total weight and show warning if not ~1.0
    total_weight = ai_deepfake_weight + ai_clip_weight + ai_temporal_weight + ai_aigc_weight + ai_audio_weight + ai_face_weight
    
    status = "✅ Configuration updated"
    if abs(total_weight - 1.0) > 0.1:
        status += f"\n⚠️ Warning: Total weight = {total_weight:.2f} (recommended: ~1.0)"
    
    return status


def switch_language(lang: str):
    set_language(lang)
    # Note: Tab labels can't be dynamically updated in Gradio
    # Use --lang zh to start with Chinese interface
    footer_note = t('footer')
    if lang == "zh":
        footer_note += " | 🔄 刷新页面以更新所有标签"
    else:
        footer_note += " | 🔄 Refresh page for full language switch"
    
    return (
        f"# {t('title')}\n**{t('subtitle')}** | {t('models')}",
        t('analyze_all'),
        t('btn_visual'),
        t('btn_audio'),
        t('btn_asr'),
        t('btn_yolo'),
        t('btn_ai_detect'),
        t('btn_consensus'),
        t('gen_report'),
        t('export_json'),
        footer_note,
    )


# =============================================================================
# Gradio UI
# =============================================================================
def create_ui():
    cfg = STATE.config
    
    # Custom CSS for orange highlight on selected video
    custom_css = """
    .video-list-radio input[type="radio"]:checked + label {
        background-color: #ff9800 !important;
        border-color: #ff9800 !important;
        color: white !important;
    }
    .video-list-radio label {
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        border: 2px solid #e0e0e0;
        transition: all 0.2s;
    }
    .video-list-radio label:hover {
        border-color: #ff9800;
        background-color: #fff3e0;
    }
    """
    
    with gr.Blocks(title="Video Style Analysis", css=custom_css) as demo:
        
        # Header
        header_md = gr.Markdown(f"# {t('title')}\n**{t('subtitle')}** | {t('models')}")
        
        with gr.Row():
            lang_radio = gr.Radio(
                choices=[("English", "en"), ("中文", "zh")],
                value="en", label="Language / 语言", scale=1
            )
        
        gr.Markdown("---")
        
        with gr.Tabs():
            # ========== Tab 1: Upload & Preview ==========
            with gr.Tab(t('tab_upload'), id="tab_upload"):
                with gr.Row():
                    # Left: Upload Section
                    with gr.Column(scale=2, min_width=400):
                        gr.Markdown("### 📤 Video Upload")
                        gr.Markdown("*Supports MP4, AVI, MOV, MKV. Upload single or multiple videos.*")
                        
                        video_input = gr.Video(
                            label="📤 Upload Video",
                            height=280,
                            elem_classes=["video-preview"]
                        )
                        
                        upload_status = gr.Textbox(
                            label="Upload Status",
                            lines=2,
                            interactive=False
                        )
                        
                        # Video list for multi-video mode
                        video_list_header = gr.Markdown(get_video_list_header())
                        
                        video_list_radio = gr.Radio(
                            choices=get_video_list_choices(),
                            value=STATE.current_index if STATE.videos else None,
                            label=None,
                            interactive=True,
                            container=False,
                            elem_classes=["video-list-radio"]
                        )
                        
                        with gr.Row():
                            add_video_btn = gr.UploadButton(
                                "➕ Add More Videos", 
                                file_types=["video"],
                                file_count="multiple",
                                size="sm",
                                scale=2
                            )
                            delete_video_btn = gr.Button("🗑️ " + t('delete_video'), size="sm", variant="secondary", scale=1)
                            clear_videos_btn = gr.Button(t('clear_all'), size="sm", variant="stop", scale=1)
                        
                        gr.Markdown("### ⚙️ Analysis Settings")
                        language_select = gr.Dropdown(
                            choices=[("English", "en"), ("中文", "zh"), ("日本語", "ja"), ("한국어", "ko"), ("Auto-detect", "auto")],
                            value="en",
                            label="Speech Recognition Language"
                        )
                    
                    # Right: Preview Section
                    with gr.Column(scale=3, min_width=500):
                        gr.Markdown("### 🎬 Media Preview")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("**🔊 Extracted Audio**")
                                gr.Markdown("*Audio track separated from video*")
                                audio_player = gr.Audio(
                                    label="Audio Preview",
                                    type="filepath"
                                )
                        
                        gr.Markdown("### 🖼️ Key Frames Gallery")
                        gr.Markdown("*Click any frame to view full size*")
                        frame_gallery = gr.Gallery(
                            label="Extracted Key Frames",
                            columns=4,
                            rows=3,
                            height=280,
                            object_fit="contain",
                            allow_preview=True,
                            preview=True
                        )
            
            # ========== Tab 2: Run Analysis ==========
            with gr.Tab(t('tab_analysis'), id="tab_analysis"):
                gr.Markdown("### 🎯 Analysis Controls")
                gr.Markdown("*Click 'Analyze All' for complete analysis, or run individual modules*")
                
                with gr.Row():
                    run_all_btn = gr.Button(
                        "🎯 Analyze Current Video",
                        variant="primary",
                        size="lg",
                        scale=2
                    )
                    run_batch_btn = gr.Button(
                        "📈 Analyze All Videos (Cross-Video)",
                        variant="secondary",
                        size="lg",
                        scale=2
                    )
                
                gr.Markdown("*Use 'Analyze Current' for single video, 'Analyze All' for multi-video comparison*")
                gr.Markdown("**Individual Analysis Modules:**")
                with gr.Row():
                    run_visual_btn = gr.Button(t('btn_visual'), size="sm")
                    run_audio_btn = gr.Button(t('btn_audio'), size="sm")
                    run_asr_btn = gr.Button(t('btn_asr'), size="sm")
                    run_yolo_btn = gr.Button(t('btn_yolo'), size="sm")
                    run_ai_btn = gr.Button(t('btn_ai_detect'), size="sm")
                    run_consensus_btn = gr.Button(t('btn_consensus'), size="sm")
                
                gr.Markdown("---")
                
                # Video selector for viewing results
                gr.Markdown("### 📋 View Results For:")
                results_video_selector = gr.Radio(
                    choices=get_video_list_choices(),
                    value=STATE.current_index if STATE.videos else None,
                    label=None,
                    interactive=True,
                    container=False,
                    elem_classes=["video-list-radio"]
                )
                
                gr.Markdown("---")
                
                # Results Tabs with meaningful names
                with gr.Tabs():
                    with gr.Tab(t('tab_visual'), id="result_visual"):
                        visual_result = gr.Markdown(f"*{t('upload_first')}*")
                        contact_img = gr.Image(
                            label="Contact Sheet",
                            height=200
                        )
                    
                    with gr.Tab(t('tab_audio'), id="result_audio"):
                        audio_result = gr.Markdown(f"*{t('upload_first')}*")
                    
                    with gr.Tab(t('tab_asr'), id="result_asr"):
                        asr_result = gr.Markdown(f"*{t('upload_first')}*")
                    
                    with gr.Tab(t('tab_yolo'), id="result_yolo"):
                        yolo_result = gr.Markdown(f"*{t('upload_first')}*")
                    
                    with gr.Tab(t('tab_ai'), id="result_ai"):
                        ai_result = gr.Markdown(f"*{t('upload_first')}*")
                    
                    with gr.Tab(t('tab_summary'), id="result_summary"):
                        consensus_result = gr.Markdown(f"*{t('run_analysis_first')}*")
            
            # ========== Tab 3: Export & Reports ==========
            with gr.Tab(t('tab_export'), id="tab_export"):
                gr.Markdown("### 📄 Export Analysis Results")
                
                with gr.Row():
                    # Left: PDF Report
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📕 PDF Report")
                        gen_report_btn = gr.Button("📄 Generate Report", variant="primary", size="lg")
                        report_status = gr.Textbox(
                            label="Status",
                            lines=1,
                            interactive=False
                        )
                        
                        gr.Markdown("**Downloads:**")
                        with gr.Row():
                            report_file = gr.File(label="Word (.docx)")
                            pdf_file = gr.File(label="PDF")
                        
                        gr.Markdown("**PDF Preview:**")
                        pdf_preview = gr.HTML(
                            value="<div style='text-align:center; padding:40px; background:#f5f5f5; border-radius:8px;'><p>📄 Generate report to preview PDF</p></div>",
                            label="PDF Preview"
                        )
                    
                    # Right: JSON Export
                    with gr.Column(scale=1):
                        gr.Markdown("#### 💾 JSON Data")
                        export_json_btn = gr.Button("💾 Export JSON", variant="secondary", size="lg")
                        json_status = gr.Textbox(
                            label="Status",
                            lines=1,
                            interactive=False
                        )
                        
                        gr.Markdown("**Download:**")
                        json_file = gr.File(label="JSON Data")
                        
                        gr.Markdown("**JSON Preview:**")
                        json_preview = gr.Code(
                            value="// Generate JSON to preview data",
                            language="json",
                            label="JSON Preview",
                            lines=15
                        )
                
                gr.Markdown("---")
                gr.Markdown("### 📋 Quick Summary")
                summary_box = gr.Textbox(
                    label="Analysis Overview",
                    lines=10,
                    interactive=False
                )
            
            # ========== Tab 4: Configuration ==========
            with gr.Tab(t('tab_config'), id="tab_config"):
                gr.Markdown("### ⚙️ Analysis Configuration")
                gr.Markdown("*Adjust parameters for each analysis module. Changes apply to next analysis.*")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📹 Visual Analysis")
                        gr.Markdown("*More frames = slower but more accurate*")
                        cfg_visual_frames = gr.Slider(
                            10, 200, value=cfg.visual.target_frames, step=10,
                            label="Target Frames"
                        )
                        cfg_visual_threshold = gr.Slider(
                            10, 50, value=cfg.visual.scene_threshold, step=1,
                            label="Scene Threshold (lower = more sensitive)"
                        )
                        
                        gr.Markdown("#### 🔍 YOLO Object Detection")
                        gr.Markdown("*n=fastest, l=most accurate*")
                        cfg_yolo_model = gr.Dropdown(
                            choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"],
                            value=cfg.yolo.model_name,
                            label="YOLO Model"
                        )
                        cfg_yolo_conf = gr.Slider(
                            0.1, 0.9, value=cfg.yolo.confidence_threshold, step=0.05,
                            label="Confidence Threshold"
                        )
                        cfg_yolo_frames = gr.Slider(
                            10, 100, value=cfg.yolo.target_frames, step=5,
                            label="Frames to Analyze"
                        )
                        
                        gr.Markdown("#### 🎤 Speech Recognition (ASR)")
                        gr.Markdown("*large-v3-turbo = best, tiny = fastest*")
                        cfg_asr_model = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
                            value=cfg.asr.whisper_model,
                            label="Whisper Model"
                        )
                        cfg_asr_beam = gr.Slider(
                            1, 10, value=cfg.asr.whisper_beam_size, step=1,
                            label="Beam Size (higher = better)"
                        )
                    
                    with gr.Column():
                        gr.Markdown("#### 🤖 AI Detection Models")
                        gr.Markdown("*⭐ = HuggingFace pretrained (high reliability)*")
                        
                        cfg_ai_enabled = gr.Checkbox(
                            value=cfg.ai_detection.enabled,
                            label="Enable AI Detection"
                        )
                        
                        gr.Markdown("**Detection Models:**")
                        cfg_ai_deepfake = gr.Checkbox(
                            value=cfg.ai_detection.use_deepfake,
                            label="⭐ DeepFake-v2 (ViT, 92% acc)"
                        )
                        cfg_ai_clip = gr.Checkbox(
                            value=cfg.ai_detection.use_clip,
                            label="⭐ CLIP Zero-Shot"
                        )
                        cfg_ai_temporal = gr.Checkbox(
                            value=cfg.ai_detection.use_temporal,
                            label="✅ CLIP-Temporal Analysis"
                        )
                        cfg_ai_aigc = gr.Checkbox(
                            value=cfg.ai_detection.use_aigc,
                            label="⭐ AIGC Detector (SD/DALL-E/MJ)"
                        )
                        cfg_ai_audio = gr.Checkbox(
                            value=cfg.ai_detection.use_audio_deepfake,
                            label="⭐ Audio Deepfake"
                        )
                        cfg_ai_face = gr.Checkbox(
                            value=cfg.ai_detection.use_face_detection,
                            label="Face Analysis"
                        )
                        
                        cfg_ai_threshold = gr.Slider(
                            0.1, 0.9, value=cfg.ai_detection.fake_threshold, step=0.05,
                            label="AI Threshold (above = flagged)"
                        )
                        
                        gr.Markdown("#### ⚖️ Ensemble Weights")
                        gr.Markdown("*Higher weight = more influence on final score. Total should ≈ 1.0*")
                        
                        cfg_ai_deepfake_weight = gr.Slider(
                            0, 1, value=cfg.ai_detection.deepfake_weight, step=0.05,
                            label="DeepFake-v2 Weight (⭐ HuggingFace)"
                        )
                        cfg_ai_clip_weight = gr.Slider(
                            0, 1, value=cfg.ai_detection.clip_weight, step=0.05,
                            label="CLIP Synthetic Weight (⭐ HuggingFace)"
                        )
                        cfg_ai_temporal_weight = gr.Slider(
                            0, 1, value=cfg.ai_detection.temporal_weight, step=0.05,
                            label="CLIP-Temporal Weight (✅ Computed)"
                        )
                        cfg_ai_aigc_weight = gr.Slider(
                            0, 1, value=cfg.ai_detection.aigc_weight, step=0.05,
                            label="AIGC Detector Weight (⭐ HuggingFace)"
                        )
                        cfg_ai_audio_weight = gr.Slider(
                            0, 1, value=cfg.ai_detection.audio_deepfake_weight, step=0.05,
                            label="Audio Deepfake Weight (⭐ HuggingFace)"
                        )
                        cfg_ai_face_weight = gr.Slider(
                            0, 1, value=cfg.ai_detection.face_weight, step=0.05,
                            label="Face Analysis Weight"
                        )
                
                gr.Markdown("---")
                config_status = gr.Textbox(label="Status", interactive=False)
                save_config_btn = gr.Button("💾 Save Configuration", variant="primary", size="lg")
                
                save_config_btn.click(
                    fn=update_config,
                    inputs=[
                        cfg_visual_frames, cfg_visual_threshold,
                        cfg_yolo_model, cfg_yolo_conf, cfg_yolo_frames,
                        cfg_asr_model, cfg_asr_beam,
                        cfg_ai_enabled, cfg_ai_deepfake, cfg_ai_clip, cfg_ai_temporal, cfg_ai_aigc, cfg_ai_audio, cfg_ai_face,
                        cfg_ai_threshold, cfg_ai_deepfake_weight, cfg_ai_clip_weight, cfg_ai_temporal_weight, cfg_ai_aigc_weight, cfg_ai_audio_weight, cfg_ai_face_weight
                    ],
                    outputs=[config_status]
                )
        
        # Footer
        footer_md = gr.Markdown(f"---\n{t('footer')}")
        
        # ========== Event Handlers ==========
        video_input.change(fn=upload_video, inputs=[video_input],
                          outputs=[upload_status, audio_player, frame_gallery, video_list_header, video_list_radio, results_video_selector])
        
        add_video_btn.upload(fn=add_more_videos, inputs=[add_video_btn],
                            outputs=[video_list_header, video_list_radio, results_video_selector])
        
        video_list_radio.change(fn=select_video_from_list, inputs=[video_list_radio],
                               outputs=[upload_status, audio_player, frame_gallery, video_input, results_video_selector])
        
        # Sync video selection between Upload tab and Analysis tab
        results_video_selector.change(fn=load_video_results, inputs=[results_video_selector],
                                     outputs=[visual_result, contact_img, audio_result, asr_result, yolo_result, ai_result])
        
        delete_video_btn.click(fn=delete_current_video, inputs=[],
                              outputs=[video_list_header, video_list_radio, upload_status, audio_player, frame_gallery, results_video_selector])
        
        clear_videos_btn.click(fn=clear_all_videos, inputs=[],
                              outputs=[video_list_header, video_list_radio, upload_status, audio_player, frame_gallery, results_video_selector])
        
        run_visual_btn.click(fn=run_visual, outputs=[visual_result, contact_img])
        run_audio_btn.click(fn=run_audio, outputs=[audio_result])
        run_asr_btn.click(fn=run_asr, inputs=[language_select], outputs=[asr_result])
        run_yolo_btn.click(fn=run_yolo, outputs=[yolo_result])
        run_ai_btn.click(fn=run_ai_detection, outputs=[ai_result])
        run_consensus_btn.click(fn=run_consensus, outputs=[consensus_result])
        
        run_all_btn.click(
            fn=run_all,
            inputs=[language_select],
            outputs=[visual_result, contact_img, audio_result, asr_result,
                     yolo_result, ai_result, consensus_result, summary_box, video_list_header, video_list_radio, results_video_selector]
        )
        
        run_batch_btn.click(
            fn=run_batch_analysis,
            inputs=[language_select],
            outputs=[visual_result, contact_img, audio_result, asr_result,
                     yolo_result, ai_result, consensus_result, summary_box, video_list_header, video_list_radio, results_video_selector]
        )
        
        gen_report_btn.click(fn=gen_report, outputs=[report_status, report_file, pdf_file, pdf_preview])
        export_json_btn.click(fn=export_json, outputs=[json_status, json_file, json_preview])
        
        lang_radio.change(
            fn=switch_language,
            inputs=[lang_radio],
            outputs=[header_md, run_all_btn, run_visual_btn, run_audio_btn,
                     run_asr_btn, run_yolo_btn, run_ai_btn, run_consensus_btn,
                     gen_report_btn, export_json_btn, footer_md]
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
