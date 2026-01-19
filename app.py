#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Style Analysis - Gradio Web Interface
SOTA Models: CLIP | CLAP | HuBERT | Whisper | YOLO11 | GenConViT
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
        "models": "CLIP · CLAP · HuBERT · Whisper · YOLO11 · GenConViT",
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
        "btn_visual": "📹 Visual",
        "btn_audio": "🎵 Audio",
        "btn_asr": "🎤 ASR",
        "btn_yolo": "🔍 YOLO",
        "btn_consensus": "📊 Summary",
        "btn_ai_detect": "🤖 AI Detect",
        "gen_report": "📄 Report",
        "export_json": "💾 JSON",
        "tab_visual": "📹 Visual",
        "tab_audio": "🎵 Audio",
        "tab_asr": "🎤 ASR",
        "tab_yolo": "🔍 YOLO",
        "tab_summary": "📊 Summary",
        "tab_ai": "🤖 AI Detection",
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
    },
    "zh": {
        "title": "🎬 视频风格分析系统",
        "subtitle": "SOTA 2025/2026 | PyTorch + HuggingFace",
        "models": "CLIP · CLAP · HuBERT · Whisper · YOLO11 · GenConViT",
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
        "btn_visual": "📹 视觉",
        "btn_audio": "🎵 音频",
        "btn_asr": "🎤 语音",
        "btn_yolo": "🔍 检测",
        "btn_consensus": "📊 汇总",
        "btn_ai_detect": "🤖 AI检测",
        "gen_report": "📄 报告",
        "export_json": "💾 JSON",
        "tab_visual": "📹 视觉",
        "tab_audio": "🎵 音频",
        "tab_asr": "🎤 语音",
        "tab_yolo": "🔍 检测",
        "tab_summary": "📊 汇总",
        "tab_ai": "🤖 AI检测",
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
class AnalysisState:
    def __init__(self):
        self.reset()
        self.config = get_default_config()
    
    def reset(self):
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
    
    return f"""## 📹 {t('visual_results')}

| Duration | FPS | Frames |
|:---:|:---:|:---:|
| **{output.duration:.2f}s** | **{output.fps:.1f}** | **{output.sampled_frames}** |

| Angle | Focal | Hue | Saturation |
|:---:|:---:|:---:|:---:|
| **{output.camera_angle}** | **{output.focal_length_tendency}** | **{output.hue_family}** | **{output.saturation_band}** |

| Brightness | Contrast | CCT | Cuts |
|:---:|:---:|:---:|:---:|
| **{output.brightness_band}** | **{output.contrast}** | **{output.cct_mean:.0f}K** | **{output.cuts}** |

### 🏠 Scene (CLIP)
{scenes}
"""


def format_audio(output: AudioOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    instruments = output.instruments.get('detected_instruments', [])
    inst_str = ", ".join(instruments[:5]) if instruments else "N/A"
    
    return f"""## 🎵 {t('audio_results')}

| BPM | Beats | Percussive |
|:---:|:---:|:---:|
| **{output.tempo_bpm:.1f}** | **{output.num_beats}** | **{output.percussive_ratio:.2f}** |

| BGM Style | Mood | Key |
|:---:|:---:|:---:|
| **{output.bgm_style}** | **{output.mood}** | **{output.key_signature}** |

**Instruments**: {inst_str}
"""


def format_asr(output: ASROutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    text_preview = output.text[:400] + '...' if len(output.text) > 400 else output.text
    
    emotion_str = ""
    if output.emotion:
        emotion_str = f"\n**Emotion**: {output.emotion.get('dominant_emotion', 'N/A')} ({output.emotion.get('confidence', 0):.1%})"
    
    prosody_str = ""
    if output.prosody:
        prosody_str = f"\n**Prosody**: {output.prosody.get('mean_pitch_hz', 0):.1f}Hz, {output.prosody.get('prosody_style', 'N/A')}"
    
    return f"""## 🎤 {t('asr_results')}

| Words | WPM | Pace |
|:---:|:---:|:---:|
| **{output.num_words}** | **{output.words_per_minute:.1f}** | **{output.pace}** |
{prosody_str}{emotion_str}

### Transcript
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
    
    objects_str = "\n".join([
        f"| {obj} | {cnt} |"
        for obj, cnt in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    ])
    
    return f"""## 🔍 {t('yolo_results')}

| Environment | Style |
|:---:|:---:|
| **{environment.get('environment_type', 'N/A')}** | **{environment.get('cooking_style', 'N/A')}** |

| Unique Objects | Total Detections |
|:---:|:---:|
| **{detection.get('unique_objects', 0)}** | **{detection.get('total_detections', 0)}** |

| Object | Count |
|:---|:---:|
{objects_str}
"""


def format_consensus(output: ConsensusOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    cct_str = f"{output.cct:.0f}K" if output.cct else "N/A"
    bpm_str = f"{output.tempo_bpm:.1f}" if output.tempo_bpm else "N/A"
    
    return f"""## 📊 {t('summary_results')}

| Camera | Hue | Saturation | Brightness |
|:---:|:---:|:---:|:---:|
| **{output.camera_angle}** | **{output.hue_family}** | **{output.saturation}** | **{output.brightness}** |

| BGM | Mood | BPM | CCT |
|:---:|:---:|:---:|:---:|
| **{output.bgm_style}** | **{output.bgm_mood}** | **{bpm_str}** | **{cct_str}** |

**Scene**: {output.scene_category}
"""


def format_ai_detection(output: AIDetectionOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    verdict_emoji = {
        "Real": "✅", "Suspicious": "⚠️", "Deepfake": "🎭",
        "Synthetic": "🤖", "AI-Generated": "🤖", "Unknown": "❓"
    }
    emoji = verdict_emoji.get(output.verdict, "❓")
    
    models_str = ", ".join(output.models_used) if output.models_used else "None"
    
    return f"""## 🤖 {t('ai_results')}

### {emoji} {t('verdict')}: **{output.verdict}**
### {t('confidence')}: **{output.confidence:.1%}**

| Model | Score | Status |
|:---|:---:|:---:|
| GenConViT (Deepfake) | **{output.genconvit_score:.1%}** | {'✅' if output.genconvit_available else '❌'} |
| CLIP (Synthetic) | **{output.clip_synthetic_score:.1%}** | {'✅' if output.clip_available else '❌'} |
| Temporal (Motion) | **{output.temporal_score:.1%}** | ✅ |

| Face Analysis | Value |
|:---|:---:|
| Faces Detected | **{output.faces_detected}** |
| Frames with Faces | **{output.frames_with_faces}/{output.frames_analyzed}** |
| No-Face Ratio | **{output.no_face_ratio:.1%}** |
| Temporal Anomalies | **{output.temporal_anomalies}** |

**Models Used**: {models_str}
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
    num_frames = STATE.config.ui.gallery_frames
    frame_paths = extract_frames_for_gallery(STATE.video_path, STATE.work_dir, num_frames)
    
    status = f"{t('uploaded')}: {video_path.name}\n"
    status += f"{t('workdir')}: {STATE.work_dir}\n"
    status += t('frames_extracted').format(n=len(frame_paths)) + "\n"
    status += t('audio_extracted') if STATE.audio_path else t('audio_failed')
    
    audio_path = str(STATE.audio_path) if STATE.audio_path else None
    return status, audio_path, frame_paths


def run_visual(progress=gr.Progress()):
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}", None
    
    progress(0.1, desc=f"{t('loading')} CLIP...")
    cfg = STATE.config.visual
    step = VisualAnalysisStep()
    input_data = VideoInput(
        video_path=STATE.video_path,
        work_dir=STATE.work_dir,
        frame_mode=cfg.frame_mode
    )
    
    progress(0.4, desc=f"{t('analyzing')}...")
    STATE.visual_output = step.run(input_data)
    
    progress(1.0, desc=t('done'))
    contact = STATE.visual_output.contact_sheet if STATE.visual_output else None
    return format_visual(STATE.visual_output), contact


def run_audio(progress=gr.Progress()):
    if STATE.audio_path is None:
        return f"❌ {t('upload_first')}"
    
    progress(0.1, desc=f"{t('loading')} CLAP...")
    step = AudioAnalysisStep()
    input_data = AudioInput(audio_path=STATE.audio_path)
    
    progress(0.4, desc=f"{t('analyzing')}...")
    STATE.audio_output = step.run(input_data)
    
    progress(1.0, desc=t('done'))
    return format_audio(STATE.audio_output)


def run_asr(language: str, progress=gr.Progress()):
    if STATE.audio_path is None:
        return f"❌ {t('upload_first')}"
    
    progress(0.1, desc=f"{t('loading')} Whisper...")
    cfg = STATE.config.asr
    step = ASRAnalysisStep()
    input_data = ASRInput(
        audio_path=STATE.audio_path,
        language=language,
        model_size=cfg.whisper_model,
        enable_prosody=True,
        enable_emotion=True
    )
    
    progress(0.4, desc=f"{t('analyzing')}...")
    STATE.asr_output = step.run(input_data)
    
    progress(1.0, desc=t('done'))
    return format_asr(STATE.asr_output)


def run_yolo(progress=gr.Progress()):
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}"
    
    progress(0.1, desc=f"{t('loading')} YOLO11...")
    cfg = STATE.config.yolo
    step = YOLOAnalysisStep()
    input_data = YOLOInput(
        video_path=STATE.video_path,
        target_frames=cfg.target_frames,
        enable_colors=cfg.enable_colors,
        enable_materials=cfg.enable_materials
    )
    
    progress(0.4, desc=f"{t('analyzing')}...")
    STATE.yolo_output = step.run(input_data)
    
    progress(1.0, desc=t('done'))
    return format_yolo(STATE.yolo_output)


def run_ai_detection(progress=gr.Progress()):
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}"
    
    cfg = STATE.config.ai_detection
    if not cfg.enabled:
        return "❌ AI Detection is disabled in configuration"
    
    progress(0.1, desc=f"{t('loading')} GenConViT + CLIP...")
    step = AIDetectionStep()
    input_data = AIDetectionInput(
        video_path=STATE.video_path,
        use_genconvit=cfg.use_genconvit,
        use_clip=cfg.use_clip,
        use_temporal=cfg.use_temporal,
        use_face_detection=cfg.use_face_detection,
        num_frames=cfg.num_frames,
        temporal_frames=cfg.temporal_frames,
        fake_threshold=cfg.fake_threshold,
        no_face_threshold=cfg.no_face_threshold,
        genconvit_weight=cfg.genconvit_weight,
        clip_weight=cfg.clip_weight,
        temporal_weight=cfg.temporal_weight,
        face_weight=cfg.face_weight,
    )
    
    progress(0.4, desc=f"{t('analyzing')}...")
    STATE.ai_output = step.run(input_data)
    
    progress(1.0, desc=t('done'))
    return format_ai_detection(STATE.ai_output)


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
    progress(0.05, desc="📹 Visual...")
    visual_result, contact = run_visual()
    
    progress(0.20, desc="🎵 Audio...")
    audio_result = run_audio()
    
    progress(0.35, desc="🎤 ASR...")
    asr_result = run_asr(language)
    
    progress(0.50, desc="🔍 YOLO...")
    yolo_result = run_yolo()
    
    progress(0.65, desc="🤖 AI Detection...")
    ai_result = run_ai_detection() if STATE.config.ai_detection.enabled else "*Disabled*"
    
    progress(0.85, desc="📊 Summary...")
    consensus_result = run_consensus()
    
    progress(1.0, desc=t('done'))
    
    # Generate summary
    lines = ["=" * 25, t('quick_summary'), "=" * 25, ""]
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
    return visual_result, contact, audio_result, asr_result, yolo_result, ai_result, consensus_result, summary


def gen_report(progress=gr.Progress()):
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}", None, None
    
    if STATE.visual_output is None and STATE.audio_output is None:
        return f"❌ {t('run_analysis_first')}", None, None
    
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
    if STATE.pdf_path:
        status += f"\n📕 {Path(STATE.pdf_path).name}"
    
    return status, STATE.report_path, STATE.pdf_path


def export_json():
    if STATE.video_path is None:
        return f"❌ {t('upload_first')}", None
    
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
    
    return f"{t('json_exported')}: {json_path.name}", str(json_path)


def update_config(
    visual_frames, visual_threshold,
    yolo_model, yolo_conf, yolo_frames,
    asr_model, asr_beam,
    ai_enabled, ai_genconvit, ai_clip, ai_temporal, ai_face,
    ai_threshold, ai_genconvit_weight, ai_clip_weight, ai_temporal_weight, ai_face_weight
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
    STATE.config.ai_detection.use_genconvit = ai_genconvit
    STATE.config.ai_detection.use_clip = ai_clip
    STATE.config.ai_detection.use_temporal = ai_temporal
    STATE.config.ai_detection.use_face_detection = ai_face
    STATE.config.ai_detection.fake_threshold = float(ai_threshold)
    STATE.config.ai_detection.genconvit_weight = float(ai_genconvit_weight)
    STATE.config.ai_detection.clip_weight = float(ai_clip_weight)
    STATE.config.ai_detection.temporal_weight = float(ai_temporal_weight)
    STATE.config.ai_detection.face_weight = float(ai_face_weight)
    
    return "✅ Configuration updated"


def switch_language(lang: str):
    set_language(lang)
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
        t('footer'),
    )


# =============================================================================
# Gradio UI
# =============================================================================
def create_ui():
    cfg = STATE.config
    
    with gr.Blocks(title="Video Style Analysis") as demo:
        
        # Header
        header_md = gr.Markdown(f"# {t('title')}\n**{t('subtitle')}** | {t('models')}")
        
        with gr.Row():
            lang_radio = gr.Radio(
                choices=[("English", "en"), ("中文", "zh")],
                value="en", label="Language / 语言", scale=1
            )
        
        gr.Markdown("---")
        
        with gr.Tabs():
            # ========== Tab 1: Analysis ==========
            with gr.Tab("🎬 Analysis"):
                with gr.Row():
                    # Left: Upload
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown(f"### {t('upload_section')}")
                        video_input = gr.Video(label=t('select_video'), height=180)
                        upload_status = gr.Textbox(label=t('status'), lines=4, interactive=False)
                        
                        gr.Markdown(f"### {t('settings_section')}")
                        language_select = gr.Dropdown(
                            choices=[("English", "en"), ("中文", "zh"), ("日本語", "ja"), ("Auto", "auto")],
                            value="en", label=t('asr_language')
                        )
                        
                        gr.Markdown(f"### {t('preview_section')}")
                        audio_player = gr.Audio(label=t('audio_preview'), type="filepath")
                        frame_gallery = gr.Gallery(label=t('keyframes'), columns=3, height=120)
                    
                    # Middle: Results
                    with gr.Column(scale=2, min_width=500):
                        gr.Markdown(f"### {t('control_section')}")
                        
                        with gr.Row():
                            run_all_btn = gr.Button(t('analyze_all'), variant="primary", size="lg", scale=2)
                        
                        with gr.Row():
                            run_visual_btn = gr.Button(t('btn_visual'), size="sm")
                            run_audio_btn = gr.Button(t('btn_audio'), size="sm")
                            run_asr_btn = gr.Button(t('btn_asr'), size="sm")
                            run_yolo_btn = gr.Button(t('btn_yolo'), size="sm")
                            run_ai_btn = gr.Button(t('btn_ai_detect'), size="sm")
                            run_consensus_btn = gr.Button(t('btn_consensus'), size="sm")
                        
                        with gr.Tabs():
                            with gr.Tab(t('tab_visual')):
                                visual_result = gr.Markdown(f"*{t('upload_first')}*")
                                contact_img = gr.Image(label="Contact Sheet", height=120)
                            
                            with gr.Tab(t('tab_audio')):
                                audio_result = gr.Markdown(f"*{t('upload_first')}*")
                            
                            with gr.Tab(t('tab_asr')):
                                asr_result = gr.Markdown(f"*{t('upload_first')}*")
                            
                            with gr.Tab(t('tab_yolo')):
                                yolo_result = gr.Markdown(f"*{t('upload_first')}*")
                            
                            with gr.Tab(t('tab_ai')):
                                ai_result = gr.Markdown(f"*{t('upload_first')}*")
                            
                            with gr.Tab(t('tab_summary')):
                                consensus_result = gr.Markdown(f"*{t('run_analysis_first')}*")
                    
                    # Right: Export
                    with gr.Column(scale=1, min_width=250):
                        gr.Markdown(f"### {t('export_section')}")
                        
                        with gr.Row():
                            gen_report_btn = gr.Button(t('gen_report'), size="sm")
                            export_json_btn = gr.Button(t('export_json'), size="sm")
                        
                        report_status = gr.Textbox(label=t('report_status'), lines=2, interactive=False)
                        report_file = gr.File(label=t('word_report'))
                        pdf_file = gr.File(label=t('pdf_report'))
                        
                        json_status = gr.Textbox(label=t('json_status'), lines=1, interactive=False)
                        json_file = gr.File(label=t('json_data'))
                        
                        gr.Markdown("---")
                        summary_box = gr.Textbox(label=t('quick_summary'), lines=10, interactive=False)
            
            # ========== Tab 2: Configuration ==========
            with gr.Tab("⚙️ Configuration"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📹 Visual Analysis")
                        cfg_visual_frames = gr.Slider(10, 200, value=cfg.visual.target_frames, step=10, label=t('visual_frames'))
                        cfg_visual_threshold = gr.Slider(10, 50, value=cfg.visual.scene_threshold, step=1, label=t('visual_scene_threshold'))
                        
                        gr.Markdown("### 🔍 YOLO Detection")
                        cfg_yolo_model = gr.Dropdown(
                            choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"],
                            value=cfg.yolo.model_name, label=t('yolo_model')
                        )
                        cfg_yolo_conf = gr.Slider(0.1, 0.9, value=cfg.yolo.confidence_threshold, step=0.05, label=t('yolo_conf'))
                        cfg_yolo_frames = gr.Slider(10, 100, value=cfg.yolo.target_frames, step=5, label=t('yolo_frames'))
                        
                        gr.Markdown("### 🎤 ASR")
                        cfg_asr_model = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
                            value=cfg.asr.whisper_model, label=t('asr_model')
                        )
                        cfg_asr_beam = gr.Slider(1, 10, value=cfg.asr.whisper_beam_size, step=1, label=t('asr_beam_size'))
                    
                    with gr.Column():
                        gr.Markdown("### 🤖 AI Detection (SOTA 2025/2026)")
                        cfg_ai_enabled = gr.Checkbox(value=cfg.ai_detection.enabled, label=t('ai_enabled'))
                        
                        gr.Markdown("**Models**")
                        cfg_ai_genconvit = gr.Checkbox(value=cfg.ai_detection.use_genconvit, label="GenConViT (Deepfake)")
                        cfg_ai_clip = gr.Checkbox(value=cfg.ai_detection.use_clip, label="CLIP (Synthetic)")
                        cfg_ai_temporal = gr.Checkbox(value=cfg.ai_detection.use_temporal, label="Temporal (Motion)")
                        cfg_ai_face = gr.Checkbox(value=cfg.ai_detection.use_face_detection, label="Face Detection")
                        
                        cfg_ai_threshold = gr.Slider(0.1, 0.9, value=cfg.ai_detection.fake_threshold, step=0.05, label="Fake Threshold")
                        
                        gr.Markdown("### ⚖️ Ensemble Weights")
                        cfg_ai_genconvit_weight = gr.Slider(0, 1, value=cfg.ai_detection.genconvit_weight, step=0.1, label="GenConViT Weight")
                        cfg_ai_clip_weight = gr.Slider(0, 1, value=cfg.ai_detection.clip_weight, step=0.1, label="CLIP Weight")
                        cfg_ai_temporal_weight = gr.Slider(0, 1, value=cfg.ai_detection.temporal_weight, step=0.1, label="Temporal Weight")
                        cfg_ai_face_weight = gr.Slider(0, 1, value=cfg.ai_detection.face_weight, step=0.1, label="Face Weight")
                
                config_status = gr.Textbox(label="Status", interactive=False)
                save_config_btn = gr.Button("💾 Save Configuration", variant="primary")
                
                save_config_btn.click(
                    fn=update_config,
                    inputs=[
                        cfg_visual_frames, cfg_visual_threshold,
                        cfg_yolo_model, cfg_yolo_conf, cfg_yolo_frames,
                        cfg_asr_model, cfg_asr_beam,
                        cfg_ai_enabled, cfg_ai_genconvit, cfg_ai_clip, cfg_ai_temporal, cfg_ai_face,
                        cfg_ai_threshold, cfg_ai_genconvit_weight, cfg_ai_clip_weight, cfg_ai_temporal_weight, cfg_ai_face_weight
                    ],
                    outputs=[config_status]
                )
        
        # Footer
        footer_md = gr.Markdown(f"---\n{t('footer')}")
        
        # ========== Event Handlers ==========
        video_input.change(fn=upload_video, inputs=[video_input],
                          outputs=[upload_status, audio_player, frame_gallery])
        
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
                     yolo_result, ai_result, consensus_result, summary_box]
        )
        
        gen_report_btn.click(fn=gen_report, outputs=[report_status, report_file, pdf_file])
        export_json_btn.click(fn=export_json, outputs=[json_status, json_file])
        
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
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    )
