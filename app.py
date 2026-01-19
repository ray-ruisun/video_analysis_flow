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
        "tech_stack": "**技术栈**: CLIP (场景) | CLAP (音频) | HuBERT (情感) | Whisper (ASR) | YOLO11 (检测)",
        "upload_video": "📤 上传视频",
        "upload_label": "上传视频文件 (mp4, avi, mov, mkv)",
        "upload_status": "上传状态",
        "audio_preview": "🎵 音频预览",
        "audio_label": "提取的音频 (自动从视频分离)",
        "settings": "⚙️ 设置",
        "asr_lang": "ASR 语言",
        "ui_lang": "界面语言",
        "frame_preview": "🖼️ 关键帧预览",
        "frame_label": "视频关键帧 (均匀采样)",
        "analysis_ctrl": "🚀 分析控制",
        "run_all": "🎯 一键分析",
        "gen_report": "📄 生成报告",
        "export_json": "💾 导出 JSON",
        "step_exec": "🔧 分步执行",
        "visual_btn": "📹 视觉",
        "audio_btn": "🎵 音频",
        "asr_btn": "🎤 语音",
        "yolo_btn": "🔍 检测",
        "consensus_btn": "🎯 共识",
        "progress": "当前进度",
        "waiting": "等待开始...",
        "tab_visual": "📹 视觉分析",
        "tab_audio": "🎵 音频分析",
        "tab_asr": "🎤 语音分析",
        "tab_yolo": "🔍 目标检测",
        "tab_consensus": "🎯 共识分析",
        "report_gen": "📊 报告生成",
        "report_status": "报告状态",
        "download": "📥 下载",
        "word_report": "Word 报告 (.docx)",
        "pdf_report": "PDF 报告 (.pdf)",
        "json_status": "JSON 导出状态",
        "json_data": "JSON 数据",
        "summary": "📋 分析摘要",
        "summary_label": "快速预览",
        "summary_placeholder": "分析完成后显示摘要...",
        "hint_upload": "*上传视频后开始分析*",
        "hint_consensus": "*运行分析后自动生成*",
        "uploading": "正在处理视频...",
        "uploaded": "✅ 视频已上传",
        "work_dir": "📁 工作目录",
        "frames_extracted": "🖼️ 提取了 {n} 个关键帧",
        "audio_extracted": "🎵 音频已提取",
        "audio_failed": "⚠️ 音频提取失败",
        "please_upload": "❌ 请先上传视频",
        "analysis_failed": "❌ 分析失败",
        "loading_model": "⏳ 加载模型中...",
        "analyzing": "🔄 分析中...",
        "done": "✅ 完成!",
        "visual_init": "⏳ 加载 CLIP 模型...",
        "visual_analyzing": "🔄 视觉分析中 (镜头/色彩/剪辑)...",
        "audio_init": "⏳ 加载 CLAP 模型...",
        "audio_analyzing": "🔄 音频分析中 (节奏/BGM/情绪)...",
        "asr_init": "⏳ 加载 Whisper 模型...",
        "asr_analyzing": "🔄 语音识别中 (转录/韵律/情感)...",
        "yolo_init": "⏳ 加载 YOLO11 模型...",
        "yolo_analyzing": "🔄 目标检测中...",
        "consensus_calc": "🔄 计算共识...",
        "report_gen_word": "🔄 生成 Word 报告...",
        "report_gen_pdf": "🔄 转换为 PDF...",
        "report_done": "✅ 报告已生成",
        "pdf_failed": "⚠️ PDF 转换失败 (需要 libreoffice)",
        "json_exported": "✅ JSON 已导出",
        "summary_title": "=== 分析摘要 ===",
        "cam_angle": "📹 镜头角度",
        "hue": "🎨 色调",
        "cuts": "✂️ 剪辑数",
        "bpm": "🎵 BPM",
        "bgm_style": "🎸 BGM 风格",
        "speech_rate": "🎤 语速",
        "objects": "🔍 检测物体",
        "footer": "**Video Style Analysis** | SOTA 2025/2026",
        # 结果格式化
        "visual_result": "# 📹 视觉分析结果",
        "duration": "时长",
        "sampled_frames": "采样帧数",
        "cam_analysis": "## 📷 镜头分析",
        "focal_tendency": "焦距倾向",
        "color_analysis": "## 🎨 色彩分析",
        "saturation": "饱和度",
        "brightness": "亮度",
        "contrast": "对比度",
        "cct": "色温",
        "edit_analysis": "## ✂️ 剪辑分析",
        "avg_shot_len": "平均镜头时长",
        "transition": "转场类型",
        "scene_class": "## 🏠 场景分类 (CLIP)",
        "audio_result": "# 🎵 音频分析结果 (CLAP)",
        "rhythm_analysis": "## 节奏分析",
        "beats": "节拍数",
        "perc_ratio": "打击乐比例",
        "bgm_style_section": "## 🎸 BGM 风格",
        "main_style": "主要风格",
        "top3_style": "Top 3 风格",
        "mood_section": "## 😊 情绪分析",
        "main_mood": "主要情绪",
        "top3_mood": "Top 3 情绪",
        "other_section": "## 🎹 其他",
        "key_sig": "调式",
        "speech_ratio": "语音比例",
        "instruments": "检测到的乐器",
        "asr_result": "# 🎤 语音分析结果 (Whisper + HuBERT)",
        "trans_stats": "## 📝 转录统计",
        "word_count": "词数",
        "pace": "节奏",
        "pauses": "停顿数",
        "pause_style": "停顿风格",
        "catchphrases": "## 🔁 口头禅 (高频短语)",
        "prosody_section": "## 🎼 韵律分析",
        "mean_pitch": "平均音高",
        "pitch_var": "音高变化",
        "tone": "音调",
        "prosody_style": "韵律风格",
        "emotion_section": "## 😊 语音情感 (HuBERT)",
        "main_emotion": "主要情感",
        "emotion_dist": "情感分布",
        "transcript": "## 📜 转录文本",
        "yolo_result": "# 🔍 目标检测结果 (YOLO11)",
        "env_analysis": "## 🏠 环境分析",
        "env_type": "环境类型",
        "cook_style": "烹饪风格",
        "appliance_tier": "设备档次",
        "det_stats": "## 📊 检测统计",
        "unique_obj": "检测物体类数",
        "total_det": "总检测次数",
        "frames_proc": "处理帧数",
        "detected_obj": "## 🎯 检测到的物体",
        "confidence": "置信度",
        "obj_colors": "## 🎨 物体颜色",
        "obj_materials": "## 🧱 物体材质",
        "consensus_result": "# 🎯 跨视频共识分析",
        "cam_consensus": "## 📷 镜头共识",
        "cam_motion": "相机运动",
        "color_consensus": "## 🎨 色彩共识",
        "edit_consensus": "## ✂️ 剪辑共识",
        "cuts_per_min": "剪辑/分钟",
        "audio_consensus": "## 🎵 音频共识",
        "beat_align": "节拍对齐",
        "scene_consensus": "## 🏠 场景共识",
        "scene_type": "场景类型",
        "yolo_consensus": "## 🔍 YOLO 共识",
        "env": "环境",
        "style": "风格",
    },
    "en": {
        "title": "🎬 Video Style Analysis",
        "subtitle": "SOTA 2025/2026 | PyTorch + HuggingFace",
        "tech_stack": "**Tech Stack**: CLIP (Scene) | CLAP (Audio) | HuBERT (Emotion) | Whisper (ASR) | YOLO11 (Detection)",
        "upload_video": "📤 Upload Video",
        "upload_label": "Upload video file (mp4, avi, mov, mkv)",
        "upload_status": "Upload Status",
        "audio_preview": "🎵 Audio Preview",
        "audio_label": "Extracted audio (auto-separated)",
        "settings": "⚙️ Settings",
        "asr_lang": "ASR Language",
        "ui_lang": "UI Language",
        "frame_preview": "🖼️ Frame Preview",
        "frame_label": "Key frames (uniform sampling)",
        "analysis_ctrl": "🚀 Analysis Control",
        "run_all": "🎯 Run All",
        "gen_report": "📄 Generate Report",
        "export_json": "💾 Export JSON",
        "step_exec": "🔧 Step by Step",
        "visual_btn": "📹 Visual",
        "audio_btn": "🎵 Audio",
        "asr_btn": "🎤 ASR",
        "yolo_btn": "🔍 YOLO",
        "consensus_btn": "🎯 Consensus",
        "progress": "Progress",
        "waiting": "Waiting to start...",
        "tab_visual": "📹 Visual",
        "tab_audio": "🎵 Audio",
        "tab_asr": "🎤 ASR",
        "tab_yolo": "🔍 YOLO",
        "tab_consensus": "🎯 Consensus",
        "report_gen": "📊 Report",
        "report_status": "Report Status",
        "download": "📥 Download",
        "word_report": "Word Report (.docx)",
        "pdf_report": "PDF Report (.pdf)",
        "json_status": "JSON Export Status",
        "json_data": "JSON Data",
        "summary": "📋 Summary",
        "summary_label": "Quick Preview",
        "summary_placeholder": "Summary will appear after analysis...",
        "hint_upload": "*Upload video to start*",
        "hint_consensus": "*Generated after analysis*",
        "uploading": "Processing video...",
        "uploaded": "✅ Video uploaded",
        "work_dir": "📁 Work directory",
        "frames_extracted": "🖼️ Extracted {n} key frames",
        "audio_extracted": "🎵 Audio extracted",
        "audio_failed": "⚠️ Audio extraction failed",
        "please_upload": "❌ Please upload a video first",
        "analysis_failed": "❌ Analysis failed",
        "loading_model": "⏳ Loading model...",
        "analyzing": "🔄 Analyzing...",
        "done": "✅ Done!",
        "visual_init": "⏳ Loading CLIP model...",
        "visual_analyzing": "🔄 Visual analysis (camera/color/editing)...",
        "audio_init": "⏳ Loading CLAP model...",
        "audio_analyzing": "🔄 Audio analysis (rhythm/BGM/mood)...",
        "asr_init": "⏳ Loading Whisper model...",
        "asr_analyzing": "🔄 Speech recognition (transcript/prosody/emotion)...",
        "yolo_init": "⏳ Loading YOLO11 model...",
        "yolo_analyzing": "🔄 Object detection...",
        "consensus_calc": "🔄 Calculating consensus...",
        "report_gen_word": "🔄 Generating Word report...",
        "report_gen_pdf": "🔄 Converting to PDF...",
        "report_done": "✅ Report generated",
        "pdf_failed": "⚠️ PDF conversion failed (requires libreoffice)",
        "json_exported": "✅ JSON exported",
        "summary_title": "=== Analysis Summary ===",
        "cam_angle": "📹 Camera angle",
        "hue": "🎨 Hue",
        "cuts": "✂️ Cuts",
        "bpm": "🎵 BPM",
        "bgm_style": "🎸 BGM style",
        "speech_rate": "🎤 Speech rate",
        "objects": "🔍 Objects",
        "footer": "**Video Style Analysis** | SOTA 2025/2026",
        "visual_result": "# 📹 Visual Analysis Results",
        "duration": "Duration",
        "sampled_frames": "Sampled frames",
        "cam_analysis": "## 📷 Camera Analysis",
        "focal_tendency": "Focal tendency",
        "color_analysis": "## 🎨 Color Analysis",
        "saturation": "Saturation",
        "brightness": "Brightness",
        "contrast": "Contrast",
        "cct": "CCT",
        "edit_analysis": "## ✂️ Editing Analysis",
        "avg_shot_len": "Avg shot length",
        "transition": "Transition type",
        "scene_class": "## 🏠 Scene Classification (CLIP)",
        "audio_result": "# 🎵 Audio Analysis Results (CLAP)",
        "rhythm_analysis": "## Rhythm Analysis",
        "beats": "Beats",
        "perc_ratio": "Percussive ratio",
        "bgm_style_section": "## 🎸 BGM Style",
        "main_style": "Main style",
        "top3_style": "Top 3 styles",
        "mood_section": "## 😊 Mood Analysis",
        "main_mood": "Main mood",
        "top3_mood": "Top 3 moods",
        "other_section": "## 🎹 Other",
        "key_sig": "Key signature",
        "speech_ratio": "Speech ratio",
        "instruments": "Detected instruments",
        "asr_result": "# 🎤 Speech Analysis Results (Whisper + HuBERT)",
        "trans_stats": "## 📝 Transcription Stats",
        "word_count": "Word count",
        "pace": "Pace",
        "pauses": "Pauses",
        "pause_style": "Pause style",
        "catchphrases": "## 🔁 Catchphrases",
        "prosody_section": "## 🎼 Prosody Analysis",
        "mean_pitch": "Mean pitch",
        "pitch_var": "Pitch variation",
        "tone": "Tone",
        "prosody_style": "Prosody style",
        "emotion_section": "## 😊 Speech Emotion (HuBERT)",
        "main_emotion": "Main emotion",
        "emotion_dist": "Emotion distribution",
        "transcript": "## 📜 Transcript",
        "yolo_result": "# 🔍 Object Detection Results (YOLO11)",
        "env_analysis": "## 🏠 Environment Analysis",
        "env_type": "Environment type",
        "cook_style": "Cooking style",
        "appliance_tier": "Appliance tier",
        "det_stats": "## 📊 Detection Stats",
        "unique_obj": "Unique objects",
        "total_det": "Total detections",
        "frames_proc": "Frames processed",
        "detected_obj": "## 🎯 Detected Objects",
        "confidence": "confidence",
        "obj_colors": "## 🎨 Object Colors",
        "obj_materials": "## 🧱 Object Materials",
        "consensus_result": "# 🎯 Cross-Video Consensus",
        "cam_consensus": "## 📷 Camera Consensus",
        "cam_motion": "Camera motion",
        "color_consensus": "## 🎨 Color Consensus",
        "edit_consensus": "## ✂️ Editing Consensus",
        "cuts_per_min": "Cuts/min",
        "audio_consensus": "## 🎵 Audio Consensus",
        "beat_align": "Beat alignment",
        "scene_consensus": "## 🏠 Scene Consensus",
        "scene_type": "Scene type",
        "yolo_consensus": "## 🔍 YOLO Consensus",
        "env": "Environment",
        "style": "Style",
    }
}

CURRENT_LANG = "zh"

def t(key: str) -> str:
    """获取翻译文本"""
    return I18N.get(CURRENT_LANG, I18N["zh"]).get(key, key)

def set_lang(lang: str):
    """设置语言"""
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
    try:
        from docx2pdf import convert
        convert(docx_path, pdf_path)
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
        t('cam_analysis'),
        f"**{t('cam_angle')}**: {output.camera_angle}",
        format_distribution(output.camera_angle_detail),
        f"\n**{t('focal_tendency')}**: {output.focal_length_tendency}",
        f"\n{t('color_analysis')}",
        f"**{t('hue')}**: {output.hue_family}",
        format_distribution(output.hue_detail),
        f"\n**{t('saturation')}**: {output.saturation_band}",
        f"\n**{t('brightness')}**: {output.brightness_band}",
        f"\n**{t('contrast')}**: {output.contrast}",
    ]
    if output.cct_mean:
        lines.append(f"\n**{t('cct')}**: {output.cct_mean:.0f}K")
    
    lines.extend([
        f"\n{t('edit_analysis')}",
        f"**{t('cuts')}**: {output.cuts}",
        f"**{t('avg_shot_len')}**: {output.avg_shot_length:.2f}s",
        f"**{t('transition')}**: {output.transition_type}",
        f"\n{t('scene_class')}",
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
        t('rhythm_analysis'),
        f"**BPM**: {output.tempo_bpm:.1f}",
        f"**{t('beats')}**: {output.num_beats}",
        f"**{t('perc_ratio')}**: {output.percussive_ratio:.2f}",
        f"\n{t('bgm_style_section')}",
        f"**{t('main_style')}**: {output.bgm_style} ({output.bgm_style_confidence:.1%})",
    ]
    
    if output.bgm_style_detail and output.bgm_style_detail.get('top_3'):
        lines.append(f"**{t('top3_style')}:**")
        for item in output.bgm_style_detail['top_3'][:3]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(f"  • {item[0]}: {item[1]:.1%}")
    
    lines.extend([
        f"\n{t('mood_section')}",
        f"**{t('main_mood')}**: {output.mood} ({output.mood_confidence:.1%})",
    ])
    
    if output.mood_detail and output.mood_detail.get('top_3'):
        lines.append(f"**{t('top3_mood')}:**")
        for item in output.mood_detail['top_3'][:3]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(f"  • {item[0]}: {item[1]:.1%}")
    
    lines.extend([
        f"\n{t('other_section')}",
        f"**{t('key_sig')}**: {output.key_signature}",
        f"**{t('speech_ratio')}**: {output.speech_ratio:.2f}",
    ])
    
    instruments = output.instruments.get('detected_instruments', [])
    if instruments:
        lines.append(f"**{t('instruments')}**: {', '.join(instruments)}")
    
    return "\n".join(lines)


def format_asr_output(output: ASROutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    lines = [
        f"{t('asr_result')}\n",
        t('trans_stats'),
        f"**{t('word_count')}**: {output.num_words}",
        f"**{t('speech_rate')}**: {output.words_per_second:.2f} w/s ({output.words_per_minute:.1f} wpm)",
        f"**{t('pace')}**: {output.pace}",
        f"**{t('pauses')}**: {output.num_pauses}",
        f"**{t('pause_style')}**: {output.pause_style}",
    ]
    
    if output.catchphrases:
        lines.append(f"\n{t('catchphrases')}")
        for phrase in output.catchphrases[:10]:
            lines.append(f'  • "{phrase}"')
    
    if output.prosody:
        lines.extend([
            f"\n{t('prosody_section')}",
            f"**{t('mean_pitch')}**: {output.prosody.get('mean_pitch_hz', 0):.1f} Hz",
            f"**{t('pitch_var')}**: {output.prosody.get('pitch_std', 0):.1f}",
            f"**{t('tone')}**: {output.prosody.get('tone', 'N/A')}",
            f"**{t('prosody_style')}**: {output.prosody.get('prosody_style', 'N/A')}",
        ])
    
    if output.emotion:
        lines.extend([
            f"\n{t('emotion_section')}",
            f"**{t('main_emotion')}**: {output.emotion.get('dominant_emotion', 'N/A')} ({output.emotion.get('confidence', 0):.1%})",
        ])
        emotion_scores = output.emotion.get('emotion_scores', {})
        if emotion_scores:
            lines.append(f"**{t('emotion_dist')}:**")
            for emo, score in list(emotion_scores.items())[:5]:
                lines.append(f"  • {emo}: {score:.1%}")
    
    if output.text:
        text_preview = output.text[:500] + ('...' if len(output.text) > 500 else '')
        lines.extend([f"\n{t('transcript')}", f"```\n{text_preview}\n```"])
    
    return "\n".join(lines)


def format_yolo_output(output: YOLOOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    detection = output.detection
    environment = output.environment
    
    lines = [
        f"{t('yolo_result')}\n",
        t('env_analysis'),
        f"**{t('env_type')}**: {environment.get('environment_type', 'N/A')}",
        f"**{t('cook_style')}**: {environment.get('cooking_style', 'N/A')}",
        f"**{t('appliance_tier')}**: {environment.get('appliance_tier', 'N/A')}",
        f"\n{t('det_stats')}",
        f"**{t('unique_obj')}**: {detection.get('unique_objects', 0)}",
        f"**{t('total_det')}**: {detection.get('total_detections', 0)}",
        f"**{t('frames_proc')}**: {detection.get('frames_processed', 0)}",
        f"\n{t('detected_obj')}",
    ]
    
    object_counts = detection.get('object_counts', {})
    avg_conf = detection.get('avg_confidence', {})
    for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        conf = avg_conf.get(obj, 0)
        lines.append(f"  • **{obj}**: {count}× ({t('confidence')}: {conf:.1%})")
    
    colors = output.colors
    if colors and colors.get('detailed_analysis'):
        lines.append(f"\n{t('obj_colors')}")
        for obj, analysis in list(colors['detailed_analysis'].items())[:5]:
            dominant = analysis.get('dominant', 'Unknown')
            lines.append(f"  • **{obj}**: {dominant}")
    
    materials = output.materials
    if materials and materials.get('detailed_analysis'):
        lines.append(f"\n{t('obj_materials')}")
        for obj, analysis in list(materials['detailed_analysis'].items())[:5]:
            dominant = analysis.get('dominant', 'Unknown')
            lines.append(f"  • **{obj}**: {dominant}")
    
    return "\n".join(lines)


def format_consensus_output(output: ConsensusOutput) -> str:
    if not output or not output.success:
        return f"❌ {t('analysis_failed')}"
    
    lines = [
        f"{t('consensus_result')}\n",
        t('cam_consensus'),
        f"**{t('cam_angle')}**: {output.camera_angle}",
        format_distribution(output.camera_angle_detail),
        f"\n**{t('focal_tendency')}**: {output.focal_length_tendency}",
        f"**{t('cam_motion')}**: {output.camera_motion}",
        f"\n{t('color_consensus')}",
        f"**{t('hue')}**: {output.hue_family}",
        format_distribution(output.hue_detail),
        f"\n**{t('saturation')}**: {output.saturation}",
        f"**{t('brightness')}**: {output.brightness}",
        f"**{t('contrast')}**: {output.contrast}",
    ]
    if output.cct:
        lines.append(f"**{t('cct')}**: {output.cct:.0f}K")
    
    lines.append(f"\n{t('edit_consensus')}")
    if output.cuts_per_minute:
        lines.append(f"**{t('cuts_per_min')}**: {output.cuts_per_minute:.2f}")
    if output.avg_shot_length:
        lines.append(f"**{t('avg_shot_len')}**: {output.avg_shot_length:.2f}s")
    lines.append(f"**{t('transition')}**: {output.transition_type}")
    
    lines.extend([
        f"\n{t('audio_consensus')}",
        f"**{t('bgm_style')}**: {output.bgm_style}",
        format_distribution(output.bgm_style_detail),
        f"\n**{t('main_mood')}**: {output.bgm_mood}",
    ])
    if output.beat_alignment:
        lines.append(f"**{t('beat_align')}**: {output.beat_alignment:.2f}")
    if output.tempo_bpm:
        lines.append(f"**BPM**: {output.tempo_bpm:.1f}")
    
    lines.extend([
        f"\n{t('scene_consensus')}",
        f"**{t('scene_type')}**: {output.scene_category}",
        format_distribution(output.scene_category_detail),
    ])
    
    if output.yolo_available:
        lines.extend([
            f"\n{t('yolo_consensus')}",
            f"**{t('env')}**: {output.yolo_environment}",
            f"**{t('style')}**: {output.yolo_style}",
        ])
    
    return "\n".join(lines)


# =============================================================================
# 处理函数
# =============================================================================
def upload_video(video_file) -> Tuple[str, str, str, str, List[str]]:
    if video_file is None:
        return None, None, t('please_upload'), "", []
    
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
    
    return str(STATE.video_path), str(STATE.audio_path) if STATE.audio_path else None, status, "", frame_paths


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
        summary_lines.append(f"{t('cam_angle')}: {STATE.visual_output.camera_angle}")
        summary_lines.append(f"{t('hue')}: {STATE.visual_output.hue_family}")
        summary_lines.append(f"{t('cuts')}: {STATE.visual_output.cuts}")
    if STATE.audio_output:
        summary_lines.append(f"{t('bpm')}: {STATE.audio_output.tempo_bpm:.1f}")
        summary_lines.append(f"{t('bgm_style')}: {STATE.audio_output.bgm_style}")
    if STATE.asr_output:
        summary_lines.append(f"{t('speech_rate')}: {STATE.asr_output.words_per_minute:.1f} wpm")
    if STATE.yolo_output:
        obj_count = STATE.yolo_output.detection.get('unique_objects', 0)
        summary_lines.append(f"{t('objects')}: {obj_count}")
    
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
# Gradio 界面
# =============================================================================
def create_ui():
    with gr.Blocks(
        title="Video Style Analysis",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
        .container { max-width: 1600px; margin: auto; }
        .result-box { min-height: 300px; }
        .step-btn { min-width: 120px; }
        .lang-btn { min-width: 80px; }
        .progress-box { background: #f0f0f0; padding: 10px; border-radius: 8px; }
        """
    ) as demo:
        
        lang_state = gr.State("zh")
        
        with gr.Row():
            with gr.Column(scale=10):
                title_md = gr.Markdown(f"# {t('title')}\n### {t('subtitle')}\n\n{t('tech_stack')}")
            with gr.Column(scale=1, min_width=200):
                lang_btn = gr.Radio(
                    choices=[("中文", "zh"), ("English", "en")],
                    value="zh",
                    label=t('ui_lang'),
                    elem_classes="lang-btn"
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                upload_md = gr.Markdown(f"## {t('upload_video')}")
                video_input = gr.Video(label=t('upload_label'), height=280)
                upload_status = gr.Textbox(label=t('upload_status'), lines=4, interactive=False)
                
                audio_md = gr.Markdown(f"## {t('audio_preview')}")
                audio_player = gr.Audio(label=t('audio_label'), type="filepath")
                
                settings_md = gr.Markdown(f"## {t('settings')}")
                language_select = gr.Dropdown(
                    choices=[("English", "en"), ("中文", "zh"), ("日本語", "ja"), ("한국어", "ko"), ("Auto", "auto")],
                    value="en",
                    label=t('asr_lang')
                )
                
                frame_md = gr.Markdown(f"## {t('frame_preview')}")
                frame_gallery = gr.Gallery(label=t('frame_label'), columns=4, rows=3, height=300, object_fit="contain")
            
            with gr.Column(scale=2):
                ctrl_md = gr.Markdown(f"## {t('analysis_ctrl')}")
                
                with gr.Row():
                    run_all_btn = gr.Button(t('run_all'), variant="primary", size="lg", scale=2)
                    generate_report_btn = gr.Button(t('gen_report'), variant="secondary", size="lg")
                    export_json_btn = gr.Button(t('export_json'), size="lg")
                
                step_md = gr.Markdown(f"### {t('step_exec')}")
                with gr.Row():
                    run_visual_btn = gr.Button(t('visual_btn'), elem_classes="step-btn")
                    run_audio_btn = gr.Button(t('audio_btn'), elem_classes="step-btn")
                    run_asr_btn = gr.Button(t('asr_btn'), elem_classes="step-btn")
                    run_yolo_btn = gr.Button(t('yolo_btn'), elem_classes="step-btn")
                    run_consensus_btn = gr.Button(t('consensus_btn'), elem_classes="step-btn")
                
                with gr.Tabs() as result_tabs:
                    with gr.TabItem(t('tab_visual'), id="visual"):
                        visual_result = gr.Markdown(value=f"*{t('hint_upload')}*")
                        contact_sheet_img = gr.Image(label="Contact Sheet", height=200)
                    
                    with gr.TabItem(t('tab_audio'), id="audio"):
                        audio_result = gr.Markdown(value=f"*{t('hint_upload')}*")
                    
                    with gr.TabItem(t('tab_asr'), id="asr"):
                        asr_result = gr.Markdown(value=f"*{t('hint_upload')}*")
                    
                    with gr.TabItem(t('tab_yolo'), id="yolo"):
                        yolo_result = gr.Markdown(value=f"*{t('hint_upload')}*")
                    
                    with gr.TabItem(t('tab_consensus'), id="consensus"):
                        consensus_result = gr.Markdown(value=f"*{t('hint_consensus')}*")
            
            with gr.Column(scale=1):
                report_md = gr.Markdown(f"## {t('report_gen')}")
                report_status = gr.Textbox(label=t('report_status'), lines=5, interactive=False)
                
                download_md = gr.Markdown(f"### {t('download')}")
                report_download = gr.File(label=t('word_report'))
                pdf_download = gr.File(label=t('pdf_report'))
                
                gr.Markdown("---")
                json_status = gr.Textbox(label=t('json_status'), lines=2, interactive=False)
                json_download = gr.File(label=t('json_data'))
                
                summary_md = gr.Markdown(f"## {t('summary')}")
                summary_box = gr.Textbox(label=t('summary_label'), lines=10, interactive=False, placeholder=t('summary_placeholder'))
        
        footer_md = gr.Markdown(f"---\n{t('footer')}")
        
        # 语言切换
        def switch_language(lang):
            set_lang(lang)
            return [
                f"# {t('title')}\n### {t('subtitle')}\n\n{t('tech_stack')}",
                f"## {t('upload_video')}",
                f"## {t('audio_preview')}",
                f"## {t('settings')}",
                f"## {t('frame_preview')}",
                f"## {t('analysis_ctrl')}",
                f"### {t('step_exec')}",
                f"## {t('report_gen')}",
                f"### {t('download')}",
                f"## {t('summary')}",
                f"---\n{t('footer')}",
                t('run_all'),
                t('gen_report'),
                t('export_json'),
                t('visual_btn'),
                t('audio_btn'),
                t('asr_btn'),
                t('yolo_btn'),
                t('consensus_btn'),
                lang
            ]
        
        lang_btn.change(
            fn=switch_language,
            inputs=[lang_btn],
            outputs=[
                title_md, upload_md, audio_md, settings_md, frame_md,
                ctrl_md, step_md, report_md, download_md, summary_md, footer_md,
                run_all_btn, generate_report_btn, export_json_btn,
                run_visual_btn, run_audio_btn, run_asr_btn, run_yolo_btn, run_consensus_btn,
                lang_state
            ]
        )
        
        # 事件绑定
        video_input.change(fn=upload_video, inputs=[video_input],
                          outputs=[video_input, audio_player, upload_status, visual_result, frame_gallery])
        
        run_visual_btn.click(fn=run_visual_analysis, outputs=[visual_result, contact_sheet_img])
        run_audio_btn.click(fn=run_audio_analysis, outputs=[audio_result])
        run_asr_btn.click(fn=run_asr_analysis, inputs=[language_select], outputs=[asr_result])
        run_yolo_btn.click(fn=run_yolo_analysis, outputs=[yolo_result])
        run_consensus_btn.click(fn=run_consensus_analysis, outputs=[consensus_result])
        
        run_all_btn.click(fn=run_all_analysis, inputs=[language_select],
                         outputs=[visual_result, contact_sheet_img, audio_result, asr_result, yolo_result, consensus_result, summary_box])
        
        generate_report_btn.click(fn=generate_report, outputs=[report_status, report_download, pdf_download])
        export_json_btn.click(fn=export_json, outputs=[json_status, json_download])
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
