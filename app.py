#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Style Analysis - Gradio Web Interface

功能:
- 上传视频文件
- 预览/播放视频和音频
- 一键处理或分步处理
- 实时显示每一步分析结果
- 生成 PDF 报告 (在线预览和下载)
- 每帧分析结果可视化

技术栈:
- CLIP (场景分类)
- CLAP (音频分类)
- HuBERT (语音情感)
- Whisper large-v3-turbo (ASR)
- YOLO11 (目标检测)
"""

import sys
import os
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import gradio as gr
import numpy as np
import cv2

# 将 src 目录加入路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from steps import (
    VisualAnalysisStep,
    AudioAnalysisStep,
    ASRAnalysisStep,
    YOLOAnalysisStep,
    ConsensusStep,
    ReportGenerationStep,
    VideoInput,
    AudioInput,
    ASRInput,
    YOLOInput,
    ConsensusInput,
    ReportInput,
    VideoMetrics,
    VisualOutput,
    AudioOutput,
    ASROutput,
    YOLOOutput,
    ConsensusOutput,
)
from report_word import generate_word_report

# ============================================================================
# 全局状态
# ============================================================================
class AnalysisState:
    """管理分析状态"""
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

# ============================================================================
# 工具函数
# ============================================================================
def extract_audio_from_video(video_path: Path, output_dir: Path) -> Optional[Path]:
    """从视频中提取音频"""
    output_path = output_dir / f"{video_path.stem}_audio.wav"
    
    if output_path.exists():
        return output_path
    
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "22050", "-ac", "1",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except Exception as e:
        print(f"音频提取失败: {e}")
        return None


def extract_frames_for_gallery(video_path: Path, output_dir: Path, num_frames: int = 12) -> List[str]:
    """提取关键帧用于画廊展示"""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        return []
    
    # 均匀采样帧
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


def get_frame_info(frame_idx: int, visual_output: Optional[VisualOutput]) -> str:
    """获取单帧的分析信息"""
    if not visual_output:
        return "未分析"
    
    # 从 per_frame_analysis 获取信息 (如果有)
    per_frame = getattr(visual_output, 'per_frame_analysis', None)
    if per_frame and frame_idx < len(per_frame):
        frame_data = per_frame[frame_idx]
        return f"亮度: {frame_data.get('brightness', 'N/A')} | 色调: {frame_data.get('hue', 'N/A')}"
    
    return f"帧 {frame_idx + 1}"


def convert_docx_to_pdf(docx_path: str) -> Optional[str]:
    """将 DOCX 转换为 PDF"""
    pdf_path = docx_path.replace('.docx', '.pdf')
    
    try:
        # 尝试使用 libreoffice
        cmd = [
            "libreoffice", "--headless", "--convert-to", "pdf",
            "--outdir", str(Path(docx_path).parent),
            docx_path
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        if Path(pdf_path).exists():
            return pdf_path
    except:
        pass
    
    try:
        # 尝试使用 docx2pdf (Windows/Mac)
        from docx2pdf import convert
        convert(docx_path, pdf_path)
        return pdf_path
    except:
        pass
    
    return None


def format_distribution(detail: Dict) -> str:
    """格式化分布信息"""
    if not detail or 'distribution' not in detail:
        return "N/A"
    
    lines = []
    for item in detail.get('distribution', [])[:5]:
        value = item.get('value', 'Unknown')
        count = item.get('count', 0)
        pct = item.get('percentage', 0)
        lines.append(f"  • {value}: {count}次 ({pct}%)")
    
    return "\n".join(lines) if lines else "N/A"


def format_visual_output(output: VisualOutput) -> str:
    """格式化视觉分析结果"""
    if not output or not output.success:
        return "❌ 分析失败"
    
    lines = [
        "# 📹 视觉分析结果\n",
        f"**时长**: {output.duration:.2f}s | **FPS**: {output.fps:.1f} | **采样帧数**: {output.sampled_frames}\n",
        
        "## 📷 镜头分析",
        f"**镜头角度**: {output.camera_angle}",
        format_distribution(output.camera_angle_detail),
        f"\n**焦距倾向**: {output.focal_length_tendency}",
        
        "\n## 🎨 色彩分析",
        f"**色调**: {output.hue_family}",
        format_distribution(output.hue_detail),
        f"\n**饱和度**: {output.saturation_band}",
        format_distribution(output.saturation_detail),
        f"\n**亮度**: {output.brightness_band}",
        format_distribution(output.brightness_detail),
        f"\n**对比度**: {output.contrast}",
        f"\n**色温**: {output.cct_mean:.0f}K" if output.cct_mean else "",
        
        "\n## ✂️ 剪辑分析",
        f"**剪辑数**: {output.cuts}",
        f"**平均镜头时长**: {output.avg_shot_length:.2f}s",
        f"**转场类型**: {output.transition_type}",
        
        "\n## 🏠 场景分类 (CLIP)",
    ]
    
    for scene in output.scene_categories[:3]:
        label = scene.get('label', 'Unknown')
        prob = scene.get('probability', 0)
        lines.append(f"  • {label}: {prob:.1%}")
    
    return "\n".join(lines)


def format_audio_output(output: AudioOutput) -> str:
    """格式化音频分析结果"""
    if not output or not output.success:
        return "❌ 分析失败"
    
    lines = [
        "# 🎵 音频分析结果 (CLAP)\n",
        
        "## 节奏分析",
        f"**BPM**: {output.tempo_bpm:.1f}",
        f"**节拍数**: {output.num_beats}",
        f"**打击乐比例**: {output.percussive_ratio:.2f}",
        
        "\n## 🎸 BGM 风格",
        f"**主要风格**: {output.bgm_style} ({output.bgm_style_confidence:.1%})",
    ]
    
    if output.bgm_style_detail and output.bgm_style_detail.get('top_3'):
        lines.append("**Top 3 风格:**")
        for item in output.bgm_style_detail['top_3'][:3]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(f"  • {item[0]}: {item[1]:.1%}")
    
    lines.extend([
        "\n## 😊 情绪分析",
        f"**主要情绪**: {output.mood} ({output.mood_confidence:.1%})",
    ])
    
    if output.mood_detail and output.mood_detail.get('top_3'):
        lines.append("**Top 3 情绪:**")
        for item in output.mood_detail['top_3'][:3]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(f"  • {item[0]}: {item[1]:.1%}")
    
    lines.extend([
        f"\n## 🎹 其他",
        f"**调式**: {output.key_signature}",
        f"**语音比例**: {output.speech_ratio:.2f}",
    ])
    
    instruments = output.instruments.get('detected_instruments', [])
    if instruments:
        lines.append(f"**检测到的乐器**: {', '.join(instruments)}")
    
    return "\n".join(lines)


def format_asr_output(output: ASROutput) -> str:
    """格式化 ASR 分析结果"""
    if not output or not output.success:
        return "❌ 分析失败"
    
    lines = [
        "# 🎤 语音分析结果 (Whisper + HuBERT)\n",
        
        "## 📝 转录统计",
        f"**词数**: {output.num_words}",
        f"**语速**: {output.words_per_second:.2f} w/s ({output.words_per_minute:.1f} wpm)",
        f"**节奏**: {output.pace}",
        f"**停顿数**: {output.num_pauses}",
        f"**停顿风格**: {output.pause_style}",
    ]
    
    if output.catchphrases:
        lines.append("\n## 🔁 口头禅 (高频短语)")
        for phrase in output.catchphrases[:10]:
            lines.append(f"  • \"{phrase}\"")
    
    if output.prosody:
        lines.extend([
            "\n## 🎼 韵律分析",
            f"**平均音高**: {output.prosody.get('mean_pitch_hz', 0):.1f} Hz",
            f"**音高变化**: {output.prosody.get('pitch_std', 0):.1f}",
            f"**音调**: {output.prosody.get('tone', 'N/A')}",
            f"**韵律风格**: {output.prosody.get('prosody_style', 'N/A')}",
        ])
    
    if output.emotion:
        lines.extend([
            "\n## 😊 语音情感 (HuBERT)",
            f"**主要情感**: {output.emotion.get('dominant_emotion', 'N/A')} ({output.emotion.get('confidence', 0):.1%})",
        ])
        emotion_scores = output.emotion.get('emotion_scores', {})
        if emotion_scores:
            lines.append("**情感分布:**")
            for emo, score in list(emotion_scores.items())[:5]:
                lines.append(f"  • {emo}: {score:.1%}")
    
    if output.text:
        lines.extend([
            "\n## 📜 转录文本",
            f"```\n{output.text[:500]}{'...' if len(output.text) > 500 else ''}\n```"
        ])
    
    return "\n".join(lines)


def format_yolo_output(output: YOLOOutput) -> str:
    """格式化 YOLO 分析结果"""
    if not output or not output.success:
        return "❌ 分析失败"
    
    detection = output.detection
    environment = output.environment
    
    lines = [
        "# 🔍 目标检测结果 (YOLO11)\n",
        
        "## 🏠 环境分析",
        f"**环境类型**: {environment.get('environment_type', 'N/A')}",
        f"**烹饪风格**: {environment.get('cooking_style', 'N/A')}",
        f"**设备档次**: {environment.get('appliance_tier', 'N/A')}",
        
        "\n## 📊 检测统计",
        f"**检测物体类数**: {detection.get('unique_objects', 0)}",
        f"**总检测次数**: {detection.get('total_detections', 0)}",
        f"**处理帧数**: {detection.get('frames_processed', 0)}",
        
        "\n## 🎯 检测到的物体",
    ]
    
    object_counts = detection.get('object_counts', {})
    avg_conf = detection.get('avg_confidence', {})
    for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        conf = avg_conf.get(obj, 0)
        lines.append(f"  • **{obj}**: {count}次 (置信度: {conf:.1%})")
    
    # 颜色分析
    colors = output.colors
    if colors and colors.get('detailed_analysis'):
        lines.append("\n## 🎨 物体颜色")
        for obj, analysis in list(colors['detailed_analysis'].items())[:5]:
            dominant = analysis.get('dominant', 'Unknown')
            lines.append(f"  • **{obj}**: {dominant}")
    
    # 材质分析
    materials = output.materials
    if materials and materials.get('detailed_analysis'):
        lines.append("\n## 🧱 物体材质")
        for obj, analysis in list(materials['detailed_analysis'].items())[:5]:
            dominant = analysis.get('dominant', 'Unknown')
            lines.append(f"  • **{obj}**: {dominant}")
    
    return "\n".join(lines)


def format_consensus_output(output: ConsensusOutput) -> str:
    """格式化共识分析结果"""
    if not output or not output.success:
        return "❌ 分析失败"
    
    lines = [
        "# 🎯 跨视频共识分析\n",
        
        "## 📷 镜头共识",
        f"**镜头角度**: {output.camera_angle}",
        format_distribution(output.camera_angle_detail),
        f"\n**焦距倾向**: {output.focal_length_tendency}",
        f"**相机运动**: {output.camera_motion}",
        
        "\n## 🎨 色彩共识",
        f"**色调**: {output.hue_family}",
        format_distribution(output.hue_detail),
        f"\n**饱和度**: {output.saturation}",
        f"**亮度**: {output.brightness}",
        f"**对比度**: {output.contrast}",
        f"**色温**: {output.cct:.0f}K" if output.cct else "",
        
        "\n## ✂️ 剪辑共识",
        f"**剪辑/分钟**: {output.cuts_per_minute:.2f}" if output.cuts_per_minute else "",
        f"**平均镜头时长**: {output.avg_shot_length:.2f}s" if output.avg_shot_length else "",
        f"**转场类型**: {output.transition_type}",
        
        "\n## 🎵 音频共识",
        f"**BGM 风格**: {output.bgm_style}",
        format_distribution(output.bgm_style_detail),
        f"\n**BGM 情绪**: {output.bgm_mood}",
        f"**节拍对齐**: {output.beat_alignment:.2f}" if output.beat_alignment else "",
        f"**BPM**: {output.tempo_bpm:.1f}" if output.tempo_bpm else "",
        
        "\n## 🏠 场景共识",
        f"**场景类型**: {output.scene_category}",
        format_distribution(output.scene_category_detail),
    ]
    
    if output.yolo_available:
        lines.extend([
            "\n## 🔍 YOLO 共识",
            f"**环境**: {output.yolo_environment}",
            f"**风格**: {output.yolo_style}",
        ])
    
    return "\n".join(lines)


# ============================================================================
# 处理函数
# ============================================================================
def upload_video(video_file) -> Tuple[str, str, str, str, List[str]]:
    """处理视频上传"""
    if video_file is None:
        return None, None, "请上传视频文件", "", []
    
    STATE.reset()
    
    # 创建工作目录
    STATE.work_dir = Path(tempfile.mkdtemp(prefix="video_analysis_"))
    
    # 复制视频到工作目录
    video_path = Path(video_file)
    STATE.video_path = STATE.work_dir / video_path.name
    
    import shutil
    shutil.copy(video_file, STATE.video_path)
    
    # 提取音频
    STATE.audio_path = extract_audio_from_video(STATE.video_path, STATE.work_dir)
    
    # 提取关键帧用于画廊
    frame_paths = extract_frames_for_gallery(STATE.video_path, STATE.work_dir, num_frames=12)
    
    status = f"✅ 视频已上传: {video_path.name}\n"
    status += f"📁 工作目录: {STATE.work_dir}\n"
    status += f"🖼️ 提取了 {len(frame_paths)} 个关键帧\n"
    
    if STATE.audio_path:
        status += f"🎵 音频已提取: {STATE.audio_path.name}"
    else:
        status += "⚠️ 音频提取失败"
    
    return str(STATE.video_path), str(STATE.audio_path) if STATE.audio_path else None, status, "", frame_paths


def run_visual_analysis(progress=gr.Progress()) -> Tuple[str, str]:
    """运行视觉分析"""
    if STATE.video_path is None:
        return "❌ 请先上传视频", None
    
    progress(0.1, desc="初始化视觉分析...")
    
    try:
        step = VisualAnalysisStep()
        input_data = VideoInput(
            video_path=STATE.video_path,
            work_dir=STATE.work_dir,
            frame_mode="edge"
        )
        
        progress(0.3, desc="分析中...")
        STATE.visual_output = step.run(input_data)
        
        progress(1.0, desc="完成!")
        
        result = format_visual_output(STATE.visual_output)
        contact_sheet = STATE.visual_output.contact_sheet if STATE.visual_output else None
        
        return result, contact_sheet
        
    except Exception as e:
        return f"❌ 视觉分析失败: {str(e)}", None


def run_audio_analysis(progress=gr.Progress()) -> str:
    """运行音频分析"""
    if STATE.audio_path is None:
        return "❌ 请先上传视频并提取音频"
    
    progress(0.1, desc="初始化音频分析...")
    
    try:
        step = AudioAnalysisStep()
        input_data = AudioInput(audio_path=STATE.audio_path)
        
        progress(0.3, desc="CLAP 分析中...")
        STATE.audio_output = step.run(input_data)
        
        progress(1.0, desc="完成!")
        
        return format_audio_output(STATE.audio_output)
        
    except Exception as e:
        return f"❌ 音频分析失败: {str(e)}"


def run_asr_analysis(language: str, progress=gr.Progress()) -> str:
    """运行 ASR 分析"""
    if STATE.audio_path is None:
        return "❌ 请先上传视频并提取音频"
    
    progress(0.1, desc="初始化 ASR...")
    
    try:
        step = ASRAnalysisStep()
        input_data = ASRInput(
            audio_path=STATE.audio_path,
            language=language,
            model_size="large-v3-turbo",
            enable_prosody=True,
            enable_emotion=True
        )
        
        progress(0.3, desc="Whisper 转录中...")
        STATE.asr_output = step.run(input_data)
        
        progress(1.0, desc="完成!")
        
        return format_asr_output(STATE.asr_output)
        
    except Exception as e:
        return f"❌ ASR 分析失败: {str(e)}"


def run_yolo_analysis(progress=gr.Progress()) -> str:
    """运行 YOLO 分析"""
    if STATE.video_path is None:
        return "❌ 请先上传视频"
    
    progress(0.1, desc="初始化 YOLO11...")
    
    try:
        step = YOLOAnalysisStep()
        input_data = YOLOInput(
            video_path=STATE.video_path,
            target_frames=36,
            enable_colors=True,
            enable_materials=True
        )
        
        progress(0.3, desc="目标检测中...")
        STATE.yolo_output = step.run(input_data)
        
        progress(1.0, desc="完成!")
        
        return format_yolo_output(STATE.yolo_output)
        
    except Exception as e:
        return f"❌ YOLO 分析失败: {str(e)}"


def run_all_analysis(language: str, progress=gr.Progress()) -> Tuple[str, str, str, str, str]:
    """一键运行所有分析"""
    results = []
    contact_sheet = None
    
    # 视觉分析
    progress(0.1, desc="视觉分析...")
    visual_result, contact_sheet = run_visual_analysis()
    results.append(visual_result)
    
    # 音频分析
    progress(0.3, desc="音频分析...")
    audio_result = run_audio_analysis()
    results.append(audio_result)
    
    # ASR 分析
    progress(0.5, desc="ASR 分析...")
    asr_result = run_asr_analysis(language)
    results.append(asr_result)
    
    # YOLO 分析
    progress(0.7, desc="YOLO 分析...")
    yolo_result = run_yolo_analysis()
    results.append(yolo_result)
    
    # 共识分析
    progress(0.9, desc="生成共识...")
    consensus_result = run_consensus_analysis()
    results.append(consensus_result)
    
    progress(1.0, desc="完成!")
    
    return results[0], contact_sheet, results[1], results[2], results[3]


def run_consensus_analysis() -> str:
    """运行共识分析"""
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
        return f"❌ 共识分析失败: {str(e)}"


def generate_report(progress=gr.Progress()) -> Tuple[str, str, str]:
    """生成报告"""
    if STATE.video_path is None:
        return "❌ 请先运行分析", None, None
    
    progress(0.2, desc="生成 Word 报告...")
    
    try:
        # 准备数据
        metrics = VideoMetrics(path=str(STATE.video_path))
        metrics.visual = STATE.visual_output
        metrics.audio = STATE.audio_output
        metrics.asr = STATE.asr_output
        metrics.yolo = STATE.yolo_output
        
        if STATE.consensus_output is None:
            run_consensus_analysis()
        
        # 转换为字典格式
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
        
        # 生成报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = STATE.work_dir / f"report_{timestamp}.docx"
        
        generate_word_report(
            video_metrics=[metrics_dict],
            consensus=consensus_dict,
            output_path=str(report_path),
            show_screenshots=True
        )
        
        STATE.report_path = str(report_path)
        
        progress(0.6, desc="转换为 PDF...")
        
        # 尝试转换为 PDF
        STATE.pdf_path = convert_docx_to_pdf(STATE.report_path)
        
        progress(1.0, desc="完成!")
        
        status = f"✅ 报告已生成\n"
        status += f"📄 Word: {report_path.name}\n"
        if STATE.pdf_path:
            status += f"📕 PDF: {Path(STATE.pdf_path).name}"
        else:
            status += "⚠️ PDF 转换失败 (需要 libreoffice)"
        
        return status, STATE.report_path, STATE.pdf_path
        
    except Exception as e:
        return f"❌ 报告生成失败: {str(e)}", None, None


def export_json() -> Tuple[str, str]:
    """导出 JSON 数据"""
    if STATE.video_path is None:
        return "❌ 请先运行分析", None
    
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
        
        return f"✅ JSON 已导出: {json_path.name}", str(json_path)
        
    except Exception as e:
        return f"❌ JSON 导出失败: {str(e)}", None


# ============================================================================
# Gradio 界面
# ============================================================================
def create_ui():
    """创建 Gradio 界面"""
    
    with gr.Blocks(
        title="Video Style Analysis",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .container { max-width: 1600px; margin: auto; }
        .header { text-align: center; padding: 20px; }
        .result-box { min-height: 300px; }
        .step-btn { min-width: 150px; }
        .frame-gallery { max-height: 400px; overflow-y: auto; }
        .status-box { font-family: monospace; }
        """
    ) as demo:
        
        # 标题
        gr.Markdown("""
        # 🎬 Video Style Analysis
        ### SOTA 视频风格分析系统 | PyTorch + HuggingFace
        
        **技术栈**: 
        🖼️ CLIP ViT-L/14 (场景) | 
        🎵 CLAP (音频) | 
        😊 HuBERT-large (情感) | 
        🎤 Whisper large-v3-turbo (ASR) | 
        🔍 YOLO11 (检测)
        """)
        
        with gr.Row():
            # ==================== 左侧: 上传和预览 ====================
            with gr.Column(scale=1):
                gr.Markdown("## 📤 上传视频")
                
                video_input = gr.Video(
                    label="上传视频文件 (支持 mp4, avi, mov, mkv)",
                    height=280
                )
                
                upload_status = gr.Textbox(
                    label="上传状态",
                    lines=4,
                    interactive=False,
                    elem_classes="status-box"
                )
                
                gr.Markdown("## 🎵 音频预览")
                audio_player = gr.Audio(
                    label="提取的音频 (自动从视频分离)",
                    type="filepath"
                )
                
                gr.Markdown("## ⚙️ 分析设置")
                with gr.Row():
                    language_select = gr.Dropdown(
                        choices=[
                            ("English", "en"),
                            ("中文", "zh"),
                            ("日本語", "ja"),
                            ("한국어", "ko"),
                            ("自动检测", "auto")
                        ],
                        value="en",
                        label="ASR 语言"
                    )
                
                # 帧画廊
                gr.Markdown("## 🖼️ 关键帧预览")
                frame_gallery = gr.Gallery(
                    label="视频关键帧 (均匀采样)",
                    columns=4,
                    rows=3,
                    height=300,
                    object_fit="contain",
                    elem_classes="frame-gallery"
                )
            
            # ==================== 中间: 控制和结果 ====================
            with gr.Column(scale=2):
                gr.Markdown("## 🚀 分析控制")
                
                with gr.Row():
                    run_all_btn = gr.Button(
                        "🎯 一键分析全部", 
                        variant="primary", 
                        size="lg",
                        scale=2
                    )
                    generate_report_btn = gr.Button(
                        "📄 生成报告", 
                        variant="secondary", 
                        size="lg"
                    )
                    export_json_btn = gr.Button(
                        "💾 导出 JSON", 
                        size="lg"
                    )
                
                gr.Markdown("### 🔧 分步执行 (可单独运行每个模块)")
                with gr.Row():
                    run_visual_btn = gr.Button("📹 视觉分析", elem_classes="step-btn")
                    run_audio_btn = gr.Button("🎵 音频分析", elem_classes="step-btn")
                    run_asr_btn = gr.Button("🎤 ASR 分析", elem_classes="step-btn")
                    run_yolo_btn = gr.Button("🔍 YOLO 分析", elem_classes="step-btn")
                    run_consensus_btn = gr.Button("🎯 共识计算", elem_classes="step-btn")
                
                # 进度条
                progress_text = gr.Textbox(
                    label="当前进度",
                    value="等待开始...",
                    interactive=False,
                    lines=1
                )
                
                # 结果选项卡
                with gr.Tabs() as result_tabs:
                    with gr.TabItem("📹 视觉分析", id="visual"):
                        visual_result = gr.Markdown(
                            value="*上传视频后点击「视觉分析」或「一键分析」开始*",
                            elem_classes="result-box"
                        )
                        contact_sheet_img = gr.Image(
                            label="Contact Sheet (关键帧拼接)",
                            height=200
                        )
                    
                    with gr.TabItem("🎵 音频分析", id="audio"):
                        audio_result = gr.Markdown(
                            value="*上传视频后点击「音频分析」或「一键分析」开始*",
                            elem_classes="result-box"
                        )
                    
                    with gr.TabItem("🎤 ASR 分析", id="asr"):
                        asr_result = gr.Markdown(
                            value="*上传视频后点击「ASR 分析」或「一键分析」开始*",
                            elem_classes="result-box"
                        )
                    
                    with gr.TabItem("🔍 YOLO 分析", id="yolo"):
                        yolo_result = gr.Markdown(
                            value="*上传视频后点击「YOLO 分析」或「一键分析」开始*",
                            elem_classes="result-box"
                        )
                    
                    with gr.TabItem("🎯 共识分析", id="consensus"):
                        consensus_result = gr.Markdown(
                            value="*运行完所有分析后自动生成*",
                            elem_classes="result-box"
                        )
            
            # ==================== 右侧: 报告和导出 ====================
            with gr.Column(scale=1):
                gr.Markdown("## 📊 报告生成")
                
                report_status = gr.Textbox(
                    label="报告状态",
                    lines=5,
                    interactive=False,
                    elem_classes="status-box"
                )
                
                gr.Markdown("### 📥 下载")
                
                report_download = gr.File(
                    label="📄 Word 报告 (.docx)"
                )
                
                pdf_download = gr.File(
                    label="📕 PDF 报告 (.pdf)"
                )
                
                gr.Markdown("---")
                
                json_status = gr.Textbox(
                    label="JSON 导出状态",
                    lines=2,
                    interactive=False
                )
                
                json_download = gr.File(
                    label="💾 JSON 数据"
                )
                
                # 分析摘要
                gr.Markdown("## 📋 分析摘要")
                summary_box = gr.Textbox(
                    label="快速预览",
                    lines=10,
                    interactive=False,
                    placeholder="分析完成后显示摘要..."
                )
        
        # ==================== 事件绑定 ====================
        
        # 上传视频
        video_input.change(
            fn=upload_video,
            inputs=[video_input],
            outputs=[video_input, audio_player, upload_status, visual_result, frame_gallery]
        )
        
        # 分步执行
        run_visual_btn.click(
            fn=run_visual_analysis,
            outputs=[visual_result, contact_sheet_img]
        )
        
        run_audio_btn.click(
            fn=run_audio_analysis,
            outputs=[audio_result]
        )
        
        run_asr_btn.click(
            fn=run_asr_analysis,
            inputs=[language_select],
            outputs=[asr_result]
        )
        
        run_yolo_btn.click(
            fn=run_yolo_analysis,
            outputs=[yolo_result]
        )
        
        run_consensus_btn.click(
            fn=run_consensus_analysis,
            outputs=[consensus_result]
        )
        
        # 一键分析
        def run_all_with_summary(language, progress=gr.Progress()):
            """一键分析并生成摘要"""
            results = list(run_all_analysis(language, progress))
            consensus = run_consensus_analysis()
            
            # 生成摘要
            summary_lines = ["=== 分析摘要 ===\n"]
            if STATE.visual_output:
                summary_lines.append(f"📹 镜头角度: {STATE.visual_output.camera_angle}")
                summary_lines.append(f"🎨 色调: {STATE.visual_output.hue_family}")
                summary_lines.append(f"✂️ 剪辑数: {STATE.visual_output.cuts}")
            if STATE.audio_output:
                summary_lines.append(f"🎵 BPM: {STATE.audio_output.tempo_bpm:.1f}")
                summary_lines.append(f"🎸 BGM 风格: {STATE.audio_output.bgm_style}")
            if STATE.asr_output:
                summary_lines.append(f"🎤 语速: {STATE.asr_output.words_per_minute:.1f} wpm")
            if STATE.yolo_output:
                obj_count = STATE.yolo_output.detection.get('unique_objects', 0)
                summary_lines.append(f"🔍 检测物体: {obj_count} 种")
            
            summary = "\n".join(summary_lines)
            return results + [consensus, summary]
        
        run_all_btn.click(
            fn=run_all_with_summary,
            inputs=[language_select],
            outputs=[visual_result, contact_sheet_img, audio_result, asr_result, yolo_result, consensus_result, summary_box]
        )
        
        # 生成报告
        generate_report_btn.click(
            fn=generate_report,
            outputs=[report_status, report_download, pdf_download]
        )
        
        # 导出 JSON
        export_json_btn.click(
            fn=export_json,
            outputs=[json_status, json_download]
        )
        
        # 页脚
        gr.Markdown("""
        ---
        **Video Style Analysis** | SOTA 2025/2026 | 
        [GitHub](https://github.com/your-repo) | 
        Models: CLIP, CLAP, HuBERT, Whisper, YOLO11
        """)
    
    return demo


# ============================================================================
# 主函数
# ============================================================================
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
