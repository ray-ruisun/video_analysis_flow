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


# =============================================================================
# 结果格式化
# =============================================================================
def format_visual(output: VisualOutput) -> str:
    if not output or not output.success:
        return "❌ 分析失败"
    return f"""### 📹 视觉分析结果

**基本信息**: 时长 {output.duration:.2f}s | FPS {output.fps:.1f} | 采样 {output.sampled_frames} 帧

**镜头**: {output.camera_angle} | 焦距 {output.focal_length_tendency}

**色彩**: {output.hue_family} | 饱和度 {output.saturation_band} | 亮度 {output.brightness_band} | 对比度 {output.contrast}

**色温**: {output.cct_mean:.0f}K

**剪辑**: {output.cuts} 次剪辑 | 平均镜头 {output.avg_shot_length:.2f}s | {output.transition_type}

**场景 (CLIP)**:
{chr(10).join([f"  • {s.get('label', '?')}: {s.get('probability', 0):.1%}" for s in output.scene_categories[:3]])}
"""


def format_audio(output: AudioOutput) -> str:
    if not output or not output.success:
        return "❌ 分析失败"
    instruments = output.instruments.get('detected_instruments', [])
    return f"""### 🎵 音频分析结果 (CLAP)

**节奏**: BPM {output.tempo_bpm:.1f} | 节拍 {output.num_beats} | 打击乐比例 {output.percussive_ratio:.2f}

**BGM 风格**: {output.bgm_style} ({output.bgm_style_confidence:.1%})

**情绪**: {output.mood} ({output.mood_confidence:.1%})

**调式**: {output.key_signature} | 语音比例 {output.speech_ratio:.2f}

**乐器**: {', '.join(instruments) if instruments else 'N/A'}
"""


def format_asr(output: ASROutput) -> str:
    if not output or not output.success:
        return "❌ 分析失败"
    
    text_preview = output.text[:300] + '...' if len(output.text) > 300 else output.text
    emotion_str = ""
    if output.emotion:
        emotion_str = f"\n**情感**: {output.emotion.get('dominant_emotion', 'N/A')} ({output.emotion.get('confidence', 0):.1%})"
    
    prosody_str = ""
    if output.prosody:
        prosody_str = f"\n**韵律**: 音高 {output.prosody.get('mean_pitch_hz', 0):.1f}Hz | {output.prosody.get('prosody_style', 'N/A')}"
    
    return f"""### 🎤 语音分析结果 (Whisper + HuBERT)

**统计**: {output.num_words} 词 | {output.words_per_minute:.1f} wpm | {output.pace}

**口头禅**: {', '.join([f'"{p}"' for p in output.catchphrases[:5]]) if output.catchphrases else 'N/A'}
{prosody_str}{emotion_str}

**转录**:
```
{text_preview}
```
"""


def format_yolo(output: YOLOOutput) -> str:
    if not output or not output.success:
        return "❌ 分析失败"
    
    detection = output.detection
    environment = output.environment
    object_counts = detection.get('object_counts', {})
    
    objects_str = "\n".join([f"  • {obj}: {cnt}×" for obj, cnt in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]])
    
    return f"""### 🔍 目标检测结果 (YOLO11)

**环境**: {environment.get('environment_type', 'N/A')} | {environment.get('cooking_style', 'N/A')}

**统计**: {detection.get('unique_objects', 0)} 种物体 | {detection.get('total_detections', 0)} 次检测

**检测到的物体**:
{objects_str}
"""


def format_consensus(output: ConsensusOutput) -> str:
    if not output or not output.success:
        return "❌ 分析失败"
    
    cct_str = f"{output.cct:.0f}K" if output.cct else "N/A"
    shot_str = f"{output.avg_shot_length:.2f}s" if output.avg_shot_length else "N/A"
    bpm_str = f"{output.tempo_bpm:.1f}" if output.tempo_bpm else "N/A"
    
    return f"""### 🎯 共识分析结果

**镜头**: {output.camera_angle} | 焦距 {output.focal_length_tendency} | 运动 {output.camera_motion}

**色彩**: {output.hue_family} | 饱和度 {output.saturation} | 亮度 {output.brightness}

**色温**: {cct_str}

**剪辑**: {output.transition_type} | 平均 {shot_str}

**音频**: {output.bgm_style} | 情绪 {output.bgm_mood} | BPM {bpm_str}

**场景**: {output.scene_category}
"""


# =============================================================================
# 处理函数
# =============================================================================
def upload_video(video_file):
    """处理视频上传"""
    if video_file is None:
        return "请上传视频", None, []
    
    STATE.reset()
    STATE.work_dir = Path(tempfile.mkdtemp(prefix="video_analysis_"))
    
    video_path = Path(video_file)
    STATE.video_path = STATE.work_dir / video_path.name
    
    import shutil
    shutil.copy(video_file, STATE.video_path)
    
    STATE.audio_path = extract_audio_from_video(STATE.video_path, STATE.work_dir)
    frame_paths = extract_frames_for_gallery(STATE.video_path, STATE.work_dir, num_frames=12)
    
    status = f"✅ 已上传: {video_path.name}\n"
    status += f"📁 工作目录: {STATE.work_dir}\n"
    status += f"🖼️ 提取 {len(frame_paths)} 帧\n"
    status += "🎵 音频已提取" if STATE.audio_path else "⚠️ 音频提取失败"
    
    audio_path = str(STATE.audio_path) if STATE.audio_path else None
    return status, audio_path, frame_paths


def run_visual(progress=gr.Progress()):
    if STATE.video_path is None:
        return "❌ 请先上传视频", None
    
    progress(0.1, desc="⏳ 加载 CLIP...")
    step = VisualAnalysisStep()
    input_data = VideoInput(video_path=STATE.video_path, work_dir=STATE.work_dir, frame_mode="edge")
    
    progress(0.4, desc="🔄 视觉分析中...")
    STATE.visual_output = step.run(input_data)
    
    progress(1.0, desc="✅ 完成")
    contact = STATE.visual_output.contact_sheet if STATE.visual_output else None
    return format_visual(STATE.visual_output), contact


def run_audio(progress=gr.Progress()):
    if STATE.audio_path is None:
        return "❌ 请先上传视频"
    
    progress(0.1, desc="⏳ 加载 CLAP...")
    step = AudioAnalysisStep()
    input_data = AudioInput(audio_path=STATE.audio_path)
    
    progress(0.4, desc="🔄 音频分析中...")
    STATE.audio_output = step.run(input_data)
    
    progress(1.0, desc="✅ 完成")
    return format_audio(STATE.audio_output)


def run_asr(language: str, progress=gr.Progress()):
    if STATE.audio_path is None:
        return "❌ 请先上传视频"
    
    progress(0.1, desc="⏳ 加载 Whisper...")
    step = ASRAnalysisStep()
    input_data = ASRInput(audio_path=STATE.audio_path, language=language,
                          model_size="large-v3-turbo", enable_prosody=True, enable_emotion=True)
    
    progress(0.4, desc="🔄 语音识别中...")
    STATE.asr_output = step.run(input_data)
    
    progress(1.0, desc="✅ 完成")
    return format_asr(STATE.asr_output)


def run_yolo(progress=gr.Progress()):
    if STATE.video_path is None:
        return "❌ 请先上传视频"
    
    progress(0.1, desc="⏳ 加载 YOLO11...")
    step = YOLOAnalysisStep()
    input_data = YOLOInput(video_path=STATE.video_path, target_frames=36,
                           enable_colors=True, enable_materials=True)
    
    progress(0.4, desc="🔄 目标检测中...")
    STATE.yolo_output = step.run(input_data)
    
    progress(1.0, desc="✅ 完成")
    return format_yolo(STATE.yolo_output)


def run_consensus():
    """运行共识分析 - 需要先运行其他分析"""
    if STATE.visual_output is None and STATE.audio_output is None:
        return "❌ 请先运行视觉或音频分析"
    
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
    """一键分析全部"""
    progress(0.1, desc="📹 视觉分析...")
    visual_result, contact = run_visual()
    
    progress(0.3, desc="🎵 音频分析...")
    audio_result = run_audio()
    
    progress(0.5, desc="🎤 语音分析...")
    asr_result = run_asr(language)
    
    progress(0.7, desc="🔍 目标检测...")
    yolo_result = run_yolo()
    
    progress(0.9, desc="🎯 共识计算...")
    consensus_result = run_consensus()
    
    progress(1.0, desc="✅ 全部完成")
    
    # 生成摘要
    summary = "=== 分析摘要 ===\n\n"
    if STATE.visual_output:
        summary += f"📹 镜头: {STATE.visual_output.camera_angle}\n"
        summary += f"🎨 色彩: {STATE.visual_output.hue_family}\n"
        summary += f"✂️ 剪辑: {STATE.visual_output.cuts} 次\n"
    if STATE.audio_output:
        summary += f"🎵 BPM: {STATE.audio_output.tempo_bpm:.1f}\n"
        summary += f"🎸 BGM: {STATE.audio_output.bgm_style}\n"
    if STATE.asr_output:
        summary += f"🎤 语速: {STATE.asr_output.words_per_minute:.1f} wpm\n"
    if STATE.yolo_output:
        summary += f"🔍 物体: {STATE.yolo_output.detection.get('unique_objects', 0)} 种\n"
    
    return visual_result, contact, audio_result, asr_result, yolo_result, consensus_result, summary


def gen_report(progress=gr.Progress()):
    """生成报告"""
    if STATE.video_path is None:
        return "❌ 请先上传视频并运行分析", None, None
    
    if STATE.visual_output is None and STATE.audio_output is None:
        return "❌ 请先运行分析", None, None
    
    progress(0.2, desc="📄 生成 Word 报告...")
    
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
    
    progress(0.7, desc="📕 转换 PDF...")
    STATE.pdf_path = convert_docx_to_pdf(STATE.report_path)
    
    progress(1.0, desc="✅ 完成")
    
    status = f"✅ 报告已生成\n📄 {report_path.name}"
    if STATE.pdf_path:
        status += f"\n📕 {Path(STATE.pdf_path).name}"
    else:
        status += "\n⚠️ PDF 转换需要 libreoffice"
    
    # 返回文件路径供下载
    return status, STATE.report_path, STATE.pdf_path


def export_json():
    """导出 JSON"""
    if STATE.video_path is None:
        return "❌ 请先运行分析", None
    
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
    
    return f"✅ 已导出: {json_path.name}", str(json_path)


# =============================================================================
# Gradio 界面
# =============================================================================
def create_ui():
    with gr.Blocks(
        title="Video Style Analysis",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
    ) as demo:
        
        gr.Markdown("""
# 🎬 视频风格分析系统
**SOTA 2025/2026** | CLIP (场景) | CLAP (音频) | HuBERT (情感) | Whisper (ASR) | YOLO11 (检测)
        """)
        
        with gr.Row():
            # ========== 左侧: 上传和设置 ==========
            with gr.Column(scale=1):
                gr.Markdown("### 📤 上传视频")
                video_input = gr.Video(label="选择视频", height=250)
                upload_status = gr.Textbox(label="状态", lines=4, interactive=False)
                
                gr.Markdown("### ⚙️ 设置")
                language_select = gr.Dropdown(
                    choices=[("English", "en"), ("中文", "zh"), ("日本語", "ja"), ("Auto", "auto")],
                    value="en",
                    label="ASR 语言"
                )
                
                gr.Markdown("### 🎵 音频")
                audio_player = gr.Audio(label="提取的音频", type="filepath")
                
                gr.Markdown("### 🖼️ 关键帧")
                frame_gallery = gr.Gallery(label="关键帧", columns=3, height=200, object_fit="contain")
            
            # ========== 中间: 分析结果 ==========
            with gr.Column(scale=2):
                gr.Markdown("### 🚀 分析控制")
                with gr.Row():
                    run_all_btn = gr.Button("🎯 一键分析全部", variant="primary", size="lg")
                
                with gr.Row():
                    run_visual_btn = gr.Button("📹 视觉")
                    run_audio_btn = gr.Button("🎵 音频")
                    run_asr_btn = gr.Button("🎤 语音")
                    run_yolo_btn = gr.Button("🔍 检测")
                    run_consensus_btn = gr.Button("🎯 共识")
                
                with gr.Tabs():
                    with gr.Tab("📹 视觉"):
                        visual_result = gr.Markdown("*请先上传视频*")
                        contact_img = gr.Image(label="Contact Sheet", height=200)
                    
                    with gr.Tab("🎵 音频"):
                        audio_result = gr.Markdown("*请先上传视频*")
                    
                    with gr.Tab("🎤 语音"):
                        asr_result = gr.Markdown("*请先上传视频*")
                    
                    with gr.Tab("🔍 检测"):
                        yolo_result = gr.Markdown("*请先上传视频*")
                    
                    with gr.Tab("🎯 共识"):
                        consensus_result = gr.Markdown("*请先运行分析*")
            
            # ========== 右侧: 报告和导出 ==========
            with gr.Column(scale=1):
                gr.Markdown("### 📊 报告与导出")
                
                with gr.Row():
                    gen_report_btn = gr.Button("📄 生成报告", variant="secondary")
                    export_json_btn = gr.Button("💾 导出 JSON")
                
                report_status = gr.Textbox(label="报告状态", lines=3, interactive=False)
                
                gr.Markdown("### 📥 下载")
                report_file = gr.File(label="Word 报告 (.docx)")
                pdf_file = gr.File(label="PDF 报告 (.pdf)")
                json_file = gr.File(label="JSON 数据")
                
                json_status = gr.Textbox(label="JSON 状态", lines=2, interactive=False)
                
                gr.Markdown("### 📋 摘要")
                summary_box = gr.Textbox(label="分析摘要", lines=12, interactive=False)
        
        gr.Markdown("---\n**Video Style Analysis** | SOTA 2025/2026")
        
        # ========== 事件绑定 ==========
        video_input.change(fn=upload_video, inputs=[video_input],
                          outputs=[upload_status, audio_player, frame_gallery])
        
        run_visual_btn.click(fn=run_visual, outputs=[visual_result, contact_img])
        run_audio_btn.click(fn=run_audio, outputs=[audio_result])
        run_asr_btn.click(fn=run_asr, inputs=[language_select], outputs=[asr_result])
        run_yolo_btn.click(fn=run_yolo, outputs=[yolo_result])
        run_consensus_btn.click(fn=run_consensus, outputs=[consensus_result])
        
        run_all_btn.click(fn=run_all, inputs=[language_select],
                         outputs=[visual_result, contact_img, audio_result, asr_result, 
                                  yolo_result, consensus_result, summary_box])
        
        gen_report_btn.click(fn=gen_report, outputs=[report_status, report_file, pdf_file])
        export_json_btn.click(fn=export_json, outputs=[json_status, json_file])
    
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Video Style Analysis Web UI")
    parser.add_argument("--port", type=int, default=8088, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )
