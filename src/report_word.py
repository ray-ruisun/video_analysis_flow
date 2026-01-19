#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word report generation module.
Creates formatted .docx reports with metrics, analysis, and visualizations.
"""

import os
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from utils import format_value


def create_report_header(doc, title="Video Style Analysis Report"):
    """Add formatted header to document."""
    # Title
    title_para = doc.add_paragraph()
    title_run = title_para.add_run(title)
    title_run.bold = True
    title_run.font.size = Pt(18)
    title_run.font.color.rgb = RGBColor(31, 78, 120)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Subtitle
    subtitle = doc.add_paragraph()
    subtitle_run = subtitle.add_run("Signal-derived Stylistic Pattern Analysis")
    subtitle_run.font.size = Pt(12)
    subtitle_run.font.color.rgb = RGBColor(100, 100, 100)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Compliance notice
    notice = doc.add_paragraph()
    notice_run = notice.add_run(
        "Compliance: Human-created content analysis. "
        "No portrait recognition. No external copyrighted assets."
    )
    notice_run.font.size = Pt(9)
    notice_run.italic = True
    notice.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph()  # Spacing


def add_section_header(doc, text, level=1):
    """Add formatted section header."""
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.bold = True
    
    if level == 1:
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(31, 78, 120)
    else:
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(80, 80, 80)
    
    return para


def add_video_source_info(doc, video_metrics):
    """Add source video information section."""
    add_section_header(doc, "Source Videos")
    
    for i, metrics in enumerate(video_metrics, 1):
        video_name = Path(metrics.get("path", f"Video {i}")).name
        duration = metrics.get("visual", {}).get("duration", 0)
        fps = metrics.get("visual", {}).get("fps", 0)
        
        doc.add_paragraph(
            f"• {video_name} | Duration: {format_value(duration, '.1f')} s | "
            f"FPS: {format_value(fps, '.1f')}"
        )
    
    doc.add_paragraph()


def add_consensus_section(doc, consensus):
    """Add cross-video consensus section."""
    add_section_header(doc, "I. Cross-Video Common Elements", level=1)
    doc.add_paragraph("Majority consensus across all analyzed videos:")
    
    # Camera and composition
    doc.add_paragraph(
        f"• Camera language: {consensus.get('camera_angle', 'N/A')}"
    )
    
    # Color and lighting
    color_text = (
        f"• Color & lighting: Hue {consensus.get('hue_family', 'N/A')}, "
        f"Saturation {consensus.get('saturation', 'N/A')}, "
        f"Brightness {consensus.get('brightness', 'N/A')}, "
        f"Contrast {consensus.get('contrast', 'N/A')}, "
        f"CCT ≈ {format_value(consensus.get('cct'), '.0f')} K"
    )
    doc.add_paragraph(color_text)
    
    # Pacing and editing
    pacing_text = (
        f"• Pacing & editing: {format_value(consensus.get('cuts_per_minute'), '.2f')} cuts/min, "
        f"Avg shot length ≈ {format_value(consensus.get('avg_shot_length'), '.2f')} s, "
        f"{consensus.get('transition_type', 'N/A')}, "
        f"Beat alignment ≈ {format_value(consensus.get('beat_alignment'), '.2f')}"
    )
    doc.add_paragraph(pacing_text)
    
    # Audio and BGM
    audio_text = (
        f"• BGM preference: {consensus.get('bgm_style', 'N/A')}, "
        f"Tempo ≈ {format_value(consensus.get('tempo_bpm'), '.1f')} BPM, "
        f"Percussive energy ≈ {format_value(consensus.get('percussive_ratio'), '.2f')}, "
        f"Speech ratio ≈ {format_value(consensus.get('speech_ratio'), '.2f')}"
    )
    doc.add_paragraph(audio_text)
    
    # Environment
    env_text = (
        f"• Background environment: Countertop {consensus.get('countertop_color', 'N/A')} / "
        f"{consensus.get('countertop_texture', 'N/A')}"
    )
    doc.add_paragraph(env_text)
    
    # YOLO results if available
    if consensus.get('yolo_available'):
        yolo_text = (
            f"• Detected objects: {consensus.get('yolo_environment', 'N/A')}, "
            f"{consensus.get('yolo_style', 'N/A')}"
        )
        doc.add_paragraph(yolo_text)
    
    doc.add_paragraph()


def add_per_video_section(doc, video_metrics, show_screenshots=True):
    """Add detailed per-video metrics section."""
    add_section_header(doc, "II. Per-Video Detailed Metrics", level=1)
    
    for i, metrics in enumerate(video_metrics, 1):
        video_name = Path(metrics.get("path", f"Video {i}")).name
        
        # Video header
        video_header = doc.add_paragraph()
        video_header.add_run(f"{video_name}").bold = True
        
        # Visual metrics
        vis = metrics.get("visual", {})
        visual_text = (
            f"   Visual: Camera {vis.get('camera_angle', 'N/A')}, "
            f"Hue {vis.get('hue_family', 'N/A')}, "
            f"Saturation {vis.get('saturation_band', 'N/A')}, "
            f"Brightness {vis.get('brightness_band', 'N/A')}, "
            f"Contrast {vis.get('contrast', 'N/A')}, "
            f"{vis.get('cuts', 0)} cuts ({vis.get('transition_type', 'N/A')}), "
            f"CCT ≈ {format_value(vis.get('cct_mean'), '.0f')} K, "
            f"Countertop {vis.get('countertop_color', 'N/A')} / "
            f"{vis.get('countertop_texture', 'N/A')}"
        )
        doc.add_paragraph(visual_text)
        
        # Audio metrics
        aud = metrics.get("audio", {})
        if aud.get("available"):
            audio_text = (
                f"   Audio: BGM {aud.get('bgm_style', 'N/A')}, "
                f"Tempo ≈ {format_value(aud.get('tempo_bpm'), '.1f')} BPM, "
                f"Percussive ≈ {format_value(aud.get('percussive_ratio'), '.2f')}, "
                f"Speech ≈ {format_value(aud.get('speech_ratio'), '.2f')}"
            )
            doc.add_paragraph(audio_text)
        else:
            doc.add_paragraph("   Audio: Not analyzed (dependencies missing or audio unavailable)")
        
        # ASR metrics
        asr = metrics.get("asr", {})
        if asr.get("available"):
            catchphrases = asr.get("catchphrases", [])
            catchphrase_str = ", ".join(catchphrases[:5]) if catchphrases else "None detected"
            
            asr_text = (
                f"   Narration (ASR): Rate ≈ {format_value(asr.get('words_per_second'), '.2f')} words/s, "
                f"Pace: {asr.get('pace', 'N/A')}, "
                f"Catchphrases: {catchphrase_str}"
            )
            doc.add_paragraph(asr_text)
        
        # YOLO metrics
        yolo = metrics.get("yolo", {})
        if yolo.get("available"):
            detection = yolo.get("detection", {})
            environment = yolo.get("environment", {})
            
            top_objects = detection.get("top_objects", [])[:5]
            objects_str = ", ".join([f"{obj} ({count})" for obj, count in top_objects])
            
            yolo_text = (
                f"   Objects detected (YOLO): {environment.get('environment_type', 'N/A')}, "
                f"{environment.get('cooking_style', 'N/A')} - "
                f"Top objects: {objects_str}"
            )
            doc.add_paragraph(yolo_text)
        
        # Add screenshot if available
        if show_screenshots:
            contact_sheet = vis.get("contact_sheet")
            if contact_sheet and os.path.exists(contact_sheet):
                try:
                    doc.add_paragraph()
                    pic_para = doc.add_picture(contact_sheet, width=Inches(6.5))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                except Exception as e:
                    doc.add_paragraph(f"   (Failed to insert contact sheet: {e})")
        
        doc.add_paragraph()  # Spacing between videos


def add_notes_section(doc):
    """Add notes and upgrade suggestions section."""
    add_section_header(doc, "III. Technical Notes & Upgrade Paths", level=1)
    
    notes = [
        ("Shot Boundary Detection", 
         "Built-in histogram-based method. For higher accuracy, consider PySceneDetect or TransNet V2."),
        
        ("Music Analysis", 
         "Uses librosa for tempo/beat. For mood, key, and instrumentation, integrate Essentia Music Extractor."),
        
        ("Object Detection", 
         "YOLOv8 detects common objects. For materials (wood/metal/ceramic), add MINC classification."),
        
        ("White Balance", 
         "CCT estimated from RGB. OpenCV's xphoto module (SimpleWB/GrayworldWB) provides more robust estimation."),
        
        ("Source Separation", 
         "For cleaner audio analysis, separate vocals/drums/bass with Spleeter or Demucs before feature extraction."),
        
        ("Speech Analysis", 
         "Whisper provides transcription. Add emotion/tone analysis with prosody features or specialized models.")
    ]
    
    for title, description in notes:
        para = doc.add_paragraph()
        para.add_run(f"• {title}: ").bold = True
        para.add_run(description)
    
    doc.add_paragraph()


def add_references_section(doc):
    """Add references and citations."""
    add_section_header(doc, "References", level=1)
    
    references = [
        "OpenCV: Computer Vision Library - https://opencv.org/",
        "librosa: Audio Analysis - https://librosa.org/",
        "YOLOv8 (Ultralytics): Object Detection - https://docs.ultralytics.com/",
        "Whisper: Speech Recognition - https://github.com/openai/whisper",
        "PySceneDetect: Shot Boundary Detection - https://www.scenedetect.com/",
        "HuggingFace Transformers: Audio Classification - https://huggingface.co/",
        "python-docx: Word Document Generation - https://python-docx.readthedocs.io/"
    ]
    
    for ref in references:
        doc.add_paragraph(ref, style='List Number')


def generate_word_report(video_metrics, consensus, output_path, 
                        show_screenshots=True, show_references=True):
    """
    Generate complete Word document report.
    
    Args:
        video_metrics: List of per-video metric dictionaries
        consensus: Cross-video consensus dictionary
        output_path: Path to save .docx file
        show_screenshots: Whether to include contact sheets
        show_references: Whether to include references section
        
    Returns:
        str: Path to generated report
    """
    # Create document
    doc = Document()
    
    # Set default font
    style = doc.styles['Normal']
    style.font.name = 'Arial'
    style.font.size = Pt(11)
    
    # Build report sections
    create_report_header(doc)
    add_video_source_info(doc, video_metrics)
    add_consensus_section(doc, consensus)
    add_per_video_section(doc, video_metrics, show_screenshots)
    add_notes_section(doc)
    
    if show_references:
        add_references_section(doc)
    
    # Save document
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    
    return str(output_path)

