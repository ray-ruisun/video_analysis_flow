#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Style Analysis Pipeline

A modular pipeline for extracting stylistic patterns from video content.
"""

__version__ = "1.0.0"
__author__ = "Video Analysis Research"

from .metrics_visual import extract_visual_metrics
from .metrics_audio import extract_audio_metrics
from .metrics_asr import extract_full_asr_metrics
from .metrics_yolo import extract_full_yolo_metrics
from .report_word import generate_word_report

__all__ = [
    "extract_visual_metrics",
    "extract_audio_metrics",
    "extract_full_asr_metrics",
    "extract_full_yolo_metrics",
    "generate_word_report"
]

