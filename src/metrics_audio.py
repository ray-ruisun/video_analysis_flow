#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio metrics extraction module (PyTorch + HuggingFace 版本)

使用纯 PyTorch 生态系统，避免 TensorFlow 依赖:
- librosa: 基础音频分析 (BPM, 节拍, 能量, 频谱)
- transformers: HuggingFace 模型 (音乐分类, 情绪分析)

分析内容:
- 节奏分析: BPM, 节拍时间点, 打击乐比例
- 能量分析: RMS 能量, 能量方差
- 频谱分析: 质心, 平坦度, 过零率, rolloff
- BGM 风格分类: 使用 HuggingFace 音频分类模型
- 情绪分析: 使用 HuggingFace 音频情绪模型
"""

import os
import numpy as np
from loguru import logger
from pathlib import Path

# Required dependencies
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError as e:
    logger.error(f"librosa/soundfile not installed: {e}")
    logger.error("Install with: pip install librosa soundfile")
    raise ImportError("librosa and soundfile are required.")

import torch

# HuggingFace transformers for audio classification (optional)
TRANSFORMERS_AUDIO_AVAILABLE = False
_AUDIO_CLASSIFIER = None
_AUDIO_FEATURE_EXTRACTOR = None

try:
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    TRANSFORMERS_AUDIO_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available for audio classification.")
    logger.warning("Install with: pip install transformers")

# 缓存目录
CACHE_DIR = Path.home() / ".cache" / "video_style_pipeline"

# HuggingFace 模型配置
# 音乐分类模型 (可选择不同模型)
MUSIC_CLASSIFIER_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"  # AudioSet 分类
# 备选: "facebook/wav2vec2-base" + 自定义分类头

# 情绪相关的 AudioSet 标签映射
MOOD_LABEL_MAPPING = {
    "happy": ["Happy music", "Exciting music", "Funny music"],
    "sad": ["Sad music", "Tender music"],
    "energetic": ["Electronic music", "Techno", "Dance music", "Drum and bass"],
    "calm": ["Ambient music", "New-age music", "Meditation"],
    "aggressive": ["Heavy metal", "Punk rock", "Grunge"],
}

# BGM 风格标签映射
GENRE_LABEL_MAPPING = {
    "Electronic": ["Electronic music", "Techno", "House music", "Trance music", "Drum and bass", "Dubstep"],
    "Pop": ["Pop music", "Dance music", "Disco"],
    "Rock": ["Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock"],
    "Classical": ["Classical music", "Orchestra", "Piano", "Violin"],
    "Jazz": ["Jazz", "Blues", "Soul music", "Funk"],
    "Ambient": ["Ambient music", "New-age music", "Drone"],
    "Hip-Hop": ["Hip hop music", "Rap", "Beatboxing"],
    "Folk": ["Folk music", "Country", "Bluegrass"],
}


def _load_audio_classifier():
    """Lazy-load HuggingFace audio classification model."""
    global _AUDIO_CLASSIFIER, _AUDIO_FEATURE_EXTRACTOR
    
    if not TRANSFORMERS_AUDIO_AVAILABLE:
        return None, None
    
    if _AUDIO_CLASSIFIER is not None:
        return _AUDIO_CLASSIFIER, _AUDIO_FEATURE_EXTRACTOR
    
    try:
        logger.info(f"Loading HuggingFace audio classifier: {MUSIC_CLASSIFIER_MODEL}")
        _AUDIO_FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(
            MUSIC_CLASSIFIER_MODEL,
            cache_dir=str(CACHE_DIR / "hf_models")
        )
        _AUDIO_CLASSIFIER = AutoModelForAudioClassification.from_pretrained(
            MUSIC_CLASSIFIER_MODEL,
            cache_dir=str(CACHE_DIR / "hf_models")
        )
        _AUDIO_CLASSIFIER.eval()
        logger.info("Audio classifier loaded successfully")
        return _AUDIO_CLASSIFIER, _AUDIO_FEATURE_EXTRACTOR
    except Exception as e:
        logger.warning(f"Failed to load audio classifier: {e}")
        return None, None


def classify_audio_with_hf(y: np.ndarray, sr: int) -> dict:
    """
    使用 HuggingFace 模型分类音频
    
    Args:
        y: 音频波形
        sr: 采样率
        
    Returns:
        dict: 分类结果 (genre, mood, top_labels)
    """
    model, feature_extractor = _load_audio_classifier()
    
    if model is None:
        return {
            "genre": "Unknown",
            "mood": "Unknown",
            "top_labels": [],
            "method": "N/A"
        }
    
    try:
        # 重采样到模型期望的采样率 (通常 16kHz)
        target_sr = feature_extractor.sampling_rate
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # 截取或填充到合适长度 (10秒)
        max_length = target_sr * 10
        if len(y) > max_length:
            # 取中间部分
            start = (len(y) - max_length) // 2
            y = y[start:start + max_length]
        elif len(y) < max_length:
            y = np.pad(y, (0, max_length - len(y)))
        
        # 提取特征
        inputs = feature_extractor(
            y,
            sampling_rate=target_sr,
            return_tensors="pt"
        )
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # 获取标签
        id2label = model.config.id2label
        top_k = 10
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
        
        top_labels = []
        for prob, idx in zip(top_probs, top_indices):
            label = id2label.get(idx.item(), f"label_{idx.item()}")
            top_labels.append({"label": label, "probability": float(prob)})
        
        # 映射到 genre 和 mood
        genre = _map_to_genre(top_labels)
        mood = _map_to_mood(top_labels)
        
        return {
            "genre": genre,
            "mood": mood,
            "top_labels": top_labels,
            "method": "HuggingFace AST"
        }
        
    except Exception as e:
        logger.warning(f"HuggingFace audio classification failed: {e}")
        return {
            "genre": "Unknown",
            "mood": "Unknown",
            "top_labels": [],
            "method": "failed"
        }


def _map_to_genre(top_labels: list) -> str:
    """将 AudioSet 标签映射到音乐风格"""
    label_names = [item["label"] for item in top_labels[:5]]
    
    for genre, keywords in GENRE_LABEL_MAPPING.items():
        for keyword in keywords:
            if any(keyword.lower() in label.lower() for label in label_names):
                return genre
    
    # 如果没有匹配，返回最高概率的标签
    if top_labels:
        return top_labels[0]["label"]
    return "Unknown"


def _map_to_mood(top_labels: list) -> str:
    """将 AudioSet 标签映射到情绪"""
    label_names = [item["label"] for item in top_labels[:5]]
    
    for mood, keywords in MOOD_LABEL_MAPPING.items():
        for keyword in keywords:
            if any(keyword.lower() in label.lower() for label in label_names):
                return mood.capitalize()
    
    return "Neutral"


def extract_audio_metrics(audio_path):
    """
    Extract comprehensive audio metrics from a wav file.
    
    使用 librosa 进行基础分析，HuggingFace 进行高级分类。
    
    Args:
        audio_path: Path to audio file (preferably 22.05kHz mono wav)
        
    Returns:
        dict: Audio metrics including tempo, beats, energy, genre, mood, etc.
    """
    if not audio_path:
        logger.error("Audio path is None or empty")
        raise ValueError("Audio path is None or empty")
    
    audio_path = str(audio_path)
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Load audio (force mono, 22.05kHz for librosa analysis)
        logger.debug(f"Loading audio from {audio_path}")
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        if len(y) == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            raise ValueError(f"Audio file is empty: {audio_path}")
        
        # Normalize
        y = y / (np.max(np.abs(y)) + 1e-6)
        
        # ====== 基础分析 (librosa) ======
        
        # Harmonic-Percussive Source Separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Percussive energy ratio
        percussive_ratio = float(
            np.sum(np.abs(y_percussive)) / (np.sum(np.abs(y)) + 1e-6)
        )
        
        # Tempo and beat detection
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_centroid = float(np.mean(spectral_centroid))
        
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        mean_flatness = float(np.mean(spectral_flatness))
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        mean_zcr = float(np.mean(zero_crossing_rate))
        
        # Speech presence proxy
        speech_ratio = float(np.clip(mean_flatness * 1.8 + mean_zcr * 0.8, 0, 1))
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        mean_energy = float(np.mean(rms))
        energy_variance = float(np.var(rms))
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        mean_rolloff = float(np.mean(rolloff))
        
        # Key signature estimation using chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_idx = int(np.argmax(chroma_mean))
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        tonal_key = f"{key_names[key_idx]} (estimated)"
        
        # ====== 高级分析 (HuggingFace) ======
        hf_result = classify_audio_with_hf(y, sr)
        
        # 如果 HuggingFace 分析失败，使用 librosa 估算情绪
        genre = hf_result.get("genre", "Unknown")
        mood = hf_result.get("mood", "Unknown")
        
        if mood == "Unknown":
            mood = classify_bgm_mood(float(tempo), percussive_ratio, mean_centroid, energy_variance)
        
        result = {
            "tempo_bpm": float(tempo),
            "beat_times": beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times),
            "num_beats": len(beat_times),
            "percussive_ratio": percussive_ratio,
            "spectral_centroid": mean_centroid,
            "spectral_flatness": mean_flatness,
            "zero_crossing_rate": mean_zcr,
            "speech_ratio": speech_ratio,
            "bgm_style": genre,
            "mood": mood,
            "mood_tags": hf_result.get("top_labels", []),
            "instruments": {
                "detected_instruments": [],
                "method": hf_result.get("method", "librosa")
            },
            "mean_energy": mean_energy,
            "energy_variance": energy_variance,
            "spectral_rolloff": mean_rolloff,
            "key_signature": tonal_key
        }
        
        logger.info(f"Audio metrics extracted successfully from {audio_path}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract audio metrics from {audio_path}: {e}")
        raise


def calculate_beat_alignment(video_duration, num_cuts, beat_times, tolerance=0.15):
    """
    Calculate how well cuts align with musical beats.
    
    Args:
        video_duration: Total video duration in seconds
        num_cuts: Number of detected cuts
        beat_times: List of beat timestamps in seconds
        tolerance: Alignment tolerance in seconds
        
    Returns:
        float: Proportion of cuts aligned with beats (0-1)
    """
    if num_cuts <= 0:
        logger.error(f"Invalid number of cuts: {num_cuts}")
        raise ValueError(f"Number of cuts must be positive, got {num_cuts}")
    
    if not beat_times:
        logger.error("No beat times provided")
        raise ValueError("Beat times list is empty")
    
    if video_duration <= 0:
        logger.error(f"Invalid video duration: {video_duration}")
        raise ValueError(f"Video duration must be positive, got {video_duration}")
    
    # Approximate cut times
    approx_cut_times = np.linspace(0, video_duration, num=num_cuts + 2)[1:-1]
    
    beat_array = np.array(beat_times)
    aligned_count = 0
    
    for cut_time in approx_cut_times:
        if beat_array.size > 0:
            min_distance = np.min(np.abs(beat_array - cut_time))
            if min_distance <= tolerance:
                aligned_count += 1
    
    alignment_ratio = aligned_count / len(approx_cut_times)
    logger.debug(f"Beat alignment: {alignment_ratio:.2%} of cuts aligned")
    return alignment_ratio


def analyze_energy_dynamics(audio_path, window_sec=5.0):
    """
    Analyze energy dynamics over time.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-6)
        
        hop_length = int(sr * window_sec)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        energy_range = float(np.max(rms) - np.min(rms))
        energy_std = float(np.std(rms))
        
        energy_derivative = np.diff(rms)
        num_peaks = len([x for x in energy_derivative if abs(x) > 0.1])
        
        return {
            "energy_range": energy_range,
            "energy_std": energy_std,
            "num_energy_changes": num_peaks,
            "dynamic_style": "High dynamics" if energy_std > 0.1 else "Steady energy"
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze energy dynamics: {e}")
        raise


def classify_bgm_mood(tempo, percussive_ratio, spectral_centroid, energy_variance):
    """
    Classify BGM mood based on audio features (librosa fallback).
    """
    if tempo < 90:
        mood = "Calm/Relaxed"
    elif tempo < 120:
        mood = "Moderate/Upbeat"
    else:
        mood = "Energetic/Fast-paced"
    
    if percussive_ratio > 0.5:
        mood += " with strong rhythm"
    
    if energy_variance > 0.015:
        mood += ", dynamic"
    else:
        mood += ", consistent"
    
    return mood
