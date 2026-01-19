#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio metrics extraction module (SOTA PyTorch + HuggingFace 版本)

使用最先进的模型:
- CLAP (Contrastive Language-Audio Pre-training): 音频分类、BGM风格、情绪
- librosa: 基础音频分析 (BPM, 节拍, 能量, 频谱)
- BEATs: 音频事件检测 (备选)

CLAP 优势:
- 零样本音频分类能力
- 音频-文本对比学习
- 更强的泛化能力
- 支持自定义音频描述

模型: laion/larger_clap_music_and_speech (HuggingFace)
备选: microsoft/BEATs (音频事件检测)
"""

import os
import numpy as np
from loguru import logger
from pathlib import Path
from typing import List, Optional, Dict, Any

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

# 缓存目录
CACHE_DIR = Path.home() / ".cache" / "video_style_pipeline"

# ============================================================================
# CLAP 模型配置
# ============================================================================
CLAP_MODEL = "laion/larger_clap_music_and_speech"
# 备选模型:
# "laion/clap-htsat-unfused"  # 更快但精度略低
# "microsoft/BEATs-iter3-AS20K"  # 音频事件检测

# 全局模型缓存
_CLAP_MODEL = None
_CLAP_PROCESSOR = None

# ============================================================================
# 预定义的音频描述 (用于零样本分类)
# ============================================================================

# BGM 风格描述
BGM_STYLE_PROMPTS = [
    "electronic music with synthesizers and beats",
    "pop music with catchy melody",
    "rock music with electric guitars",
    "classical orchestral music",
    "jazz music with saxophone and piano",
    "ambient relaxing background music",
    "hip hop music with rap and beats",
    "folk acoustic guitar music",
    "cinematic epic trailer music",
    "lo-fi chill beats",
    "upbeat happy commercial music",
    "corporate background music",
]

# 情绪描述
MOOD_PROMPTS = [
    "happy upbeat energetic music",
    "sad melancholic emotional music",
    "calm peaceful relaxing music",
    "intense dramatic exciting music",
    "romantic soft tender music",
    "mysterious suspenseful music",
    "playful fun cheerful music",
    "aggressive powerful heavy music",
    "nostalgic sentimental music",
    "neutral background music",
]

# 乐器检测描述
INSTRUMENT_PROMPTS = [
    "piano playing",
    "guitar playing acoustic or electric",
    "drums and percussion",
    "violin or strings",
    "synthesizer electronic sounds",
    "bass guitar or bass line",
    "vocals singing",
    "trumpet or brass instruments",
    "flute or woodwind instruments",
]

# 简化标签映射
STYLE_SIMPLIFY = {
    "electronic music with synthesizers and beats": "Electronic",
    "pop music with catchy melody": "Pop",
    "rock music with electric guitars": "Rock",
    "classical orchestral music": "Classical",
    "jazz music with saxophone and piano": "Jazz",
    "ambient relaxing background music": "Ambient",
    "hip hop music with rap and beats": "Hip-Hop",
    "folk acoustic guitar music": "Folk",
    "cinematic epic trailer music": "Cinematic",
    "lo-fi chill beats": "Lo-Fi",
    "upbeat happy commercial music": "Commercial/Upbeat",
    "corporate background music": "Corporate",
}

MOOD_SIMPLIFY = {
    "happy upbeat energetic music": "Happy/Energetic",
    "sad melancholic emotional music": "Sad/Melancholic",
    "calm peaceful relaxing music": "Calm/Relaxing",
    "intense dramatic exciting music": "Intense/Dramatic",
    "romantic soft tender music": "Romantic",
    "mysterious suspenseful music": "Mysterious",
    "playful fun cheerful music": "Playful",
    "aggressive powerful heavy music": "Aggressive",
    "nostalgic sentimental music": "Nostalgic",
    "neutral background music": "Neutral",
}


def _load_clap_model():
    """Lazy-load CLAP model from HuggingFace."""
    global _CLAP_MODEL, _CLAP_PROCESSOR
    
    if _CLAP_MODEL is not None:
        return _CLAP_MODEL, _CLAP_PROCESSOR
    
    try:
        from transformers import ClapProcessor, ClapModel
        
        logger.info(f"Loading CLAP model: {CLAP_MODEL}")
        
        _CLAP_PROCESSOR = ClapProcessor.from_pretrained(
            CLAP_MODEL,
            cache_dir=str(CACHE_DIR / "hf_models")
        )
        _CLAP_MODEL = ClapModel.from_pretrained(
            CLAP_MODEL,
            cache_dir=str(CACHE_DIR / "hf_models")
        )
        
        # 使用 GPU 如果可用
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _CLAP_MODEL = _CLAP_MODEL.to(device)
        _CLAP_MODEL.eval()
        
        logger.info(f"CLAP model loaded successfully on {device}")
        return _CLAP_MODEL, _CLAP_PROCESSOR
        
    except ImportError:
        logger.warning("transformers not available for CLAP audio classification.")
        return None, None
    except Exception as e:
        logger.warning(f"Failed to load CLAP model: {e}")
        return None, None


def classify_audio_with_clap(
    y: np.ndarray,
    sr: int,
    prompts: List[str],
    label_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    使用 CLAP 进行零样本音频分类
    
    Args:
        y: 音频波形
        sr: 采样率
        prompts: 文本提示列表
        label_mapping: 可选的标签简化映射
        
    Returns:
        dict: 分类结果
    """
    model, processor = _load_clap_model()
    
    if model is None:
        return {
            "best_match": "Unknown",
            "confidence": 0.0,
            "all_scores": {},
            "method": "N/A"
        }
    
    device = next(model.parameters()).device
    
    try:
        # 重采样到 48kHz (CLAP 期望的采样率)
        if sr != 48000:
            y = librosa.resample(y, orig_sr=sr, target_sr=48000)
            sr = 48000
        
        # 截取到合适长度 (10秒)
        max_length = sr * 10
        if len(y) > max_length:
            # 取中间部分
            start = (len(y) - max_length) // 2
            y = y[start:start + max_length]
        
        # 处理输入
        inputs = processor(
            text=prompts,
            audios=y,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_audio = outputs.logits_per_audio
            probs = logits_per_audio.softmax(dim=-1)[0]
        
        # 获取结果
        best_idx = torch.argmax(probs).item()
        best_prompt = prompts[best_idx]
        best_label = label_mapping.get(best_prompt, best_prompt) if label_mapping else best_prompt
        
        all_scores = {}
        for i, prompt in enumerate(prompts):
            label = label_mapping.get(prompt, prompt) if label_mapping else prompt
            all_scores[label] = float(probs[i].item())
        
        # 排序并获取 top-k
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "best_match": best_label,
            "confidence": float(probs[best_idx].item()),
            "all_scores": dict(sorted_scores),
            "top_3": sorted_scores[:3],
            "method": "CLAP"
        }
        
    except Exception as e:
        logger.warning(f"CLAP classification failed: {e}")
        return {
            "best_match": "Unknown",
            "confidence": 0.0,
            "all_scores": {},
            "method": "failed"
        }


def extract_audio_metrics(audio_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive audio metrics from an audio file.
    
    使用 CLAP 进行高级分类，librosa 进行基础分析。
    
    Args:
        audio_path: Path to audio file
        
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
        
        # ====== 高级分析 (CLAP) ======
        
        # BGM 风格分类
        style_result = classify_audio_with_clap(y, sr, BGM_STYLE_PROMPTS, STYLE_SIMPLIFY)
        
        # 情绪分类
        mood_result = classify_audio_with_clap(y, sr, MOOD_PROMPTS, MOOD_SIMPLIFY)
        
        # 乐器检测
        instrument_result = classify_audio_with_clap(y, sr, INSTRUMENT_PROMPTS)
        detected_instruments = [
            inst for inst, score in instrument_result.get("all_scores", {}).items()
            if score > 0.15  # 阈值
        ]
        
        result = {
            # 基础节奏分析
            "tempo_bpm": float(tempo),
            "beat_times": beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times),
            "num_beats": len(beat_times),
            "percussive_ratio": percussive_ratio,
            
            # 频谱特征
            "spectral_centroid": mean_centroid,
            "spectral_flatness": mean_flatness,
            "zero_crossing_rate": mean_zcr,
            "spectral_rolloff": mean_rolloff,
            "speech_ratio": speech_ratio,
            
            # 能量分析
            "mean_energy": mean_energy,
            "energy_variance": energy_variance,
            
            # 调式
            "key_signature": tonal_key,
            
            # CLAP 分类结果 (BGM 风格)
            "bgm_style": style_result.get("best_match", "Unknown"),
            "bgm_style_confidence": style_result.get("confidence", 0.0),
            "bgm_style_detail": {
                "all_scores": style_result.get("all_scores", {}),
                "top_3": style_result.get("top_3", []),
                "method": style_result.get("method", "N/A")
            },
            
            # CLAP 分类结果 (情绪)
            "mood": mood_result.get("best_match", "Unknown"),
            "mood_confidence": mood_result.get("confidence", 0.0),
            "mood_detail": {
                "all_scores": mood_result.get("all_scores", {}),
                "top_3": mood_result.get("top_3", []),
                "method": mood_result.get("method", "N/A")
            },
            
            # 乐器检测
            "instruments": {
                "detected_instruments": detected_instruments,
                "instrument_scores": instrument_result.get("all_scores", {}),
                "method": instrument_result.get("method", "N/A")
            },
            
            # 兼容旧接口
            "mood_tags": mood_result.get("top_3", []),
        }
        
        logger.info(f"Audio metrics extracted successfully from {audio_path}")
        logger.info(f"  → BGM: {result['bgm_style']} ({result['bgm_style_confidence']:.1%}) | "
                   f"Mood: {result['mood']} ({result['mood_confidence']:.1%}) | "
                   f"BPM: {result['tempo_bpm']:.1f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract audio metrics from {audio_path}: {e}")
        raise


def calculate_beat_alignment(
    video_duration: float,
    num_cuts: int,
    beat_times: List[float],
    tolerance: float = 0.15
) -> float:
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


def analyze_energy_dynamics(audio_path: str, window_sec: float = 5.0) -> Dict[str, Any]:
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


def get_audio_embedding(audio_path: str) -> np.ndarray:
    """
    Get CLAP audio embedding for an audio file.
    
    Useful for similarity search or clustering.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        np.ndarray: Audio embedding vector
    """
    model, processor = _load_clap_model()
    
    if model is None:
        raise RuntimeError("CLAP model not available")
    
    device = next(model.parameters()).device
    
    y, sr = librosa.load(audio_path, sr=48000, mono=True)
    
    # 截取到合适长度
    max_length = sr * 10
    if len(y) > max_length:
        start = (len(y) - max_length) // 2
        y = y[start:start + max_length]
    
    inputs = processor(audios=y, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        audio_features = model.get_audio_features(**inputs)
    
    embedding = audio_features.cpu().numpy().squeeze()
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding
