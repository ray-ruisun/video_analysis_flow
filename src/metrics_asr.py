#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Speech Recognition (ASR) module (SOTA PyTorch + HuggingFace 版本)

使用最先进的模型:
- Whisper large-v3-turbo: 最新最强的多语言ASR
- Emotion2Vec / HuBERT: 语音情感识别
- librosa: 韵律分析 (pitch, intensity)

Whisper 升级:
- small → large-v3-turbo (更准确，支持更多语言)
- faster-whisper 实现 (CTranslate2 加速)

情感识别升级:
- wav2vec2 → Emotion2Vec+ / HuBERT-large
"""

import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from loguru import logger

# Required dependencies
try:
    import soundfile as sf
except ImportError as e:
    logger.error(f"soundfile not installed: {e}")
    raise ImportError("soundfile is required. Install with: pip install soundfile")

# ASR implementation
ASR_AVAILABLE = False
ASR_IMPLEMENTATION = None

try:
    from faster_whisper import WhisperModel
    ASR_AVAILABLE = True
    ASR_IMPLEMENTATION = "faster-whisper"
except ImportError:
    try:
        import whisper
        ASR_AVAILABLE = True
        ASR_IMPLEMENTATION = "openai-whisper"
    except ImportError:
        pass

if not ASR_AVAILABLE:
    logger.error("No Whisper implementation found. Install one of:")
    logger.error("  pip install faster-whisper")
    logger.error("  pip install openai-whisper")
    raise ImportError("Whisper is required.")

# Prosody analysis - librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available for prosody analysis.")

# Emotion analysis - HuggingFace transformers
import torch
EMOTION_AVAILABLE = False
_EMOTION_CLASSIFIER = None
_EMOTION_PROCESSOR = None

try:
    from transformers import (
        AutoModelForAudioClassification, 
        AutoFeatureExtractor,
        Wav2Vec2FeatureExtractor,
        HubertForSequenceClassification
    )
    EMOTION_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available for emotion analysis.")
    logger.warning("Install with: pip install transformers torch")

# 缓存目录
CACHE_DIR = Path.home() / ".cache" / "video_style_pipeline"

# ============================================================================
# 模型配置
# ============================================================================

# Whisper 模型 (推荐 large-v3-turbo)
DEFAULT_WHISPER_MODEL = "large-v3-turbo"
# 备选: "large-v3", "medium", "small", "base", "tiny"

# 语音情感识别模型 (SOTA)
# 选项1: superb/hubert-large-superb-er (HuBERT-large on SUPERB)
# 选项2: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
# 选项3: facebook/hubert-large-ls960-ft (需要 fine-tune)
EMOTION_MODEL = "superb/hubert-large-superb-er"
# 备选更强模型:
# "speechbrain/emotion-recognition-wav2vec2-IEMOCAP" (如果安装了 speechbrain)

# 情感标签映射 (根据模型)
EMOTION_LABELS = {
    "superb/hubert-large-superb-er": ["angry", "happy", "neutral", "sad"],
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition": [
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
    ]
}


def _load_emotion_classifier():
    """Lazy-load emotion recognition model from HuggingFace."""
    global _EMOTION_CLASSIFIER, _EMOTION_PROCESSOR
    
    if not EMOTION_AVAILABLE:
        return None, None
    
    if _EMOTION_CLASSIFIER is not None:
        return _EMOTION_CLASSIFIER, _EMOTION_PROCESSOR
    
    try:
        logger.info(f"Loading emotion model: {EMOTION_MODEL}")
        
        _EMOTION_PROCESSOR = AutoFeatureExtractor.from_pretrained(
            EMOTION_MODEL,
            cache_dir=str(CACHE_DIR / "hf_models")
        )
        _EMOTION_CLASSIFIER = AutoModelForAudioClassification.from_pretrained(
            EMOTION_MODEL,
            cache_dir=str(CACHE_DIR / "hf_models")
        )
        
        # 使用 GPU 如果可用
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _EMOTION_CLASSIFIER = _EMOTION_CLASSIFIER.to(device)
        _EMOTION_CLASSIFIER.eval()
        
        logger.info(f"Emotion model loaded successfully on {device}")
        return _EMOTION_CLASSIFIER, _EMOTION_PROCESSOR
        
    except Exception as e:
        logger.warning(f"Failed to load emotion classifier: {e}")
        return None, None


def transcribe_audio(
    audio_path: str,
    language: str = "en",
    model_size: str = DEFAULT_WHISPER_MODEL,
    beam_size: int = 5
) -> Dict[str, Any]:
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_path: Path to audio file
        language: Language code (default: "en", use "auto" for auto-detect)
        model_size: Model size ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
        beam_size: Beam search width for decoding (default: 5)
        
    Returns:
        dict: Transcription results with text, timing, and metadata
    """
    if not audio_path:
        raise ValueError("Audio path is None or empty")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        transcribed_text = ""
        segments_list = []
        detected_language = language
        
        if ASR_IMPLEMENTATION == "faster-whisper":
            logger.debug(f"Using faster-whisper ({model_size}) for transcription: {audio_path}")
            
            # 使用 GPU 如果可用
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            
            # 自动语言检测
            if language == "auto":
                segments, info = model.transcribe(audio_path, vad_filter=True, beam_size=beam_size)
                detected_language = info.language
            else:
                segments, info = model.transcribe(audio_path, language=language, vad_filter=True, beam_size=beam_size)
                detected_language = language
            
            for segment in segments:
                text = segment.text.strip()
                transcribed_text += text + " "
                segments_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": text,
                    "confidence": getattr(segment, 'avg_logprob', None)
                })
        
        else:  # openai-whisper
            logger.debug(f"Using openai-whisper ({model_size}) for transcription: {audio_path}")
            model = whisper.load_model(model_size)
            
            if language == "auto":
                result = model.transcribe(audio_path, beam_size=beam_size)
            else:
                result = model.transcribe(audio_path, language=language, beam_size=beam_size)
            
            transcribed_text = result.get("text", "")
            detected_language = result.get("language", language)
            
            if "segments" in result:
                for seg in result["segments"]:
                    segments_list.append({
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", "").strip(),
                        "confidence": seg.get("avg_logprob")
                    })
        
        if not transcribed_text.strip():
            logger.warning(f"No transcription obtained from {audio_path}")
            return {
                "implementation": ASR_IMPLEMENTATION,
                "model_size": model_size,
                "text": "",
                "segments": [],
                "language": detected_language,
                "word_count": 0
            }
        
        word_count = len(transcribed_text.split())
        logger.info(f"Transcription completed: {word_count} words, language={detected_language}")
        
        return {
            "implementation": ASR_IMPLEMENTATION,
            "model_size": model_size,
            "text": transcribed_text.strip(),
            "segments": segments_list,
            "language": detected_language,
            "word_count": word_count
        }
        
    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}")
        raise


def analyze_speech_rate(
    transcription_result: Dict[str, Any],
    audio_path: str
) -> Dict[str, Any]:
    """Calculate speech rate (words per second)."""
    if not transcription_result or "text" not in transcription_result:
        raise ValueError("Invalid transcription result")
    
    text = transcription_result.get("text", "")
    words = re.findall(r"\b[\w']+\b", text.lower())
    num_words = len(words)
    
    if num_words == 0:
        return {
            "num_words": 0,
            "duration": 0,
            "words_per_second": 0,
            "words_per_minute": 0,
            "pace": "No speech"
        }
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        info = sf.info(audio_path)
        duration = info.duration
    except Exception as e:
        raise ValueError(f"Failed to get audio duration: {e}")
    
    if duration <= 0:
        raise ValueError(f"Invalid audio duration: {duration}")
    
    words_per_second = num_words / duration
    words_per_minute = words_per_second * 60
    
    # 语速分类
    if words_per_second < 1.5:
        pace = "Slow/Deliberate"
    elif words_per_second < 2.5:
        pace = "Moderate"
    elif words_per_second < 3.5:
        pace = "Fast"
    else:
        pace = "Very fast"
    
    return {
        "num_words": num_words,
        "duration": duration,
        "words_per_second": words_per_second,
        "words_per_minute": words_per_minute,
        "pace": pace
    }


def extract_catchphrases(
    transcription_result: Dict[str, Any],
    min_frequency: int = 2,
    topk: int = 5
) -> Dict[str, Any]:
    """Extract repeated phrases (n-grams) as potential catchphrases."""
    if not transcription_result or "text" not in transcription_result:
        raise ValueError("Invalid transcription result")
    
    text = transcription_result.get("text", "")
    words = re.findall(r"\b[\w']+\b", text.lower())
    
    if len(words) < 2:
        return {
            "bigrams": [],
            "trigrams": [],
            "all_catchphrases": []
        }
    
    # Bigrams
    bigrams = [tuple(words[i:i+2]) for i in range(len(words) - 1)]
    bigram_counts = Counter(bigrams)
    top_bigrams = [
        {"phrase": " ".join(phrase), "count": count}
        for phrase, count in bigram_counts.most_common(topk * 2)
        if count >= min_frequency
    ][:topk]
    
    # Trigrams
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    trigram_counts = Counter(trigrams)
    top_trigrams = [
        {"phrase": " ".join(phrase), "count": count}
        for phrase, count in trigram_counts.most_common(topk * 2)
        if count >= min_frequency
    ][:topk]
    
    all_catchphrases = list(set(
        [b["phrase"] for b in top_bigrams] + [t["phrase"] for t in top_trigrams]
    ))
    
    return {
        "bigrams": top_bigrams,
        "trigrams": top_trigrams,
        "all_catchphrases": all_catchphrases
    }


def analyze_speech_pauses(
    transcription_result: Dict[str, Any],
    min_pause: float = 0.5
) -> Dict[str, Any]:
    """Analyze pause patterns in speech."""
    if not transcription_result or "segments" not in transcription_result:
        raise ValueError("Invalid transcription result (missing segments)")
    
    segments = transcription_result.get("segments", [])
    
    if len(segments) < 2:
        return {
            "num_pauses": 0,
            "mean_pause": 0,
            "max_pause": 0,
            "pause_distribution": [],
            "style": "Continuous speech"
        }
    
    pauses = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1]["start"] - segments[i]["end"]
        if gap >= min_pause:
            pauses.append({
                "start": segments[i]["end"],
                "duration": gap
            })
    
    if pauses:
        pause_durations = [p["duration"] for p in pauses]
        return {
            "num_pauses": len(pauses),
            "mean_pause": float(sum(pause_durations) / len(pause_durations)),
            "max_pause": float(max(pause_durations)),
            "min_pause": float(min(pause_durations)),
            "pause_distribution": pauses[:10],  # Top 10 pauses
            "style": "Frequent pauses" if len(pauses) > len(segments) * 0.3 else "Fluent speech"
        }
    else:
        return {
            "num_pauses": 0,
            "mean_pause": 0,
            "max_pause": 0,
            "pause_distribution": [],
            "style": "Continuous speech"
        }


def analyze_prosody(audio_path: str) -> Dict[str, Any]:
    """
    Analyze speech prosody using librosa.
    
    提取:
    - Pitch (基频/F0): 使用 librosa.pyin
    - Intensity (强度): 使用 librosa.feature.rms
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for prosody analysis.")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        logger.debug(f"Analyzing prosody with librosa: {audio_path}")
        
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # Pitch extraction using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=sr
        )
        
        pitch_values = f0[~np.isnan(f0)]
        
        if len(pitch_values) == 0:
            logger.warning("No pitch values extracted")
            mean_pitch = 0.0
            pitch_std = 0.0
            pitch_range = 0.0
        else:
            mean_pitch = float(np.mean(pitch_values))
            pitch_std = float(np.std(pitch_values))
            pitch_range = float(np.max(pitch_values) - np.min(pitch_values))
        
        # Intensity (RMS energy)
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        mean_intensity = float(np.mean(rms_db))
        intensity_std = float(np.std(rms_db))
        
        # Classify tone
        if mean_pitch < 150:
            tone = "Low"
        elif mean_pitch < 250:
            tone = "Medium"
        else:
            tone = "High"
        
        # Classify prosody style
        if pitch_std < 20:
            prosody_style = "Monotone"
        elif pitch_std < 50:
            prosody_style = "Moderate variation"
        else:
            prosody_style = "Expressive"
        
        voiced_ratio = float(np.sum(~np.isnan(f0))) / len(f0) if len(f0) > 0 else 0.0
        
        return {
            "mean_pitch_hz": mean_pitch,
            "pitch_std": pitch_std,
            "pitch_range": pitch_range,
            "mean_intensity_db": mean_intensity,
            "intensity_std": intensity_std,
            "tone": tone,
            "prosody_style": prosody_style,
            "voiced_ratio": voiced_ratio,
            "method": "librosa.pyin"
        }
        
    except Exception as e:
        logger.error(f"Prosody analysis failed: {e}")
        raise


def analyze_emotion(audio_path: str) -> Dict[str, Any]:
    """
    Analyze speech emotion using HuggingFace model.
    
    使用 HuBERT-large 或 Wav2Vec2 情感识别模型。
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    model, processor = _load_emotion_classifier()
    
    if model is None:
        return {
            "dominant_emotion": "Unknown",
            "emotion_scores": {},
            "confidence": 0.0,
            "method": "N/A"
        }
    
    device = next(model.parameters()).device
    
    try:
        logger.debug(f"Running emotion classifier on {audio_path}")
        
        # Load audio at 16kHz (model expected rate)
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Limit length (30 seconds max)
        max_length = 16000 * 30
        if len(y) > max_length:
            y = y[:max_length]
        
        # Extract features
        inputs = processor(
            y,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Get labels from model config
        id2label = model.config.id2label
        emotion_scores = {}
        for i, prob in enumerate(probs):
            label = id2label.get(i, f"emotion_{i}")
            emotion_scores[label] = float(prob)
        
        # Find dominant emotion
        dominant_idx = torch.argmax(probs).item()
        dominant_emotion = id2label.get(dominant_idx, "Unknown")
        confidence = float(probs[dominant_idx])
        
        # Sort by score
        sorted_scores = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "emotion_scores": dict(sorted_scores),
            "top_3": sorted_scores[:3],
            "method": f"HuggingFace {EMOTION_MODEL.split('/')[-1]}"
        }
        
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        return {
            "dominant_emotion": "Unknown",
            "emotion_scores": {},
            "confidence": 0.0,
            "method": "failed",
            "error": str(e)
        }


def extract_full_asr_metrics(
    audio_path: str,
    language: str = "en",
    model_size: str = DEFAULT_WHISPER_MODEL,
    beam_size: int = 5,
    enable_prosody: bool = True,
    enable_emotion: bool = True
) -> Dict[str, Any]:
    """
    Extract comprehensive ASR metrics from audio.
    
    Args:
        audio_path: Path to audio file
        language: Language code (or "auto" for auto-detect)
        model_size: Whisper model size
        enable_prosody: Enable prosody analysis
        enable_emotion: Enable emotion analysis
        
    Returns:
        dict: Complete ASR analysis
    """
    # Transcribe
    transcription = transcribe_audio(audio_path, language, model_size, beam_size)
    
    # Analyze speech rate
    speech_rate = analyze_speech_rate(transcription, audio_path)
    
    # Extract catchphrases
    catchphrases = extract_catchphrases(transcription)
    
    # Analyze pauses
    pauses = analyze_speech_pauses(transcription)
    
    result = {
        # 基础信息
        "text": transcription.get("text", ""),
        "implementation": transcription.get("implementation", ""),
        "model_size": transcription.get("model_size", ""),
        "detected_language": transcription.get("language", ""),
        
        # 语速分析
        "num_words": speech_rate.get("num_words", 0),
        "words_per_second": speech_rate.get("words_per_second"),
        "words_per_minute": speech_rate.get("words_per_minute"),
        "pace": speech_rate.get("pace", "Unknown"),
        
        # 口头禅
        "catchphrases": catchphrases.get("all_catchphrases", []),
        "catchphrase_detail": catchphrases,
        
        # 停顿分析
        "num_pauses": pauses.get("num_pauses", 0),
        "pause_style": pauses.get("style", "Unknown"),
        "pause_detail": pauses,
        
        # 转录片段
        "segments": transcription.get("segments", [])
    }
    
    # Prosody analysis (optional)
    if enable_prosody:
        try:
            prosody = analyze_prosody(audio_path)
            result["prosody"] = prosody
        except ImportError:
            logger.warning("Prosody analysis skipped (librosa not available)")
        except Exception as e:
            logger.warning(f"Prosody analysis failed: {e}")
    
    # Emotion analysis (optional)
    if enable_emotion:
        try:
            emotion = analyze_emotion(audio_path)
            result["emotion"] = emotion
        except ImportError:
            logger.warning("Emotion analysis skipped (transformers not available)")
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
    
    logger.info(f"ASR metrics extracted successfully from {audio_path}")
    logger.info(f"  → Words: {result['num_words']} | Pace: {result['pace']} | "
               f"Language: {result['detected_language']}")
    
    return result
