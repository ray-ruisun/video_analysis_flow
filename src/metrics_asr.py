#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Speech Recognition (ASR) module (PyTorch + HuggingFace 版本)

使用纯 PyTorch 生态系统:
- faster-whisper: 高效的 Whisper 实现 (CTranslate2)
- librosa: 韵律分析 (pitch, intensity)
- transformers: HuggingFace 情感分析模型

分析内容:
- 语音转录: Whisper
- 语速分析: 每秒/每分钟词数
- 口头禅检测: n-gram 频率分析
- 停顿分析: 基于转录片段间隔
- 韵律分析: 基于 librosa (pitch, intensity)
- 情感分析: HuggingFace 音频情感模型
"""

import os
import re
from collections import Counter
from pathlib import Path

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

# Emotion analysis - HuggingFace transformers (optional)
EMOTION_AVAILABLE = False
_EMOTION_CLASSIFIER = None
_EMOTION_FEATURE_EXTRACTOR = None

try:
    import torch
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    EMOTION_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available for emotion analysis.")
    logger.warning("Install with: pip install transformers torch")

# 缓存目录
CACHE_DIR = Path.home() / ".cache" / "video_style_pipeline"

# HuggingFace 情感分析模型
# 使用 wav2vec2 情感识别模型
EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
# 备选模型:
# "superb/wav2vec2-base-superb-er"  # SUPERB 情感识别
# "facebook/wav2vec2-large-xlsr-53"  # 需要 fine-tune


def _load_emotion_classifier():
    """Lazy-load HuggingFace emotion recognition model."""
    global _EMOTION_CLASSIFIER, _EMOTION_FEATURE_EXTRACTOR
    
    if not EMOTION_AVAILABLE:
        return None, None
    
    if _EMOTION_CLASSIFIER is not None:
        return _EMOTION_CLASSIFIER, _EMOTION_FEATURE_EXTRACTOR
    
    try:
        logger.info(f"Loading HuggingFace emotion model: {EMOTION_MODEL}")
        _EMOTION_FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(
            EMOTION_MODEL,
            cache_dir=str(CACHE_DIR / "hf_models")
        )
        _EMOTION_CLASSIFIER = AutoModelForAudioClassification.from_pretrained(
            EMOTION_MODEL,
            cache_dir=str(CACHE_DIR / "hf_models")
        )
        _EMOTION_CLASSIFIER.eval()
        logger.info("Emotion classifier loaded successfully")
        return _EMOTION_CLASSIFIER, _EMOTION_FEATURE_EXTRACTOR
    except Exception as e:
        logger.warning(f"Failed to load emotion classifier: {e}")
        return None, None


def transcribe_audio(audio_path, language="en", model_size="small"):
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_path: Path to audio file
        language: Language code (default: "en")
        model_size: Model size ("tiny", "base", "small", "medium", "large")
        
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
        
        if ASR_IMPLEMENTATION == "faster-whisper":
            logger.debug(f"Using faster-whisper for transcription: {audio_path}")
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments, info = model.transcribe(audio_path, language=language, vad_filter=True)
            
            for segment in segments:
                text = segment.text.strip()
                transcribed_text += text + " "
                segments_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": text
                })
        
        else:  # openai-whisper
            logger.debug(f"Using openai-whisper for transcription: {audio_path}")
            model = whisper.load_model(model_size)
            result = model.transcribe(audio_path, language=language)
            transcribed_text = result.get("text", "")
            
            if "segments" in result:
                for seg in result["segments"]:
                    segments_list.append({
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", "").strip()
                    })
        
        if not transcribed_text.strip():
            logger.warning(f"No transcription obtained from {audio_path}")
            raise ValueError(f"No transcription obtained from {audio_path}")
        
        logger.info(f"Transcription completed: {len(transcribed_text.split())} words")
        
        return {
            "implementation": ASR_IMPLEMENTATION,
            "text": transcribed_text.strip(),
            "segments": segments_list
        }
        
    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}")
        raise


def analyze_speech_rate(transcription_result, audio_path):
    """Calculate speech rate (words per second)."""
    if not transcription_result or "text" not in transcription_result:
        raise ValueError("Invalid transcription result")
    
    text = transcription_result.get("text", "")
    words = re.findall(r"\b[\w']+\b", text.lower())
    num_words = len(words)
    
    if num_words == 0:
        raise ValueError("No words found in transcription")
    
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


def extract_catchphrases(transcription_result, min_frequency=2, topk=5):
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
        " ".join(phrase) 
        for phrase, count in bigram_counts.most_common(topk * 2)
        if count >= min_frequency
    ][:topk]
    
    # Trigrams
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    trigram_counts = Counter(trigrams)
    top_trigrams = [
        " ".join(phrase)
        for phrase, count in trigram_counts.most_common(topk * 2)
        if count >= min_frequency
    ][:topk]
    
    all_catchphrases = list(set(top_bigrams + top_trigrams))
    
    return {
        "bigrams": top_bigrams,
        "trigrams": top_trigrams,
        "all_catchphrases": all_catchphrases
    }


def analyze_speech_pauses(transcription_result, min_pause=0.5):
    """Analyze pause patterns in speech."""
    if not transcription_result or "segments" not in transcription_result:
        raise ValueError("Invalid transcription result (missing segments)")
    
    segments = transcription_result.get("segments", [])
    
    if len(segments) < 2:
        return {
            "num_pauses": 0,
            "mean_pause": 0,
            "max_pause": 0,
            "style": "Continuous speech"
        }
    
    pauses = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1]["start"] - segments[i]["end"]
        if gap >= min_pause:
            pauses.append(gap)
    
    if pauses:
        return {
            "num_pauses": len(pauses),
            "mean_pause": float(sum(pauses) / len(pauses)),
            "max_pause": float(max(pauses)),
            "style": "Frequent pauses" if len(pauses) > len(segments) * 0.3 else "Continuous speech"
        }
    else:
        return {
            "num_pauses": 0,
            "mean_pause": 0,
            "max_pause": 0,
            "style": "Continuous speech"
        }


def analyze_prosody(audio_path):
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
        else:
            mean_pitch = float(np.mean(pitch_values))
            pitch_std = float(np.std(pitch_values))
        
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


def analyze_emotion(audio_path):
    """
    Analyze speech emotion using HuggingFace model.
    
    使用 wav2vec2 情感识别模型。
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    model, feature_extractor = _load_emotion_classifier()
    
    if model is None:
        raise ImportError("Emotion classifier not available.")
    
    try:
        logger.debug(f"Running HuggingFace emotion classifier on {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Limit length (30 seconds max)
        max_length = 16000 * 30
        if len(y) > max_length:
            y = y[:max_length]
        
        # Extract features
        inputs = feature_extractor(
            y,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Get labels
        id2label = model.config.id2label
        emotion_scores = {}
        for i, prob in enumerate(probs):
            label = id2label.get(i, f"emotion_{i}")
            emotion_scores[label] = float(prob)
        
        # Find dominant emotion
        dominant_idx = torch.argmax(probs).item()
        dominant_emotion = id2label.get(dominant_idx, "Unknown")
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": emotion_scores,
            "method": "HuggingFace wav2vec2"
        }
        
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        raise


def extract_full_asr_metrics(audio_path, language="en", model_size="small", 
                            enable_prosody=True, enable_emotion=True):
    """
    Extract comprehensive ASR metrics from audio.
    
    Args:
        audio_path: Path to audio file
        language: Language code
        model_size: Whisper model size
        enable_prosody: Enable prosody analysis
        enable_emotion: Enable emotion analysis
        
    Returns:
        dict: Complete ASR analysis
    """
    # Transcribe
    transcription = transcribe_audio(audio_path, language, model_size)
    
    # Analyze speech rate
    speech_rate = analyze_speech_rate(transcription, audio_path)
    
    # Extract catchphrases
    catchphrases = extract_catchphrases(transcription)
    
    # Analyze pauses
    pauses = analyze_speech_pauses(transcription)
    
    result = {
        "text": transcription.get("text", ""),
        "implementation": transcription.get("implementation", ""),
        "num_words": speech_rate.get("num_words", 0),
        "words_per_second": speech_rate.get("words_per_second"),
        "words_per_minute": speech_rate.get("words_per_minute"),
        "pace": speech_rate.get("pace", "Unknown"),
        "catchphrases": catchphrases.get("all_catchphrases", []),
        "num_pauses": pauses.get("num_pauses", 0),
        "pause_style": pauses.get("style", "Unknown")
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
    return result
