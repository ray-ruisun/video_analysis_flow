#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Speech Recognition (ASR) module.

Transcribes narration using Whisper and extracts speech rate, catchphrases,
speech patterns, prosody (tone), and emotion analysis.
"""

import os
import re
from collections import Counter

import numpy as np
from loguru import logger

# Required dependencies
try:
    import soundfile as sf
except ImportError as e:
    logger.error(f"soundfile not installed: {e}")
    logger.error("Install with: pip install soundfile")
    raise ImportError("soundfile is required. Install with: pip install soundfile")

# ASR implementation - at least one must be available
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
    raise ImportError("Whisper is required. Install faster-whisper or openai-whisper")

# Prosody analysis - 使用 librosa 替代 parselmouth
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available for prosody analysis. Install with: pip install librosa")

try:
    from speechbrain.pretrained import EncoderClassifier
except ImportError as e:
    logger.error(f"speechbrain not installed: {e}")
    logger.error("Install with: pip install speechbrain torchaudio")
    raise ImportError("speechbrain is required. Install with: pip install speechbrain torchaudio")

import torch
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "video_style_pipeline"
EMOTION_MODEL_DIR = CACHE_DIR / "speechbrain_emotion"
EMOTION_MODEL_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
_EMOTION_CLASSIFIER = None


def _load_emotion_classifier():
    """Lazy-load SpeechBrain emotion recognition model."""
    global _EMOTION_CLASSIFIER
    if _EMOTION_CLASSIFIER is not None:
        return _EMOTION_CLASSIFIER
    
    EMOTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading SpeechBrain emotion recognition model (IEMOCAP).")
    _EMOTION_CLASSIFIER = EncoderClassifier.from_hparams(
        source=EMOTION_MODEL_SOURCE,
        savedir=str(EMOTION_MODEL_DIR)
    )
    return _EMOTION_CLASSIFIER


def transcribe_audio(audio_path, language="en", model_size="small"):
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_path: Path to audio file (preferably wav)
        language: Language code (default: "en")
        model_size: Model size ("tiny", "base", "small", "medium", "large")
        
    Returns:
        dict: Transcription results with text, timing, and metadata
        
    Raises:
        FileNotFoundError: If audio file not found
        ValueError: If transcription fails
    """
    if not audio_path:
        logger.error("Audio path is None or empty")
        raise ValueError("Audio path is None or empty")
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
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
    """
    Calculate speech rate (words per second).
    
    Args:
        transcription_result: Result from transcribe_audio()
        audio_path: Path to audio file (to get duration)
        
    Returns:
        dict: Speech rate metrics
        
    Raises:
        ValueError: If transcription or audio file is invalid
    """
    if not transcription_result or "text" not in transcription_result:
        logger.error("Invalid transcription result")
        raise ValueError("Invalid transcription result")
    
    text = transcription_result.get("text", "")
    
    # Extract words
    words = re.findall(r"\b[\w']+\b", text.lower())
    num_words = len(words)
    
    if num_words == 0:
        logger.warning("No words found in transcription")
        raise ValueError("No words found in transcription")
    
    # Get audio duration
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        info = sf.info(audio_path)
        duration = info.duration
    except Exception as e:
        logger.error(f"Failed to get audio duration: {e}")
        raise ValueError(f"Failed to get audio duration: {e}")
    
    if duration <= 0:
        logger.error(f"Invalid audio duration: {duration}")
        raise ValueError(f"Invalid audio duration: {duration}")
    
    # Calculate rates
    words_per_second = num_words / duration
    words_per_minute = words_per_second * 60
    
    # Classify speech pace
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
    """
    Extract repeated phrases (n-grams) as potential catchphrases.
    
    Args:
        transcription_result: Result from transcribe_audio()
        min_frequency: Minimum occurrences to consider
        topk: Number of top phrases to return
        
    Returns:
        dict: Catchphrases and repeated patterns
        
    Raises:
        ValueError: If transcription is invalid
    """
    if not transcription_result or "text" not in transcription_result:
        logger.error("Invalid transcription result")
        raise ValueError("Invalid transcription result")
    
    text = transcription_result.get("text", "")
    
    # Tokenize into words
    words = re.findall(r"\b[\w']+\b", text.lower())
    
    if len(words) < 2:
        logger.warning("Insufficient words for catchphrase extraction")
        return {
            "bigrams": [],
            "trigrams": [],
            "all_catchphrases": []
        }
    
    # Extract bigrams (2-word phrases)
    bigrams = [tuple(words[i:i+2]) for i in range(len(words) - 1)]
    bigram_counts = Counter(bigrams)
    top_bigrams = [
        " ".join(phrase) 
        for phrase, count in bigram_counts.most_common(topk * 2)
        if count >= min_frequency
    ][:topk]
    
    # Extract trigrams (3-word phrases)
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    trigram_counts = Counter(trigrams)
    top_trigrams = [
        " ".join(phrase)
        for phrase, count in trigram_counts.most_common(topk * 2)
        if count >= min_frequency
    ][:topk]
    
    # Combine unique catchphrases
    all_catchphrases = list(set(top_bigrams + top_trigrams))
    
    return {
        "bigrams": top_bigrams,
        "trigrams": top_trigrams,
        "all_catchphrases": all_catchphrases
    }


def analyze_speech_pauses(transcription_result, min_pause=0.5):
    """
    Analyze pause patterns in speech.
    
    Args:
        transcription_result: Result from transcribe_audio() with segments
        min_pause: Minimum pause duration to consider (seconds)
        
    Returns:
        dict: Pause analysis metrics
        
    Raises:
        ValueError: If transcription is invalid
    """
    if not transcription_result or "segments" not in transcription_result:
        logger.error("Invalid transcription result (missing segments)")
        raise ValueError("Invalid transcription result (missing segments)")
    
    segments = transcription_result.get("segments", [])
    
    if len(segments) < 2:
        return {
            "num_pauses": 0,
            "mean_pause": 0,
            "max_pause": 0,
            "style": "Continuous speech"
        }
    
    # Calculate gaps between segments
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
    Analyze speech prosody (tone, pitch, intensity) using librosa.
    
    使用 librosa.pyin 提取基频 (F0/pitch)，使用 librosa.feature.rms 提取强度。
    这是 parselmouth/Praat 的替代方案，无需额外安装。
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        dict: Prosody analysis metrics
        
    Raises:
        ImportError: If librosa is not available
        FileNotFoundError: If audio file not found
    """
    if not LIBROSA_AVAILABLE:
        logger.error("librosa is required for prosody analysis")
        logger.error("Install with: pip install librosa")
        raise ImportError("librosa is required. Install with: pip install librosa")
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        logger.debug(f"Analyzing prosody with librosa: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # Extract pitch using pyin (probabilistic YIN)
        # fmin/fmax 设置人声基频范围 (约 80-400 Hz)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C6'),  # ~1047 Hz
            sr=sr
        )
        
        # Filter out unvoiced frames (NaN values)
        pitch_values = f0[~np.isnan(f0)]
        
        if len(pitch_values) == 0:
            logger.warning("No pitch values extracted")
            mean_pitch = 0.0
            pitch_std = 0.0
        else:
            mean_pitch = float(np.mean(pitch_values))
            pitch_std = float(np.std(pitch_values))
        
        # Extract intensity (RMS energy)
        rms = librosa.feature.rms(y=y)[0]
        # Convert to dB scale (similar to Praat intensity)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        mean_intensity = float(np.mean(rms_db))
        intensity_std = float(np.std(rms_db))
        
        # Classify tone based on mean pitch
        if mean_pitch < 150:
            tone = "Low"
        elif mean_pitch < 250:
            tone = "Medium"
        else:
            tone = "High"
        
        # Classify prosody style based on pitch variation
        if pitch_std < 20:
            prosody_style = "Monotone"
        elif pitch_std < 50:
            prosody_style = "Moderate variation"
        else:
            prosody_style = "Expressive"
        
        # Additional metrics: voiced ratio (有声段比例)
        voiced_ratio = float(np.sum(~np.isnan(f0))) / len(f0) if len(f0) > 0 else 0.0
        
        return {
            "mean_pitch_hz": mean_pitch,
            "pitch_std": pitch_std,
            "mean_intensity_db": mean_intensity,
            "intensity_std": intensity_std,
            "tone": tone,
            "prosody_style": prosody_style,
            "voiced_ratio": voiced_ratio,  # 新增：有声段比例
            "method": "librosa.pyin"  # 标记使用的方法
        }
        
    except Exception as e:
        logger.error(f"Prosody analysis failed: {e}")
        raise


def analyze_emotion(audio_path):
    """
    Analyze speech emotion using SpeechBrain's wav2vec2-based classifier.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        dict: Emotion analysis results
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        classifier = _load_emotion_classifier()
        logger.debug(f"Running SpeechBrain emotion classifier on {audio_path}")
        out_prob, score, index, predicted_label = classifier.classify_file(audio_path)
        probabilities = torch.softmax(out_prob.squeeze(), dim=-1)
        
        label_encoder = classifier.hparams.label_encoder
        emotion_scores = {}
        for i in range(probabilities.shape[0]):
            label = label_encoder.decode_torch(torch.tensor([i])).strip()
            emotion_scores[label] = float(probabilities[i])
        
        return {
            "dominant_emotion": str(predicted_label),
            "emotion_scores": emotion_scores
        }
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        raise


def extract_full_asr_metrics(audio_path, language="en", model_size="small", 
                            enable_prosody=False, enable_emotion=False):
    """
    Extract comprehensive ASR metrics from audio.
    
    Args:
        audio_path: Path to audio file
        language: Language code
        model_size: Whisper model size
        enable_prosody: Enable prosody analysis (requires Praat)
        enable_emotion: Enable emotion analysis (requires SpeechBrain)
        
    Returns:
        dict: Complete ASR analysis
        
    Raises:
        FileNotFoundError: If audio file not found
        ImportError: If required dependencies missing
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
            logger.warning("Prosody analysis skipped (Praat not available)")
        except Exception as e:
            logger.warning(f"Prosody analysis failed: {e}")
    
    # Emotion analysis (optional)
    if enable_emotion:
        try:
            emotion = analyze_emotion(audio_path)
            result["emotion"] = emotion
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
    
    logger.info(f"ASR metrics extracted successfully from {audio_path}")
    return result
