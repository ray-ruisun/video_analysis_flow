#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Speech Recognition (ASR) module.
Transcribes narration and extracts speech rate, catchphrases, and speech patterns.
"""

import os
import re
from collections import Counter

# Optional dependency handling
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

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


def transcribe_audio(audio_path, language="en", model_size="small"):
    """
    Transcribe audio using Whisper (faster-whisper or OpenAI Whisper).
    
    Args:
        audio_path: Path to audio file (preferably wav)
        language: Language code (default: "en")
        model_size: Model size ("tiny", "base", "small", "medium", "large")
        
    Returns:
        dict: Transcription results with text, timing, and metadata
    """
    if not ASR_AVAILABLE:
        return {
            "available": False,
            "error": "Whisper not installed. Install with: pip install faster-whisper (or openai-whisper)"
        }
    
    if not audio_path or not os.path.exists(audio_path):
        return {
            "available": False,
            "error": "Audio file not found"
        }
    
    try:
        transcribed_text = ""
        segments_list = []
        
        if ASR_IMPLEMENTATION == "faster-whisper":
            # faster-whisper implementation (more efficient)
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
        
        return {
            "available": True,
            "implementation": ASR_IMPLEMENTATION,
            "text": transcribed_text.strip(),
            "segments": segments_list
        }
        
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


def analyze_speech_rate(transcription_result, audio_path):
    """
    Calculate speech rate (words per second).
    
    Args:
        transcription_result: Result from transcribe_audio()
        audio_path: Path to audio file (to get duration)
        
    Returns:
        dict: Speech rate metrics
    """
    if not transcription_result.get("available"):
        return {"available": False}
    
    text = transcription_result.get("text", "")
    
    # Extract words (alphanumeric + apostrophes)
    words = re.findall(r"\b[\w']+\b", text.lower())
    num_words = len(words)
    
    # Get audio duration
    duration = None
    if SOUNDFILE_AVAILABLE and os.path.exists(audio_path):
        try:
            info = sf.info(audio_path)
            duration = info.duration
        except Exception:
            pass
    
    # Calculate rates
    words_per_second = None
    words_per_minute = None
    
    if duration and duration > 0:
        words_per_second = num_words / duration
        words_per_minute = words_per_second * 60
    
    # Classify speech pace
    if words_per_second:
        if words_per_second < 1.5:
            pace = "Slow/Deliberate"
        elif words_per_second < 2.5:
            pace = "Moderate"
        elif words_per_second < 3.5:
            pace = "Fast"
        else:
            pace = "Very fast"
    else:
        pace = "Unknown"
    
    return {
        "available": True,
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
    """
    if not transcription_result.get("available"):
        return {"available": False}
    
    text = transcription_result.get("text", "")
    
    # Tokenize into words
    words = re.findall(r"\b[\w']+\b", text.lower())
    
    if len(words) < 2:
        return {
            "available": True,
            "bigrams": [],
            "trigrams": []
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
        "available": True,
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
    """
    if not transcription_result.get("available"):
        return {"available": False}
    
    segments = transcription_result.get("segments", [])
    
    if len(segments) < 2:
        return {
            "available": True,
            "num_pauses": 0,
            "mean_pause": 0,
            "max_pause": 0
        }
    
    # Calculate gaps between segments
    pauses = []
    for i in range(len(segments) - 1):
        gap = segments[i + 1]["start"] - segments[i]["end"]
        if gap >= min_pause:
            pauses.append(gap)
    
    if pauses:
        return {
            "available": True,
            "num_pauses": len(pauses),
            "mean_pause": float(sum(pauses) / len(pauses)),
            "max_pause": float(max(pauses)),
            "style": "Frequent pauses" if len(pauses) > len(segments) * 0.3 else "Continuous speech"
        }
    else:
        return {
            "available": True,
            "num_pauses": 0,
            "mean_pause": 0,
            "max_pause": 0,
            "style": "Continuous speech"
        }


def extract_full_asr_metrics(audio_path, language="en", model_size="small"):
    """
    Extract comprehensive ASR metrics from audio.
    
    Args:
        audio_path: Path to audio file
        language: Language code
        model_size: Whisper model size
        
    Returns:
        dict: Complete ASR analysis
    """
    # Transcribe
    transcription = transcribe_audio(audio_path, language, model_size)
    
    if not transcription.get("available"):
        return transcription
    
    # Analyze speech rate
    speech_rate = analyze_speech_rate(transcription, audio_path)
    
    # Extract catchphrases
    catchphrases = extract_catchphrases(transcription)
    
    # Analyze pauses
    pauses = analyze_speech_pauses(transcription)
    
    # Combine results
    return {
        "available": True,
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

