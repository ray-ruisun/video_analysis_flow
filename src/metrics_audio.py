#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio metrics extraction module.

Analyzes tempo (BPM), beat timing, energy distribution, narration presence,
BGM style, instruments, and mood.
"""

import os
import numpy as np
from loguru import logger

# Required dependencies
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError as e:
    logger.error(f"librosa/soundfile not installed: {e}")
    logger.error("Install with: pip install librosa soundfile")
    raise ImportError("librosa and soundfile are required. Install with: pip install librosa soundfile")

# Essentia is optional - provides BGM style, mood, and instrument detection
ESSENTIA_AVAILABLE = False
MUSIC_EXTRACTOR = None

try:
    import essentia.standard as es
    # Try to create MusicExtractor with different parameter combinations
    # (API changed between versions)
    try:
        # Newer version (without highlevelStats)
        MUSIC_EXTRACTOR = es.MusicExtractor(
            lowlevelStats=['mean', 'stdev'],
            rhythmStats=['mean', 'stdev'],
            tonalStats=['mean', 'stdev']
        )
        ESSENTIA_AVAILABLE = True
        logger.debug("Essentia MusicExtractor initialized (new API)")
    except (TypeError, ValueError):
        try:
            # Older version (with highlevelStats)
            MUSIC_EXTRACTOR = es.MusicExtractor(
                lowlevelStats=['mean', 'stdev'],
                rhythmStats=['mean', 'stdev'],
                tonalStats=['mean', 'stdev'],
                highlevelStats=['mean']
            )
            ESSENTIA_AVAILABLE = True
            logger.debug("Essentia MusicExtractor initialized (legacy API)")
        except Exception as e:
            logger.warning(f"Failed to initialize MusicExtractor: {e}")
            ESSENTIA_AVAILABLE = False
except ImportError:
    logger.warning("Essentia not available. BGM style, mood, and instrument detection will be limited.")
    logger.warning("Install with: pip install essentia-tensorflow")

MOOD_TAGS = [
    "mood_happy",
    "mood_sad",
    "mood_aggressive",
    "mood_relaxed",
    "mood_party",
    "mood_acoustic",
    "mood_electronic"
]


def extract_audio_metrics(audio_path):
    """
    Extract comprehensive audio metrics from a wav file.
    
    Args:
        audio_path: Path to audio file (preferably 22.05kHz mono wav)
        
    Returns:
        dict: Audio metrics including tempo, beats, energy, speech ratio, instruments, etc.
        
    Raises:
        FileNotFoundError: If audio file not found
        ValueError: If audio processing fails
    """
    if not audio_path:
        logger.error("Audio path is None or empty")
        raise ValueError("Audio path is None or empty")
    
    audio_path = str(audio_path)
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Load audio (force mono, 22.05kHz)
        logger.debug(f"Loading audio from {audio_path}")
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        if len(y) == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            raise ValueError(f"Audio file is empty: {audio_path}")
        
        # Normalize
        y = y / (np.max(np.abs(y)) + 1e-6)
        
        # Harmonic-Percussive Source Separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Percussive energy ratio (proxy for rhythmic content)
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
        
        # Speech presence proxy (combination of flatness and ZCR)
        speech_ratio = float(np.clip(mean_flatness * 1.8 + mean_zcr * 0.8, 0, 1))
        
        # RMS energy over time
        rms = librosa.feature.rms(y=y)[0]
        mean_energy = float(np.mean(rms))
        energy_variance = float(np.var(rms))
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        mean_rolloff = float(np.mean(rolloff))
        
        # Essentia high-level descriptors (optional)
        genre = "Unknown"
        mood_summary = "Unknown"
        mood_tags = []
        instruments = {"detected_instruments": [], "method": "N/A"}
        tonal_key = None
        
        if ESSENTIA_AVAILABLE:
            try:
                essentia_features = extract_essentia_features(audio_path)
                if essentia_features:
                    # Genre detection
                    if 'highlevel.genre_dortmund.value' in essentia_features:
                        genre = str(essentia_features['highlevel.genre_dortmund.value'])
                    
                    # Mood analysis
                    try:
                        mood_summary, mood_tags = summarize_mood_from_essentia(essentia_features)
                    except Exception as e:
                        logger.warning(f"Mood analysis failed: {e}")
                    
                    # Instrument detection
                    try:
                        instruments = detect_instruments(essentia_features)
                    except Exception as e:
                        logger.warning(f"Instrument detection failed: {e}")
                    
                    # Key signature
                    if 'tonal.key_key' in essentia_features and 'tonal.key_scale' in essentia_features:
                        tonal_key = f"{str(essentia_features['tonal.key_key']).capitalize()} {str(essentia_features['tonal.key_scale']).capitalize()}"
            except Exception as e:
                logger.warning(f"Essentia analysis failed, using librosa fallback: {e}")
        else:
            logger.debug("Essentia not available, using librosa-only analysis")
            # Fallback: estimate mood from tempo and energy
            mood_summary = classify_bgm_mood(float(tempo), percussive_ratio, mean_centroid, energy_variance)
        
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
            "mood": mood_summary,
            "mood_tags": mood_tags,
            "instruments": instruments,
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


def extract_essentia_features(audio_path):
    """Run Essentia MusicExtractor on the audio file."""
    if not ESSENTIA_AVAILABLE or MUSIC_EXTRACTOR is None:
        logger.warning("Essentia MusicExtractor not available")
        return None
    
    try:
        features, _ = MUSIC_EXTRACTOR(audio_path)
        return features
    except Exception as e:
        logger.error(f"Essentia MusicExtractor failed: {e}")
        return None


def summarize_mood_from_essentia(essentia_features):
    """Summarize mood predictions from Essentia high-level descriptors."""
    if essentia_features is None:
        return "Unknown", []
    
    mood_entries = []
    for tag in MOOD_TAGS:
        value_key = f"highlevel.{tag}.value"
        prob_key = f"highlevel.{tag}.probability"
        if value_key in essentia_features and prob_key in essentia_features:
            label = str(essentia_features[value_key])
            probability = float(essentia_features[prob_key])
            mood_entries.append({"label": label, "probability": probability})
    
    if not mood_entries:
        logger.warning("Essentia mood descriptors not found in features")
        return "Unknown", []
    
    top_entry = max(mood_entries, key=lambda item: item["probability"])
    summary = f"{top_entry['label']} ({top_entry['probability']:.2f})"
    return summary, mood_entries


def detect_instruments(essentia_features):
    """
    Detect instrumentation characteristics from Essentia high-level descriptors.
    """
    if essentia_features is None:
        return {"detected_instruments": [], "method": "N/A"}
    
    instrumentation = []
    
    # Check if the required key exists
    voice_key = 'highlevel.voice_instrumental.value'
    if voice_key not in essentia_features:
        return {"detected_instruments": [], "method": "Essentia (limited)"}
    
    voice_value = str(essentia_features[voice_key])
    if voice_value == "instrumental":
        instrumentation.append("Instrumental focus")
    else:
        instrumentation.append("Vocal focus")
    
    # Safely check for acoustic/electronic keys
    acoustic_key = 'highlevel.mood_acoustic.value'
    electronic_key = 'highlevel.mood_electronic.value'
    
    if acoustic_key in essentia_features and str(essentia_features[acoustic_key]) == "acoustic":
        instrumentation.append("Acoustic timbre")
    if electronic_key in essentia_features and str(essentia_features[electronic_key]) == "electronic":
        instrumentation.append("Electronic elements")
    
    instrumentation = list(dict.fromkeys(instrumentation))  # deduplicate while preserving order
    
    return {
        "detected_instruments": instrumentation,
        "method": "Essentia MusicExtractor"
    }


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
        
    Raises:
        ValueError: If inputs are invalid
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
    
    # Approximate cut times (assuming uniform distribution)
    approx_cut_times = np.linspace(0, video_duration, num=num_cuts + 2)[1:-1]
    
    beat_array = np.array(beat_times)
    aligned_count = 0
    
    for cut_time in approx_cut_times:
        # Check if any beat is within tolerance
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
    
    Args:
        audio_path: Path to audio file
        window_sec: Window size for energy analysis
        
    Returns:
        dict: Energy dynamics metrics
        
    Raises:
        FileNotFoundError: If audio file not found
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-6)
        
        # Calculate RMS energy in windows
        hop_length = int(sr * window_sec)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Dynamics metrics
        energy_range = float(np.max(rms) - np.min(rms))
        energy_std = float(np.std(rms))
        
        # Detect energy peaks (buildups/drops)
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
    Classify BGM mood based on audio features.
    
    Args:
        tempo: BPM
        percussive_ratio: Proportion of percussive energy
        spectral_centroid: Mean spectral centroid
        energy_variance: Variance in energy
        
    Returns:
        str: Mood description
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
