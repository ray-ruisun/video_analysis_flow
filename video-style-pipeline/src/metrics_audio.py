#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio metrics extraction module.
Analyzes tempo (BPM), beat timing, energy distribution, and narration presence.
"""

import os
import numpy as np

# Optional dependency handling
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


def extract_audio_metrics(audio_path):
    """
    Extract comprehensive audio metrics from a wav file.
    
    Args:
        audio_path: Path to audio file (preferably 22.05kHz mono wav)
        
    Returns:
        dict: Audio metrics including tempo, beats, energy, speech ratio, etc.
    """
    if not AUDIO_AVAILABLE:
        return {
            "available": False,
            "error": "librosa not installed"
        }
    
    if not audio_path or not os.path.exists(audio_path):
        return {
            "available": False,
            "error": "Audio file not found"
        }
    
    try:
        # Load audio (force mono, 22.05kHz)
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
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
        # High flatness + moderate ZCR suggests speech-like content
        speech_ratio = float(np.clip(mean_flatness * 1.8 + mean_zcr * 0.8, 0, 1))
        
        # BGM style classification (heuristic)
        if percussive_ratio < 0.45 and mean_centroid < 2200:
            bgm_style = "Acoustic/lofi (proxy)"
        else:
            bgm_style = "Electronic/beat-driven (proxy)"
        
        # RMS energy over time
        rms = librosa.feature.rms(y=y)[0]
        mean_energy = float(np.mean(rms))
        energy_variance = float(np.var(rms))
        
        # Spectral rolloff (frequency below which 85% of energy lies)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        mean_rolloff = float(np.mean(rolloff))
        
        return {
            "available": True,
            "tempo_bpm": float(tempo),
            "beat_times": beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times),
            "num_beats": len(beat_times),
            "percussive_ratio": percussive_ratio,
            "spectral_centroid": mean_centroid,
            "spectral_flatness": mean_flatness,
            "zero_crossing_rate": mean_zcr,
            "speech_ratio": speech_ratio,
            "bgm_style": bgm_style,
            "mean_energy": mean_energy,
            "energy_variance": energy_variance,
            "spectral_rolloff": mean_rolloff
        }
        
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
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
        float or None: Proportion of cuts aligned with beats (0-1)
    """
    if num_cuts <= 0 or not beat_times or video_duration <= 0:
        return None
    
    # Approximate cut times (assuming uniform distribution)
    # This is a simplification; ideally cuts would have precise timestamps
    approx_cut_times = np.linspace(0, video_duration, num=num_cuts + 2)[1:-1]
    
    beat_array = np.array(beat_times)
    aligned_count = 0
    
    for cut_time in approx_cut_times:
        # Check if any beat is within tolerance
        if beat_array.size > 0:
            min_distance = np.min(np.abs(beat_array - cut_time))
            if min_distance <= tolerance:
                aligned_count += 1
    
    return aligned_count / len(approx_cut_times)


def analyze_energy_dynamics(audio_path, window_sec=5.0):
    """
    Analyze energy dynamics over time.
    
    Args:
        audio_path: Path to audio file
        window_sec: Window size for energy analysis
        
    Returns:
        dict: Energy dynamics metrics
    """
    if not AUDIO_AVAILABLE or not os.path.exists(audio_path):
        return {"available": False}
    
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
            "available": True,
            "energy_range": energy_range,
            "energy_std": energy_std,
            "num_energy_changes": num_peaks,
            "dynamic_style": "High dynamics" if energy_std > 0.1 else "Steady energy"
        }
        
    except Exception as e:
        return {"available": False, "error": str(e)}


def classify_bgm_mood(tempo, percussive_ratio, spectral_centroid, energy_variance):
    """
    Heuristic mood classification based on audio features.
    
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

