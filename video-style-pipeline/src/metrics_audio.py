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

# Required dependencies - raise error if missing
try:
    import librosa
    import soundfile as sf
except ImportError as e:
    logger.error(f"Required audio libraries not installed: {e}")
    logger.error("Install with: pip install librosa soundfile")
    raise ImportError("librosa and soundfile are required. Install with: pip install librosa soundfile")

# Optional: Essentia for advanced music analysis
try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    logger.warning("Essentia not available. Install for advanced music analysis: pip install essentia-tensorflow")


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
        
        # BGM style classification (heuristic)
        if percussive_ratio < 0.45 and mean_centroid < 2200:
            bgm_style = "Acoustic/lofi"
        else:
            bgm_style = "Electronic/beat-driven"
        
        # RMS energy over time
        rms = librosa.feature.rms(y=y)[0]
        mean_energy = float(np.mean(rms))
        energy_variance = float(np.var(rms))
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        mean_rolloff = float(np.mean(rolloff))
        
        # Mood classification
        mood = classify_bgm_mood(tempo, percussive_ratio, mean_centroid, energy_variance)
        
        # Instrument detection (using Essentia if available, else heuristic)
        instruments = detect_instruments(y, sr)
        
        result = {
            "tempo_bpm": float(tempo),
            "beat_times": beat_times.tolist() if hasattr(beat_times, 'tolist') else list(beat_times),
            "num_beats": len(beat_times),
            "percussive_ratio": percussive_ratio,
            "spectral_centroid": mean_centroid,
            "spectral_flatness": mean_flatness,
            "zero_crossing_rate": mean_zcr,
            "speech_ratio": speech_ratio,
            "bgm_style": bgm_style,
            "mood": mood,
            "instruments": instruments,
            "mean_energy": mean_energy,
            "energy_variance": energy_variance,
            "spectral_rolloff": mean_rolloff
        }
        
        logger.info(f"Audio metrics extracted successfully from {audio_path}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to extract audio metrics from {audio_path}: {e}")
        raise


def detect_instruments(y, sr):
    """
    Detect instruments in audio using Essentia (if available) or heuristic methods.
    
    Args:
        y: Audio signal
        sr: Sample rate
        
    Returns:
        dict: Instrument detection results
    """
    if ESSENTIA_AVAILABLE:
        try:
            logger.debug("Using Essentia for instrument detection")
            # Use Essentia Music Extractor
            extractor = es.MusicExtractor()
            features, _ = extractor(y)
            
            # Extract instrument-related features
            instruments_detected = []
            
            # Check for common instruments based on spectral features
            if features.get('lowlevel.mfcc.mean', None) is not None:
                mfcc = features['lowlevel.mfcc.mean']
                # Heuristic: different MFCC patterns indicate different instruments
                if len(mfcc) > 0:
                    # Piano: typically has strong low MFCCs
                    if mfcc[0] > 0.5:
                        instruments_detected.append("Piano")
                    # Guitar: mid-range MFCCs
                    if len(mfcc) > 5 and abs(mfcc[3]) > 0.3:
                        instruments_detected.append("Guitar")
            
            # Percussion detection
            if features.get('rhythm.beats_loudness.mean', None) is not None:
                if features['rhythm.beats_loudness.mean'] > 0.1:
                    instruments_detected.append("Drums/Percussion")
            
            if not instruments_detected:
                instruments_detected = ["Unknown"]
            
            return {
                "detected_instruments": instruments_detected,
                "method": "Essentia"
            }
        except Exception as e:
            logger.warning(f"Essentia instrument detection failed: {e}, falling back to heuristic")
    
    # Heuristic fallback
    logger.debug("Using heuristic method for instrument detection")
    instruments_detected = []
    
    # Analyze spectral characteristics
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(spectral_centroid)
    
    # Chroma features for harmonic content
    chroma = librosa.feature.chroma(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Percussion detection
    percussive_ratio = np.sum(np.abs(librosa.effects.hpss(y)[1])) / (np.sum(np.abs(y)) + 1e-6)
    if percussive_ratio > 0.4:
        instruments_detected.append("Drums/Percussion")
    
    # Harmonic instruments
    if mean_centroid < 2000:
        instruments_detected.append("Bass")
    elif mean_centroid < 3000:
        instruments_detected.append("Guitar/Piano")
    else:
        instruments_detected.append("High-frequency instruments")
    
    if not instruments_detected:
        instruments_detected = ["Unknown"]
    
    return {
        "detected_instruments": instruments_detected,
        "method": "Heuristic"
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
