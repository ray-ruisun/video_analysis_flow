#!/usr/bin/env bash
# Quick-start script for video style analysis

set -e

# Check if videos exist
if [ ! -f "videos/v1.mp4" ] || [ ! -f "videos/v2.mp4" ] || [ ! -f "videos/v3.mp4" ]; then
    echo "ERROR: Please place v1.mp4, v2.mp4, v3.mp4 in the videos/ directory"
    exit 1
fi

echo "=== Video Style Analysis Pipeline ==="
echo ""

# Check dependencies
echo "Checking dependencies..."
if ! python -c "import numpy, cv2, librosa, docx" 2>/dev/null; then
    echo "ERROR: Core dependencies missing. Run: pip install -r requirements.txt"
    exit 1
fi

# Create work directory
mkdir -p work

# Extract audio (if ffmpeg available)
if command -v ffmpeg &> /dev/null; then
    echo ""
    echo "Extracting audio tracks..."
    for video in videos/v1.mp4 videos/v2.mp4 videos/v3.mp4; do
        base=$(basename "$video" .mp4)
        if [ ! -f "work/$base.wav" ]; then
            echo "  Processing $video..."
            ffmpeg -i "$video" -ar 22050 -ac 1 -y "work/$base.wav" -loglevel error
        else
            echo "  $base.wav already exists, skipping"
        fi
    done
    AUDIO_ARGS="--audios work/v1.wav work/v2.wav work/v3.wav"
else
    echo ""
    echo "WARNING: ffmpeg not found. Audio analysis will be skipped."
    echo "Install ffmpeg for full functionality: https://ffmpeg.org/download.html"
    AUDIO_ARGS=""
fi

# Check optional features
OPTIONAL_ARGS=""
if python -c "import ultralytics" 2>/dev/null; then
    echo "✓ YOLOv8 detected - enabling object detection"
    OPTIONAL_ARGS="$OPTIONAL_ARGS --enable-yolo"
fi

if python -c "import faster_whisper" 2>/dev/null || python -c "import whisper" 2>/dev/null; then
    echo "✓ Whisper detected - enabling ASR"
    OPTIONAL_ARGS="$OPTIONAL_ARGS --enable-asr"
fi

# Run analysis
echo ""
echo "Running analysis..."
python src/analyze.py \
    --videos videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \
    $AUDIO_ARGS \
    --report style_report.docx \
    --frames edge \
    $OPTIONAL_ARGS

echo ""
echo "=== Analysis Complete ==="
echo "Report saved to: style_report.docx"

