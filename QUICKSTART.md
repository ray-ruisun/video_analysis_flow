# Quick Start Guide

Get started with the Video Style Analysis Pipeline in 5 minutes.

## Prerequisites

- Python 3.8+
- ffmpeg (optional, for audio extraction)

## Installation

```bash
cd video-style-pipeline

# Install core dependencies
pip install -r requirements.txt

# (Optional) Install YOLOv8 for object detection
pip install ultralytics

# (Optional) Install Whisper for speech analysis
pip install faster-whisper
```

## Basic Usage (3 Steps)

### Step 1: Add Videos

Place 3 human-verified videos in the `videos/` directory:

```bash
cp /path/to/your/video1.mp4 videos/v1.mp4
cp /path/to/your/video2.mp4 videos/v2.mp4
cp /path/to/your/video3.mp4 videos/v3.mp4
```

### Step 2: Run Analysis

**Easy way** (using bash script):
```bash
./analyze.sh
```

**Manual way** (using Python directly):
```bash
python src/analyze.py \
  --videos videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \
  --report style_report.docx
```

### Step 3: View Report

Open `style_report.docx` in Microsoft Word, LibreOffice, or Google Docs.

## Advanced Usage

### Extract Audio First (Recommended)

For better audio analysis, pre-extract audio using ffmpeg:

```bash
make extract-audio
# Or manually:
ffmpeg -i videos/v1.mp4 -ar 22050 -ac 1 work/v1.wav
ffmpeg -i videos/v2.mp4 -ar 22050 -ac 1 work/v2.wav
ffmpeg -i videos/v3.mp4 -ar 22050 -ac 1 work/v3.wav
```

Then run with audio:
```bash
python src/analyze.py \
  --videos videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \
  --audios work/v1.wav work/v2.wav work/v3.wav \
  --report style_report.docx
```

### Enable All Features

```bash
python src/analyze.py \
  --videos videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \
  --audios work/v1.wav work/v2.wav work/v3.wav \
  --enable-yolo \
  --enable-asr \
  --frames edge \
  --report style_report_full.docx
```

### Using Makefile

```bash
# Install dependencies
make install

# Run basic analysis
make run

# Run full analysis (with all features)
make run-full

# Clean temporary files
make clean

# Check dependencies
make test
```

## Output

The pipeline generates a Word document (`style_report.docx`) containing:

1. **Cross-video consensus**: Common stylistic patterns across all videos
   - Camera angles and composition
   - Color palette and lighting
   - Editing pace and rhythm
   - Music style and tempo
   - Environmental elements

2. **Per-video metrics**: Detailed breakdown for each video
   - Visual characteristics
   - Audio analysis
   - Narration style (if ASR enabled)
   - Object detection (if YOLO enabled)
   - Screenshot contact sheets

3. **Technical notes**: Recommendations for advanced analysis

## Troubleshooting

### "librosa not installed"
```bash
pip install librosa soundfile
```

### "YOLOv8 not available"
```bash
pip install ultralytics
```

### "Whisper not available"
```bash
pip install faster-whisper
# or
pip install openai-whisper
```

### "ffmpeg not found"
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Import errors
```bash
# Make sure you're in the video-style-pipeline directory
cd video-style-pipeline

# Run from project root
python src/analyze.py --help
```

## Performance Tips

1. **Audio**: Pre-extract audio with ffmpeg for faster processing
2. **Frames**: Use `--frames off` to skip screenshot generation
3. **Sampling**: Large videos are automatically downsampled
4. **ASR**: Whisper is slow; expect 1-2 minutes per video
5. **YOLO**: First run downloads model (~6MB)

## Examples

### Cooking Videos
```bash
python src/analyze.py \
  --videos cooking1.mp4 cooking2.mp4 cooking3.mp4 \
  --enable-yolo \
  --report cooking_style.docx
```

### Tutorial Videos
```bash
python src/analyze.py \
  --videos tutorial1.mp4 tutorial2.mp4 tutorial3.mp4 \
  --enable-asr \
  --report tutorial_style.docx
```

### Product Reviews
```bash
python src/analyze.py \
  --videos review1.mp4 review2.mp4 review3.mp4 \
  --enable-yolo --enable-asr \
  --report review_style.docx
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [Makefile](Makefile) for available commands
- Explore individual modules in `src/` for customization
- See LICENSE for usage terms

## Support

For issues or questions:
1. Check that all dependencies are installed: `make test`
2. Verify video files are valid: `ffmpeg -i videos/v1.mp4`
3. Run with verbose Python errors: `python -u src/analyze.py ...`

Happy analyzing! 🎥📊

