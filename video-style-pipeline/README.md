# Video Style Analysis Pipeline / 视频风格分析流水线

Research-grade pipeline for analyzing video style patterns: cinematography, editing, audio, and environment.

科研级视频风格分析流水线：镜头语言、剪辑节奏、音频特征、环境分析。

## Features / 功能

Analyzes 3 human-created videos to extract common stylistic patterns:

分析3个人类创作的视频，提取共同的风格模式：

- 🎥 **Camera & Composition** / 镜头与构图: angle, movement, framing
- 🎨 **Color & Lighting** / 色彩与光线: hues, white balance, contrast, CCT
- ✂️ **Editing & Pacing** / 剪辑与节奏: shot length, transitions, beat alignment
- 🎵 **Music & Audio** / 音乐与音频: tempo, energy, style, speech ratio
- 🏠 **Environment** / 环境: scene type, countertop, utensils (optional YOLO)
- 🗣️ **Narration** / 旁白: speech rate, catchphrases (optional Whisper)

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Optional Enhancements

**YOLOv8** (kitchen/utensils detection):
```bash
pip install ultralytics
```

**Whisper ASR** (narration analysis):
```bash
# Option 1: faster-whisper (recommended for speed)
pip install faster-whisper

# Option 2: OpenAI Whisper
pip install openai-whisper
```

**Advanced Scene Detection** (optional upgrades):
```bash
pip install scenedetect[opencv]
pip install essentia-tensorflow  # For advanced music analysis
```

## Usage

### Basic Usage

Place 3 human-created videos in the `videos/` directory, then:

```bash
./analyze.sh
```

Or use the Python script directly:

```bash
python src/analyze.py \
  --videos videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \
  --report output_report.docx
```

### With Audio Pre-extraction

For better audio analysis, extract audio first:

```bash
# Extract audio (22.05kHz mono wav)
ffmpeg -i videos/v1.mp4 -ar 22050 -ac 1 work/v1.wav
ffmpeg -i videos/v2.mp4 -ar 22050 -ac 1 work/v2.wav
ffmpeg -i videos/v3.mp4 -ar 22050 -ac 1 work/v3.wav

python src/analyze.py \
  --videos videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \
  --audios work/v1.wav work/v2.wav work/v3.wav \
  --report output_report.docx
```

### Enable Optional Features

```bash
python src/analyze.py \
  --videos videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \
  --enable-yolo \
  --enable-asr \
  --frames edge \
  --report output_report.docx
```

### Options

- `--videos`: 3 video paths (required)
- `--audios`: 3 pre-extracted wav files (optional, recommended)
- `--report`: Output Word document path (required)
- `--frames`: Contact sheet style: `mosaic`, `edge`, or `off` (default: `edge`)
- `--enable-yolo`: Enable YOLOv8 kitchen/utensils detection
- `--enable-asr`: Enable Whisper ASR for narration analysis

## Project Structure

```
video-style-pipeline/
├── README.md
├── requirements.txt
├── Makefile                    # Build/test/clean commands
├── analyze.sh                  # Quick-start bash script
├── videos/                     # Place your 3 videos here
│   ├── v1.mp4
│   ├── v2.mp4
│   └── v3.mp4
├── work/                       # Runtime: audio extracts, frames, cache
└── src/
    ├── analyze.py              # Main orchestration script
    ├── metrics_visual.py       # Visual/editing/color/white balance
    ├── metrics_audio.py        # BPM/beat/energy/narration ratio
    ├── metrics_asr.py          # (Optional) Whisper transcription
    ├── metrics_yolo.py         # (Optional) YOLOv8 detection
    └── report_word.py          # Word report generation
```

## Output

The pipeline generates a `.docx` report containing:
1. **Cross-video common elements**: majority consensus across 3 videos
2. **Per-video metrics**: detailed breakdown for each video
3. **Optional screenshots**: contact sheets showing key frames
4. **Upgrade suggestions**: recommendations for enhanced analysis

## Compliance & Ethics

- Analyze only **human-created content** (e.g., verified via Deepware)
- No portrait/face analysis
- No external copyrighted assets
- Suitable for research and style transfer applications

## Optional Enhancements / 可选增强

Want better accuracy? See **OPTIMIZATION.md** for detailed recommendations:

想要更高准确率？查看 **OPTIMIZATION.md** 获取详细升级建议：

**High Priority Tools / 高优先级工具:**
- 🔴 **PySceneDetect**: 95% shot detection accuracy (vs. 70% current)
- 🔴 **Essentia**: 100+ music features (vs. 5 current)  
- 🔴 **Pyannote**: Speaker diarization (who speaks when)
- 🔴 **Places365**: Scene classification (365 scene types)

**Medium Priority / 中优先级:**
- 🟡 **Demucs**: Audio source separation
- 🟡 **OpenSMILE**: 6000+ prosody features
- 🟡 **Madmom**: Better beat tracking

Expected improvement: 50-95% analysis depth / 预期提升：50-95%分析深度

## Documentation / 文档

- **README.md** (this file) - Main documentation / 主文档
- **QUICKSTART.md** - 5-minute tutorial / 5分钟教程
- **OPTIMIZATION.md** - Upgrade guide / 升级指南

## References / 参考

**Core Dependencies:** OpenCV, NumPy, librosa, python-docx, Pillow  
**Optional:** YOLOv8, Whisper, loguru, tqdm  
**Optimization:** See OPTIMIZATION.md for advanced tools

## License

MIT License - Research and educational use

## Citation

If you use this pipeline in your research, please cite appropriately and ensure compliance with all component licenses.

