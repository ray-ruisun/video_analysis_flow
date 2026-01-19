.PHONY: help install install-full clean test run web

help:
	@echo "Video Style Analysis Pipeline - Makefile Commands"
	@echo ""
	@echo "  make install       - Install basic dependencies"
	@echo "  make install-full  - Install all dependencies (including optional)"
	@echo "  make web           - Launch Gradio Web UI (http://localhost:7860)"
	@echo "  make run           - Run analysis on videos/ (basic mode)"
	@echo "  make run-full      - Run analysis with all features enabled"
	@echo "  make extract-audio - Extract audio from videos to work/"
	@echo "  make clean         - Remove work/ and generated files"
	@echo "  make test          - Run basic sanity checks"

install:
	pip install -r requirements.txt

install-full:
	pip install -r requirements.txt
	pip install ultralytics faster-whisper scenedetect[opencv]

web:
	@echo "🌐 Starting Gradio Web UI..."
	@echo "📍 Open http://localhost:7860 in your browser"
	python app.py

extract-audio:
	@mkdir -p work
	@for video in videos/*.mp4; do \
		base=$$(basename "$$video" .mp4); \
		echo "Extracting audio from $$video..."; \
		ffmpeg -i "$$video" -ar 22050 -ac 1 -y "work/$$base.wav" -loglevel error; \
	done
	@echo "Audio extraction complete."

run:
	@if [ ! -f videos/v1.mp4 ] || [ ! -f videos/v2.mp4 ] || [ ! -f videos/v3.mp4 ]; then \
		echo "ERROR: Please place v1.mp4, v2.mp4, v3.mp4 in videos/ directory"; \
		exit 1; \
	fi
	python main.py \
		--videos videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \
		--output style_report.docx \
		--frames edge

run-full: extract-audio
	@if [ ! -f videos/v1.mp4 ] || [ ! -f videos/v2.mp4 ] || [ ! -f videos/v3.mp4 ]; then \
		echo "ERROR: Please place v1.mp4, v2.mp4, v3.mp4 in videos/ directory"; \
		exit 1; \
	fi
	python main.py \
		--videos videos/v1.mp4 videos/v2.mp4 videos/v3.mp4 \
		--audios work/v1.wav work/v2.wav work/v3.wav \
		--output style_report_full.docx \
		--frames edge \
		--yolo \
		--asr

clean:
	rm -rf work/*
	rm -f *.docx
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

test:
	@echo "Running sanity checks..."
	@python -c "import numpy, cv2, librosa, docx; print('✓ Core dependencies OK')"
	@if python -c "import ultralytics" 2>/dev/null; then echo "✓ YOLOv8 available"; else echo "✗ YOLOv8 not installed (optional)"; fi
	@if python -c "import faster_whisper" 2>/dev/null; then echo "✓ faster-whisper available"; elif python -c "import whisper" 2>/dev/null; then echo "✓ OpenAI Whisper available"; else echo "✗ ASR not installed (optional)"; fi
	@echo "✓ All checks passed"

