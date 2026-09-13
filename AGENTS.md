# Bird Detect Pipeline

AI-powered bird detection pipeline using YOLO11l. Detects birds in video files, groups detections into temporal series, and generates DaVinci Resolve-compatible FCPXML v1.9 timelines.

## Architecture

```
Video → bird.py (YOLO11l) → bird.csv → bird.resolve.py → timeline.fcpxml → DaVinci Resolve
```

## Components

| File | Purpose |
|---|---|
| `bird.py` | Runs YOLO11l on video frame-by-frame, saves detected bird frames to CSV |
| `bird.resolve.py` | Reads CSV, groups detections, generates FCPXML v1.9 for DaVinci Resolve |
| `requirements.txt` | Python dependencies (includes PyTorch CUDA 12.1) |
| `install-cuda.bat` | Bootstrap script to install PyTorch + all dependencies |

## Quick Start

### 1. Install

```bash
# Full install (PyTorch CUDA 12.1 + dependencies)
E:/yolo/install-cuda.bat

# Dependencies only (if PyTorch already installed)
E:/yolo/Scripts/pip.exe install -r requirements.txt
```

### 2. Detect birds

```bash
E:/yolo/Scripts/python.exe bird.py "C:/PATH/TO/video.mp4"
```

Outputs `bird.csv` next to the video:

```csv
frame_number,video_path
120,C:\PATH\TO\video.mp4
121,C:\PATH\TO\video.mp4
...
```

### 3. Generate Resolve timeline

```bash
E:/yolo/Scripts/python.exe bird.resolve.py bird.csv
```

Outputs `timeline.fcpxml` in the script directory.

## Configuration

Editable constants in `bird.resolve.py`:

| Constant | Default | Description |
|---|---|---|
| `MIN_FRAMES` | 10 | Minimum detected frames to form a series |
| `SERIES_GAP` | 60 | Gap threshold (frames) to split series |
| `PRE_ROLL` | 2.0 | Seconds before first detection |
| `POST_ROLL` | 2.0 | Seconds after last detection |

## How It Works

### bird.py

1. Opens video with OpenCV
2. Runs YOLO11l inference on each frame (`class=14` → COCO "bird")
3. Checks `len(results[0].boxes) > 0`
4. Appends detected frame numbers to CSV

### bird.resolve.py

1. Reads `bird.csv` (frame numbers + video paths)
2. Groups detections by file, splits into series when gap > `SERIES_GAP`
3. Adds pre/post-roll, merges overlapping ranges
4. Queries ffprobe for FPS (Fraction) and duration
5. Converts to FCPXML v1.9 with standard SMPTE timeline FPS
6. All `<asset-clip>` times in `N/fpsNum s` format (timeline frames)

Key Resolve-specific fixes:
- Timeline FPS normalised to SMPTE standards (5994→60000/1001, 30p, 24p...)
- `file://` URIs use raw paths (no `%20` URL encoding — breaks on Windows Resolve)
- `<asset>` has no `start="0s"` — avoids conflicts with camera embedded timecodes
- All durations use exact float from ffprobe (no integer rounding)

## Requirements

- **Python 3.11+**
- **FFmpeg** — portable at `E:/PORTABLE/ffmpeg/bin/`
- **NVIDIA GPU** — PyTorch CUDA 12.1
- **DaVinci Resolve** — to import the FCPXML

## Troubleshooting

- **Media Offline in Resolve**: Import source files into Media Pool manually first, then import the XML
- **ffprobe not found**: Verify portable FFmpeg exists at `E:/PORTABLE/ffmpeg/bin/`
- **No series generated**: Lower `MIN_FRAMES` — may not have enough consecutive detections

## Project Structure

```
E:/yolo/
├── bird.py              # Bird detection
├── bird.resolve.py      # FCPXML generation
├── requirements.txt     # Dependencies (CUDA)
├── install-cuda.bat     # Install script
├── Scripts/             # Python venv (3.11)
└── *.pt                 # YOLO models (auto-download on first use)
```
