# NOVA — Lowlight Vehicle Speed Detection Dashboard

```
███╗   ██╗ ██████╗ ██╗   ██╗ █████╗
████╗  ██║██╔═══██╗██║   ██║██╔══██╗
██╔██╗ ██║██║   ██║██║   ██║███████║
██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║
██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║
╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝

Nocturnal Object & Vehicle Analysis
Final Research Deliverable · 2026
```

> **This repository represents the final implementation phase of the NOVA research project** —
> a real-time, lowlight-capable vehicle speed detection system with a live dashboard UI,
> automated speeding logs, and ONNX-accelerated inference.

---

## Authorship

| Field       | Detail                         |
|-------------|--------------------------------|
| **Codebase Author**  | Clarence Jay Fetalino          |
| **Year**    | 2026                           |
| **Phase**   | Final — UI + Deployment Layer  |
| **Project** | NOVA Lowlight Speed Detection  |

---

## What NOVA Is

NOVA is a computer vision research system designed to detect and measure vehicle speeds
under **lowlight and nighttime conditions** — scenarios where conventional speed detection
systems fail or require expensive infrastructure.

The system uses a fine-tuned YOLOv11 model trained on 1,000+ lowlight vehicle samples,
combined with a perspective homography pipeline for accurate real-world speed estimation,
and a ByteTrack multi-object tracker for stable vehicle identity across frames.

This repository is the **final chapter** of the NOVA research — the live deployment
interface that brings together all prior work into an operational dashboard.

---

## Research Progression

```
Phase 1 · Data Collection & Annotation
        └─ Lowlight video dataset, manual bounding box labels

Phase 2 · Model Training
        └─ YOLOv11 nano/large fine-tuned on NOVA dataset
           Exported to ONNX for hardware-agnostic inference

Phase 3 · Speed Estimation Pipeline
        └─ Perspective homography (ViewTransformer)
           ByteTrack integration
           Gate-based calibration (SOURCE polygon → real-world metres)

Phase 4 · Offline Validation  [speed_test.py]
        └─ Video file processing, CSV export, playback speed control
           Proved accuracy of homography math vs flat mpp approximation

Phase 5 · Live Dashboard UI  [nova_live_app.py]  ◄ THIS REPO
        └─ Streamlit interface, live camera input, speeding logs
           ONNX DirectML acceleration for AMD GPU (RX 580)
           Camera discovery (USB, OBS Virtual, capture cards)
```

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    NOVA LIVE DASHBOARD                  │
│                   (nova_live_app.py)                    │
├──────────────┬──────────────────────┬───────────────────┤
│  Camera Feed │   Inference Engine   │   Speed Panel     │
│  (threaded)  │   ONNX + DirectML    │   Live km/h       │
│  CAP_DSHOW   │   YOLOv11 nano 320px │   ByteTrack IDs   │
│  USB / OBS   │   infer_every N      │   Speeding alerts │
│  Virtual cam │   NMS + zone filter  │   CSV export      │
└──────────────┴──────────────────────┴───────────────────┘
```

### Core Speed Math

Speed is derived from a **perspective homography**, not a flat pixel-to-metre approximation:

```
SOURCE polygon (pixel quad)  ──►  ViewTransformer  ──►  TARGET (real-world metres)
                                  getPerspectiveTransform
                                  4 × 17 m lane

Bottom-centre anchor  ──►  transform_points()  ──►  y_transformed (metres)

speed (km/h) = ( |y_end − y_start| / time_seconds ) × 3.6
```

This approach correctly accounts for perspective distortion across the full field of view.

---

## Files

| File | Description |
|------|-------------|
| `nova_live_app.py` | **Main deliverable.** Streamlit live dashboard — camera input, real-time speed detection, speeding logs, CSV export |
| `speed_test_onnx.py` | Offline video processor — ONNX inference, homography speed math, summary + raw sample CSV output |
| `speed_test.py` | Original validation script — PyTorch `.pt` model, confirmed homography math correctness |
| `yolo11nano_1k_320/best.onnx` | Exported ONNX model — YOLOv11 nano, trained on NOVA lowlight dataset, 320px input |

---

## Features

### Live Dashboard (`nova_live_app.py`)
- **Camera discovery** — scans indices 0–7 via `CAP_DSHOW`, detects USB cameras,
  OBS Virtual Camera, NDI sources, capture cards, any device OpenCV can open
- **Live speed readout** per tracked vehicle with colour-coded alerts (red above 80 km/h)
- **Speeding log CSV** — downloadable at any point during or after a session
- **ROI polygon overlay** — visual confirmation of the active detection zone
- **Vehicle trace** — trajectory history per tracked ID
- **ONNX DirectML** — GPU-accelerated inference on AMD hardware (RX 580);
  graceful CPU fallback with install instructions if DirectML is unavailable
- **`infer_every N`** — configurable inference skipping for FPS headroom
- **Three SOURCE presets** — `long`, `short`, `middle` — switchable from the sidebar
  without code edits

### Offline Processor (`speed_test_onnx.py`)
- Identical homography math to the live app
- Summary CSV: `tracker_id | n_samples | min_kmh | mean_kmh | max_kmh`
- Optional raw samples CSV for per-frame analysis
- `--no_display` flag for maximum throughput (headless processing)
- `--playback_speed` for faster-than-realtime review

---

## Setup

### Requirements

```
python >= 3.10
streamlit
opencv-python
numpy
supervision
onnxruntime          # CPU baseline
onnxruntime-directml # AMD/Intel GPU (replaces onnxruntime)
```

### Install

```bash
# Create and activate venv
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

# Install dependencies
pip install streamlit opencv-python numpy supervision onnxruntime

# For RX 580 / AMD GPU acceleration (Windows only):
pip uninstall onnxruntime
pip install onnxruntime-directml
```

### Run Live Dashboard

```bash
streamlit run nova_live_app.py --server.runOnSave true
```

### Run Offline Processor

```bash
python speed_test_onnx.py \
  --source_video_path "videos/20kmh.mp4" \
  --onnx_path "yolo11nano_1k_320/best.onnx" \
  --source_preset middle \
  --infer_every 2 \
  --conf_th 0.40 \
  --img_size 320 \
  --out_csv speed_summary.csv
```

---

## Calibration

The system requires a one-time calibration per camera installation.

**SOURCE polygon** — four pixel coordinates forming a quadrilateral that maps to a known
real-world area in the scene. Edit `SOURCE_PRESETS` in either script, or use the
`--source_preset` flag to choose between pre-configured setups.

**TARGET dimensions** — `TARGET_WIDTH = 4` (metres) and `TARGET_HEIGHT = 17` (metres)
define the real-world size of the SOURCE quad. Measure the actual lane width and
gate-to-gate distance at your installation site and update these values for accurate speeds.

```
Current presets (pixel coordinates):

  long   · [[38,56],  [345,9],  [1207,367], [370,715]]
  short  · [[140,230],[661,132],[1207,367],  [370,715]]
  middle · [[128,151],[461,90], [1137,397],  [370,715]]
```

---

## Performance

| Hardware | Model | img_size | infer_every | FPS (approx) |
|----------|-------|----------|-------------|--------------|
| RX 580 (DirectML) | YOLOv11 nano | 320 | 2 | ~20–25 |
| CPU only | YOLOv11 nano | 320 | 2 | ~12–18 |
| CPU only | YOLOv11 nano | 320 | 1 | ~8–12  |

> Install `onnxruntime-directml` to move inference off the CPU onto AMD/Intel GPUs on Windows.

---

## Known Limitations

- Speed accuracy depends on correct `TARGET_WIDTH` / `TARGET_HEIGHT` calibration for the
  specific installation site. Uncalibrated values will produce proportionally scaled errors.
- The `coordinates` deque uses a 1-second rolling window. Vehicles that traverse the zone
  faster than the warmup period (`fps / 2`) will not generate a speed reading.
- ByteTrack ID reassignment can occur when a vehicle temporarily leaves the ROI polygon.
- DirectML is Windows-only. Linux/macOS users fall back to CPU.

---

## License

This project is part of the NOVA research study.  
All rights reserved · Clarence Jay Fetalino · 2026

---

```
◈ END OF NOVA RESEARCH · FINAL DELIVERABLE
  Lowlight Vehicle Speed Detection Dashboard
  Clarence Jay Fetalino · 2026
```
