"""
speed_test_onnx.py
==================
Script 2 (correct perspective homography + CSV export)
  + Script 1 (ONNX DirectML inference + infer_every skipping)

FPS boost strategy:
  1. ONNX model via ONNXRuntime DirectML  -> runs on RX 580 GPU
     (auto-falls back to CPU if onnxruntime-directml not installed)
  2. infer_every N                        -> skip inference on intermediate frames
  3. actual video FPS fed to ByteTrack    -> correct tracking
  4. Script 2 ViewTransformer math        -> correct speed (not flat mpp)

To enable GPU on RX 580:
  pip uninstall onnxruntime
  pip install onnxruntime-directml
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np
import supervision as sv
import onnxruntime as ort
from collections import defaultdict, deque


# =============================================================
# SOURCE quads  (all three presets from original script 2)
# =============================================================
SOURCE_PRESETS = {
    "long":   np.array([[38, 56],   [345, 9],  [1207, 367], [370, 715]], dtype=np.int32),
    "short":  np.array([[140, 230], [661, 132], [1207, 367], [370, 715]], dtype=np.int32),
    "middle": np.array([[128, 151], [461, 90],  [1137, 397], [370, 715]], dtype=np.int32),
}

TARGET_WIDTH  = 4   # metres  (real-world lane width)
TARGET_HEIGHT = 17  # metres  (real-world gate-to-gate distance)

TARGET = np.array(
    [
        [0,               0],
        [TARGET_WIDTH-1,  0],
        [TARGET_WIDTH-1,  TARGET_HEIGHT-1],
        [0,               TARGET_HEIGHT-1],
    ],
    dtype=np.float32,
)


# =============================================================
# Perspective transformer  (identical to Script 2)
# =============================================================
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.m).reshape(-1, 2)


# =============================================================
# ONNX helpers
# =============================================================
def letterbox(im, new_shape, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = new_shape - nw
    pad_h = new_shape - nh
    left,  right  = pad_w // 2, pad_w - pad_w // 2
    top,   bottom = pad_h // 2, pad_h - pad_h // 2
    padded = cv2.copyMakeBorder(
        im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, r, left, top


def iou_xyxy(a, b):
    xx1 = np.maximum(a[0], b[:, 0])
    yy1 = np.maximum(a[1], b[:, 1])
    xx2 = np.minimum(a[2], b[:, 2])
    yy2 = np.minimum(a[3], b[:, 3])
    inter  = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, scores, iou_th=0.5):
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious  = iou_xyxy(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_th]
    return keep


def make_session(onnx_path: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 2

    available = ort.get_available_providers()
    if "DmlExecutionProvider" in available:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        print("[ORT] WARNING: DmlExecutionProvider not found - running on CPU.")
        print("      To unlock RX 580 GPU, run in your venv:")
        print("        pip uninstall onnxruntime")
        print("        pip install onnxruntime-directml")
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    print(f"[ORT] Providers active: {sess.get_providers()}")
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    return sess, inp, out


def onnx_detect(sess, inp_name, out_name, frame_bgr, img_size, conf_th, iou_th, topk):
    img, r, pad_x, pad_y = letterbox(frame_bgr, img_size)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]

    y = sess.run([out_name], {inp_name: x})[0]
    y = np.squeeze(y)
    if y.ndim == 2 and y.shape[0] < y.shape[1]:
        y = y.T

    xywh = y[:, 0:4]
    conf = y[:, 4:].max(axis=1)
    m    = conf >= conf_th

    if not np.any(m):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    xywh = xywh[m]
    conf = conf[m]

    cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    boxes = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)

    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= r

    H, W = frame_bgr.shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H - 1)

    if conf.size > topk:
        idx   = np.argsort(conf)[::-1][:topk]
        boxes = boxes[idx]
        conf  = conf[idx]

    keep = nms(boxes, conf, iou_th)
    return boxes[keep].astype(np.float32), conf[keep].astype(np.float32)


def to_sv_detections(boxes, conf):
    if boxes.shape[0] == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=boxes,
        confidence=conf,
        class_id=np.zeros(len(boxes), dtype=int),
    )


# =============================================================
# Zone filter  (identical to Script 2)
# =============================================================
def in_zone_mask(dets: sv.Detections, polygon: np.ndarray) -> np.ndarray:
    if len(dets) == 0:
        return np.array([], dtype=bool)
    cx  = (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2.0
    cy  = (dets.xyxy[:, 1] + dets.xyxy[:, 3]) / 2.0
    pts = np.stack([cx, cy], axis=1).astype(np.float32)
    return np.array(
        [cv2.pointPolygonTest(polygon, (float(p[0]), float(p[1])), False) >= 0
         for p in pts],
        dtype=bool,
    )


def safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# =============================================================
# CLI
# =============================================================
def parse_arguments():
    p = argparse.ArgumentParser(
        description="NOVA Speed Estimation - ONNX DirectML + homography"
    )
    p.add_argument("--source_video_path", required=True)
    p.add_argument(
        "--onnx_path",
        default=r"G:\cla_projects\NOVA Lowlight Speed Detection Dashboard\yolo11nano_1k_320\best.onnx",
    )
    p.add_argument("--source_preset",    default="middle",
                   choices=list(SOURCE_PRESETS.keys()),
                   help="Which SOURCE quad to use (long / short / middle)")
    p.add_argument("--img_size",         default=320,  type=int)
    p.add_argument("--conf_th",          default=0.40, type=float)
    p.add_argument("--iou_th",           default=0.50, type=float)
    p.add_argument("--topk",             default=120,  type=int)
    p.add_argument("--infer_every",      default=2,    type=int,
                   help="Run ONNX inference every N frames")
    p.add_argument("--out_csv",          default="speed_summary.csv")
    p.add_argument("--out_samples_csv",  default="")
    p.add_argument("--playback_speed",   default=5.0,  type=float)
    p.add_argument("--no_display",       action="store_true",
                   help="Skip cv2.imshow for maximum throughput")
    return p.parse_args()


# =============================================================
# Main
# =============================================================
def main():
    args = parse_arguments()

    if not os.path.exists(args.source_video_path):
        raise FileNotFoundError(f"Video not found: {args.source_video_path}")
    if not os.path.exists(args.onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_path}")

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    fps        = float(video_info.fps)
    print(f"[Video] {video_info.resolution_wh}  {fps:.2f} fps")

    sess, inp_name, out_name = make_session(args.onnx_path)

    byte_track = sv.ByteTrack(frame_rate=fps)   # actual fps, not hardcoded 30

    thickness  = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_ann    = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
    lbl_ann    = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK,
    )
    trace_ann  = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(fps * 2),
        position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK,
    )

    SOURCE           = SOURCE_PRESETS[args.source_preset]
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates   = defaultdict(lambda: deque(maxlen=int(fps)))
    speed_samples = defaultdict(list)

    delay_ms    = max(1, int((1000 / fps) / max(0.05, args.playback_speed)))
    frame_idx   = 0
    last_det    = sv.Detections.empty()
    last_inf_ms = 0.0
    t0_wall     = time.perf_counter()
    frames_done = 0

    for frame in sv.get_video_frames_generator(args.source_video_path):

        # ONNX inference every N frames
        if frame_idx % args.infer_every == 0:
            t0 = time.perf_counter()
            boxes, conf = onnx_detect(
                sess, inp_name, out_name,
                frame, args.img_size, args.conf_th, args.iou_th, args.topk,
            )
            last_inf_ms = (time.perf_counter() - t0) * 1000.0
            last_det    = to_sv_detections(boxes, conf)

        # FIX: index-slice last_det to get an independent object so
        # zone filtering never mutates last_det across skipped infer frames
        if len(last_det) > 0:
            dets = last_det[np.arange(len(last_det))]
        else:
            dets = sv.Detections.empty()

        mask = in_zone_mask(dets, SOURCE)
        dets = dets[mask]
        dets = byte_track.update_with_detections(detections=dets)

        # Homography transform  (Script 2 math — unchanged)
        points = dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points).astype(int)

        if dets.tracker_id is not None and len(dets.tracker_id) > 0:
            for tid, (_, y) in zip(dets.tracker_id, points):
                coordinates[int(tid)].append(int(y))

        # Speed labels
        labels = []
        warmup = fps / 2
        if dets.tracker_id is not None and len(dets.tracker_id) > 0:
            for tid in dets.tracker_id:
                tid  = int(tid)
                hist = coordinates[tid]
                if len(hist) < warmup:
                    labels.append(f"#{tid}")
                else:
                    dist   = abs(int(hist[-1]) - int(hist[0]))
                    time_s = len(hist) / fps
                    speed  = (dist / time_s) * 3.6
                    speed_samples[tid].append(float(speed))
                    labels.append(f"#{tid} {int(speed)} km/h")

        # Display
        if not args.no_display:
            ann = frame.copy()
            sv.draw_polygon(ann, polygon=SOURCE, color=sv.Color.RED)
            ann = trace_ann.annotate(scene=ann, detections=dets)
            ann = box_ann.annotate(scene=ann, detections=dets)
            ann = lbl_ann.annotate(scene=ann, detections=dets, labels=labels)

            elapsed  = time.perf_counter() - t0_wall
            live_fps = frames_done / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                ann,
                f"FPS:{live_fps:.1f}  infer:{last_inf_ms:.0f}ms  every:{args.infer_every}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            cv2.imshow("NOVA Speed (ONNX)", ann)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break

        frame_idx   += 1
        frames_done += 1

    cv2.destroyAllWindows()

    elapsed_total = time.perf_counter() - t0_wall
    avg_fps = frames_done / elapsed_total if elapsed_total > 0 else 0
    print(f"\n[Done] {frames_done} frames in {elapsed_total:.1f}s -> avg {avg_fps:.1f} FPS")
    print("\n=== Speed Summary per Vehicle (km/h) ===")

    summary_rows = []
    for tid in sorted(speed_samples.keys()):
        s = speed_samples[tid]
        if not s:
            continue
        row = dict(
            tracker_id=tid, n_samples=len(s),
            min_kmh=min(s), mean_kmh=safe_mean(s), max_kmh=max(s),
        )
        summary_rows.append(row)
        print(f"ID {tid:>3} | n={len(s):>4} | "
              f"min={min(s):7.2f} | mean={safe_mean(s):7.2f} | max={max(s):7.2f}")

    if summary_rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["tracker_id", "n_samples", "min_kmh", "mean_kmh", "max_kmh"]
            )
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSaved: {os.path.abspath(args.out_csv)}")
    else:
        print("\nNo speed samples collected.")

    if args.out_samples_csv:
        with open(args.out_samples_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tracker_id", "sample_index", "speed_kmh"])
            for tid in sorted(speed_samples.keys()):
                for i, s in enumerate(speed_samples[tid]):
                    w.writerow([tid, i, f"{s:.6f}"])
        print(f"Saved samples: {os.path.abspath(args.out_samples_csv)}")


if __name__ == "__main__":
    main()