import argparse
import csv
import os
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from collections import defaultdict, deque

"""
# LONG DIST
SOURCE = np.array([
    [38, 56],
    [345, 9],
    [1207, 367],
    [370, 715]
], dtype=np.int32)
"""

"""
# SHORT DIST
SOURCE = np.array([
    [140, 230],
    [661, 132],
    [1207, 367],
    [370, 715]
], dtype=np.int32)
"""

# MIDDLE DIST
SOURCE = np.array([
    [128, 151],
    [461, 90],
    [1137, 397],
    [370, 715]
], dtype=np.int32)

TARGET_WIDTH = 4
TARGET_HEIGHT = 17

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using YOLO + Supervision + ByteTrack"
    )

    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )

    parser.add_argument(
        "--model_path",
        default=r"G:\cla_projects\NOVA\onnx\yolo11l_1k_320\best.pt",
        help="Path to YOLO .pt model",
        type=str,
    )

    parser.add_argument(
        "--out_csv",
        default="speed_summary.csv",
        type=str,
        help="Output CSV for per-vehicle summary (min/max/mean km/h).",
    )

    parser.add_argument(
        "--out_samples_csv",
        default="",
        type=str,
        help="Optional output CSV for raw speed samples (leave empty to skip).",
    )

    parser.add_argument(
        "--playback_speed",
        default=5.0,
        type=float,
        help="Playback speed factor for display window. 1.0=real-time, 2.0=faster, 0.5=slower",
    )

    return parser.parse_args()


def in_zone_mask(dets: sv.Detections, polygon: np.ndarray) -> np.ndarray:
    """
    Keep detections whose center point lies inside polygon.
    """
    if len(dets) == 0:
        return np.array([], dtype=bool)

    xyxy = dets.xyxy
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
    pts = np.stack([cx, cy], axis=1).astype(np.float32)

    return np.array(
        [cv2.pointPolygonTest(polygon, tuple(p), False) >= 0 for p in pts],
        dtype=bool
    )


def safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists(args.source_video_path):
        raise FileNotFoundError(f"Video not found: {args.source_video_path}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = YOLO(args.model_path)

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    bounding_box_annotator = sv.BoxAnnotator(
        thickness=thickness,
        color_lookup=sv.ColorLookup.TRACK
    )

    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK
    )

    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=int(video_info.fps * 2),
        position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK
    )

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    coordinates = defaultdict(lambda: deque(maxlen=int(video_info.fps)))  # store last ~1 second of y positions

    # speed collection
    speed_samples = defaultdict(list)  # tracker_id -> list of speed_kmh estimates

    # display delay control
    playback_speed = max(0.05, float(args.playback_speed))
    delay_ms = max(1, int((1000 / video_info.fps) / playback_speed))

    for frame in frame_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        # filter by polygon zone
        mask = in_zone_mask(detections, SOURCE)
        detections = detections[mask]

        # track
        detections = byte_track.update_with_detections(detections=detections)

        # anchor points (bottom center) -> transform
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        points = view_transformer.transform_points(points=points).astype(int)

        # store y history per tracker
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            for tracker_id, (_, y) in zip(detections.tracker_id, points):
                coordinates[int(tracker_id)].append(int(y))

        # labels (speed)
        labels = []
        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            for tracker_id in detections.tracker_id:
                tid = int(tracker_id)
                if len(coordinates[tid]) < video_info.fps / 2:
                    labels.append(f"#{tid}")
                else:
                    # NOTE: This is based on transformed units, not real meters unless calibrated.
                    coordinate_start = coordinates[tid][-1]
                    coordinate_end = coordinates[tid][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time_s = len(coordinates[tid]) / video_info.fps
                    speed_kmh = (distance / time_s) * 3.6

                    speed_samples[tid].append(float(speed_kmh))
                    labels.append(f"#{tid} {int(speed_kmh)} km/h")
        else:
            labels = []

        # annotate
        annotated_frame = frame.copy()
        sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)

        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    # ---- SUMMARY (end of video) ----
    print("\n=== Speed Summary per Vehicle (km/h) ===")

    summary_rows = []
    for tid in sorted(speed_samples.keys()):
        samples = speed_samples[tid]
        if not samples:
            continue

        mn = min(samples)
        mx = max(samples)
        mean = safe_mean(samples)
        n = len(samples)

        summary_rows.append({
            "tracker_id": tid,
            "n_samples": n,
            "min_kmh": mn,
            "mean_kmh": mean,
            "max_kmh": mx
        })

        print(f"ID {tid:>3} | n={n:>4} | min={mn:7.2f} | mean={mean:7.2f} | max={mx:7.2f}")

    # write summary CSV
    if summary_rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["tracker_id", "n_samples", "min_kmh", "mean_kmh", "max_kmh"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved summary CSV: {os.path.abspath(args.out_csv)}")
    else:
        print("\nNo speed samples collected (maybe no detections reached the speed window).")

    # optional: write raw samples CSV
    if args.out_samples_csv:
        with open(args.out_samples_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tracker_id", "sample_index", "speed_kmh"])
            for tid in sorted(speed_samples.keys()):
                for i, s in enumerate(speed_samples[tid]):
                    writer.writerow([tid, i, f"{s:.6f}"])
        print(f"Saved samples CSV: {os.path.abspath(args.out_samples_csv)}")
