"""
nova_live_app.py
================
NOVA Live Speed Detection - Streamlit UI

Run:
  streamlit run nova_live_app.py --server.runOnSave true

Key design rules to avoid Streamlit errors:
  - while loop (NOT st.rerun) so widgets are never rebuilt -> no flicker
  - download_button rendered ONCE outside the loop via session_state csv string
  - annotators built ONCE before loop
  - only .empty() slots updated per frame
"""

import csv
import io
import os
import threading
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st
import supervision as sv

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NOVA · Live Speed Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Barlow Condensed', sans-serif;
    background-color: #0a0c0f; color: #c8d6e5;
}
section[data-testid="stSidebar"] {
    background: #0e1217; border-right: 1px solid #1e2a36;
}
section[data-testid="stSidebar"] * { font-family: 'Barlow Condensed', sans-serif !important; }
.nova-title {
    font-family: 'Share Tech Mono', monospace; font-size: 2.1rem;
    letter-spacing: 0.18em; color: #00e5ff;
    text-shadow: 0 0 18px rgba(0,229,255,0.35); margin-bottom: 0; line-height: 1;
}
.nova-sub {
    font-family: 'Share Tech Mono', monospace; font-size: 0.75rem;
    letter-spacing: 0.25em; color: #3a5068; margin-top: 2px; margin-bottom: 1.4rem;
}
.stat-card {
    background: #0e1520; border: 1px solid #1a2840;
    border-left: 3px solid #00e5ff; padding: 0.55rem 0.9rem;
    margin-bottom: 0.45rem; border-radius: 2px;
}
.stat-label { font-size: 0.68rem; letter-spacing: 0.18em; color: #3a5068; text-transform: uppercase; }
.stat-value { font-family: 'Share Tech Mono', monospace; font-size: 1.35rem; color: #00e5ff; line-height: 1.1; }
.stat-value.warn { color: #ffb300; }
.stat-value.ok   { color: #00e676; }
.speed-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.3rem 0.6rem; margin-bottom: 2px;
    background: #0e1520; border: 1px solid #1a2840; border-radius: 2px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.82rem;
}
.speed-row .tid { color: #3a8fb5; }
.speed-row .kmh { color: #00e676; font-size: 1rem; }
.speed-row .kmh.fast { color: #ff5252; }
.sec-hdr {
    font-size: 0.68rem; letter-spacing: 0.22em; color: #3a5068;
    text-transform: uppercase; border-bottom: 1px solid #1a2840;
    padding-bottom: 3px; margin: 0.9rem 0 0.5rem 0;
}
.cam-badge {
    display: inline-block; background: #0e2030; border: 1px solid #1a4060;
    color: #00b4d8; font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem; padding: 2px 10px; border-radius: 2px; letter-spacing: 0.12em;
}
.prov-gpu { color: #00e676; }
.prov-cpu { color: #ffb300; }
div[data-testid="stButton"] > button {
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.1em; border-radius: 2px !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: #00b4d8 !important; border: none !important; color: #000 !important;
}
.stSelectbox label, .stSlider label, .stNumberInput label,
.stTextInput label, .stCheckbox label {
    font-size: 0.75rem !important; letter-spacing: 0.14em !important;
    color: #3a8fb5 !important; text-transform: uppercase !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SOURCE PRESETS (verbatim)
# ─────────────────────────────────────────────────────────────
SOURCE_PRESETS = {
    "long":   np.array([[38, 56],   [345, 9],  [1207, 367], [370, 715]], dtype=np.int32),
    "short":  np.array([[140, 230], [661, 132], [1207, 367], [370, 715]], dtype=np.int32),
    "middle": np.array([[128, 151], [461, 90],  [1137, 397], [370, 715]], dtype=np.int32),
}
TARGET_WIDTH  = 4
TARGET_HEIGHT = 17
TARGET = np.array(
    [[0, 0], [TARGET_WIDTH-1, 0],
     [TARGET_WIDTH-1, TARGET_HEIGHT-1], [0, TARGET_HEIGHT-1]],
    dtype=np.float32,
)

# ─────────────────────────────────────────────────────────────
# CAMERA DISCOVERY
# ─────────────────────────────────────────────────────────────
def discover_cameras(max_test=8):
    hints = {0: "Default Camera", 1: "USB / OBS Virtual", 2: "Camera 2",
             3: "Camera 3", 4: "Camera 4", 5: "Camera 5", 6: "Camera 6", 7: "Camera 7"}
    found = []
    for idx in range(max_test):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, _ = cap.read()
            if not ok:
                cap.release()
                continue
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            found.append(dict(index=idx, label=hints.get(idx, f"Camera {idx}"),
                              width=w, height=h, fps=fps))
            break
    return found

# ─────────────────────────────────────────────────────────────
# CORE ENGINE (speed_test_onnx.py — unchanged)
# ─────────────────────────────────────────────────────────────
class ViewTransformer:
    def __init__(self, source, target):
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32))
    def transform_points(self, points):
        if points.size == 0: return points
        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2).astype(np.float32), self.m).reshape(-1, 2)

def letterbox(im, new_shape, color=(114, 114, 114)):
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph  = new_shape - nw, new_shape - nh
    l, r2   = pw // 2, pw - pw // 2
    t, b    = ph // 2, ph - ph // 2
    return cv2.copyMakeBorder(resized, t, b, l, r2, cv2.BORDER_CONSTANT, value=color), r, l, t

def iou_xyxy(a, b):
    xx1 = np.maximum(a[0], b[:, 0]); yy1 = np.maximum(a[1], b[:, 1])
    xx2 = np.minimum(a[2], b[:, 2]); yy2 = np.minimum(a[3], b[:, 3])
    inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[:,2]-b[:,0])*(b[:,3]-b[:,1]) - inter + 1e-9)

def nms(boxes, scores, iou_th=0.5):
    order = scores.argsort()[::-1]; keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        order = order[1:][iou_xyxy(boxes[i], boxes[order[1:]]) < iou_th]
    return keep

def in_zone_mask(dets, polygon):
    if len(dets) == 0: return np.array([], dtype=bool)
    cx = (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2.0
    cy = (dets.xyxy[:, 1] + dets.xyxy[:, 3]) / 2.0
    pts = np.stack([cx, cy], axis=1).astype(np.float32)
    return np.array([cv2.pointPolygonTest(polygon, (float(p[0]), float(p[1])), False) >= 0
                     for p in pts], dtype=bool)

# ─────────────────────────────────────────────────────────────
# THREADED CAPTURE
# ─────────────────────────────────────────────────────────────
class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
    def set(self, f):
        with self._lock: self._frame = f
    def get(self):
        with self._lock: return self._frame

def start_capture(cam_index, width, height):
    buf  = LatestFrame()
    stop = threading.Event()
    def _loop():
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_index)
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        while not stop.is_set():
            ok, frame = cap.read()
            if ok and frame is not None: buf.set(frame)
            else: time.sleep(0.005)
        cap.release()
    threading.Thread(target=_loop, daemon=True).start()
    return buf, stop

# ─────────────────────────────────────────────────────────────
# ORT SESSION (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def make_ort_session(onnx_path):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 2
    avail = ort.get_available_providers()
    if "DmlExecutionProvider" in avail:
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]; gpu = True
    else:
        providers = ["CPUExecutionProvider"]; gpu = False
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name, gpu

def onnx_detect(sess, inp_name, out_name, frame_bgr, img_size, conf_th, iou_th, topk):
    img, r, px, py = letterbox(frame_bgr, img_size)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    y = sess.run([out_name], {inp_name: x})[0]
    y = np.squeeze(y)
    if y.ndim == 2 and y.shape[0] < y.shape[1]: y = y.T
    xywh = y[:, 0:4]; conf = y[:, 4:].max(axis=1); m = conf >= conf_th
    if not np.any(m): return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    xywh, conf = xywh[m], conf[m]
    cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    boxes = np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1)
    boxes[:, [0, 2]] -= px; boxes[:, [1, 3]] -= py; boxes /= r
    H, W = frame_bgr.shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W-1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H-1)
    if conf.size > topk:
        idx = np.argsort(conf)[::-1][:topk]; boxes, conf = boxes[idx], conf[idx]
    keep = nms(boxes, conf, iou_th)
    return boxes[keep].astype(np.float32), conf[keep].astype(np.float32)

def to_sv(boxes, conf):
    if boxes.shape[0] == 0: return sv.Detections.empty()
    return sv.Detections(xyxy=boxes, confidence=conf,
                         class_id=np.zeros(len(boxes), dtype=int))

def build_csv(speed_samples):
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=["tracker_id","n_samples","min_kmh","mean_kmh","max_kmh"])
    w.writeheader()
    for tid in sorted(speed_samples.keys()):
        s = speed_samples[tid]
        if s:
            w.writerow(dict(tracker_id=tid, n_samples=len(s),
                            min_kmh=round(min(s), 2),
                            mean_kmh=round(sum(s)/len(s), 2),
                            max_kmh=round(max(s), 2)))
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────
# SESSION STATE  (minimal — only what survives repress of Start)
# ─────────────────────────────────────────────────────────────
for _k, _v in [("running", False), ("cameras", []), ("cam_scanned", False),
               ("last_csv", "")]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="nova-title">NOVA</div>', unsafe_allow_html=True)
    st.markdown('<div class="nova-sub">LIVE SPEED DETECTION</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Camera Source</div>', unsafe_allow_html=True)
    if st.button("🔍 Scan Cameras", use_container_width=True):
        with st.spinner("Scanning..."):
            st.session_state.cameras     = discover_cameras(8)
            st.session_state.cam_scanned = True

    if st.session_state.cam_scanned and st.session_state.cameras:
        cam_opts = {
            f"[{c['index']}] {c['label']}  ({c['width']}x{c['height']} @ {c['fps']:.0f}fps)": c
            for c in st.session_state.cameras
        }
        sel            = st.selectbox("Select Camera", list(cam_opts.keys()))
        sc             = cam_opts[sel]
        cam_index      = sc["index"]
        cam_label      = sc["label"]
        cam_fps_native = float(sc["fps"])
    elif st.session_state.cam_scanned:
        st.warning("No cameras found.")
        cam_index = 0; cam_label = "Camera 0"; cam_fps_native = 30.0
    else:
        st.info("Click **Scan Cameras** to detect USB / OBS / virtual cameras.")
        cam_index      = st.number_input("Manual camera index", 0, 10, 0, 1)
        cam_label      = f"Camera {cam_index}"
        cam_fps_native = 30.0

    cam_w = st.selectbox("Resolution width",  [640, 960, 1280, 1920], index=2)
    cam_h = st.selectbox("Resolution height", [480, 540, 720, 1080],  index=2)

    st.markdown('<div class="sec-hdr">Model</div>', unsafe_allow_html=True)
    onnx_path = st.text_input("ONNX path",
        value=r"G:\cla_projects\NOVA Lowlight Speed Detection Dashboard\yolo11nano_1k_320\best.onnx")
    conf_th     = st.slider("Conf threshold",       0.05, 0.95, 0.40, 0.01)
    iou_th      = st.slider("IOU threshold",        0.10, 0.90, 0.50, 0.01)
    topk        = st.slider("TopK pre-NMS",         20,   400,  120,  5)
    img_size    = st.selectbox("Inference size px", [320, 416, 512, 640], index=0)
    infer_every = st.slider("Infer every N frames", 1, 8, 2, 1)

    st.markdown('<div class="sec-hdr">ROI / Speed Zone</div>', unsafe_allow_html=True)
    preset_name = st.selectbox("Source preset", ["middle", "long", "short"])

    st.markdown('<div class="sec-hdr">Display</div>', unsafe_allow_html=True)
    show_poly  = st.checkbox("Show ROI polygon",   value=True)
    show_trace = st.checkbox("Show vehicle trace", value=True)

    st.markdown('<div class="sec-hdr">Controls</div>', unsafe_allow_html=True)
    col_s, col_e = st.columns(2)
    btn_start = col_s.button("▶ START", type="primary", use_container_width=True)
    btn_stop  = col_e.button("⏹ STOP",                  use_container_width=True)

# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT  — all slots declared once, never rebuilt
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="nova-title">NOVA LIVE</div>', unsafe_allow_html=True)
st.markdown('<div class="nova-sub">VEHICLE SPEED DETECTION · ONNX DirectML · ByteTrack</div>',
            unsafe_allow_html=True)

col_feed, col_stats = st.columns([2.4, 1], gap="large")
with col_feed:
    feed_slot = st.empty()
with col_stats:
    st.markdown('<div class="sec-hdr">System</div>', unsafe_allow_html=True)
    fps_slot   = st.empty()
    inf_slot   = st.empty()
    prov_slot  = st.empty()
    cam_slot   = st.empty()
    st.markdown('<div class="sec-hdr">Live Speeds</div>', unsafe_allow_html=True)
    speed_slot = st.empty()
    st.markdown('<div class="sec-hdr">Export</div>', unsafe_allow_html=True)
    # download_button rendered ONCE here, updated via session_state.last_csv
    export_slot = st.empty()

# Render export button once from whatever CSV is in state
if st.session_state.last_csv:
    export_slot.download_button(
        "⬇ Download Speed CSV",
        data=st.session_state.last_csv,
        file_name="nova_speed_summary.csv",
        mime="text/csv",
        use_container_width=True,
        key="export_btn",
    )

# ─────────────────────────────────────────────────────────────
# STOP BUTTON
# ─────────────────────────────────────────────────────────────
if btn_stop:
    st.session_state.running = False

# ─────────────────────────────────────────────────────────────
# IDLE STATE
# ─────────────────────────────────────────────────────────────
if not st.session_state.running and not btn_start:
    feed_slot.markdown("""
    <div style="background:#0e1520;border:1px solid #1a2840;border-radius:4px;
                padding:3rem;text-align:center;font-family:'Share Tech Mono',monospace;
                color:#1e3a50;font-size:1rem;letter-spacing:0.15em;">
        ◈ SYSTEM STANDBY<br>
        <span style="font-size:0.7rem;letter-spacing:0.2em;color:#122030;">
        SCAN CAMERAS · CONFIGURE · PRESS START</span>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# VALIDATE + LOAD
# ─────────────────────────────────────────────────────────────
if not os.path.exists(onnx_path):
    st.error(f"ONNX model not found: {onnx_path}")
    st.stop()

fps_live = cam_fps_native if cam_fps_native > 1 else 30.0
SOURCE   = SOURCE_PRESETS[preset_name]

try:
    sess, inp_name, out_name, gpu = make_ort_session(onnx_path)
except Exception as e:
    st.error(f"Failed to load ONNX: {e}")
    st.stop()

prov_html = ('<span class="prov-gpu">● GPU (DirectML)</span>' if gpu
             else '<span class="prov-cpu">● CPU · pip install onnxruntime-directml</span>')
prov_slot.markdown(
    f'<div class="stat-card"><div class="stat-label">Provider</div>'
    f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.85rem;">{prov_html}</div></div>',
    unsafe_allow_html=True)
cam_slot.markdown(
    f'<div class="stat-card"><div class="stat-label">Camera</div>'
    f'<div class="cam-badge">[{cam_index}] {cam_label}</div></div>',
    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# START CAPTURE + BUILD OBJECTS ONCE
# ─────────────────────────────────────────────────────────────
cap_buf, cap_stop = start_capture(int(cam_index), int(cam_w), int(cam_h))
time.sleep(0.15)

byte_track       = sv.ByteTrack(frame_rate=fps_live)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
coordinates      = defaultdict(lambda: deque(maxlen=int(fps_live)))
speed_samples    = defaultdict(list)

thickness  = sv.calculate_optimal_line_thickness(resolution_wh=(int(cam_w), int(cam_h)))
text_scale = sv.calculate_optimal_text_scale(resolution_wh=(int(cam_w), int(cam_h)))

# Annotators built ONCE
box_ann   = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
lbl_ann   = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness,
                               text_position=sv.Position.BOTTOM_CENTER,
                               color_lookup=sv.ColorLookup.TRACK)
trace_ann = sv.TraceAnnotator(thickness=thickness, trace_length=int(fps_live * 2),
                               position=sv.Position.BOTTOM_CENTER,
                               color_lookup=sv.ColorLookup.TRACK)

frame_idx   = 0
last_det    = sv.Detections.empty()
last_inf_ms = 0.0
fps_counter = 0
fps_t0      = time.time()
fps_est     = fps_live
frames_done = 0
t0_wall     = time.perf_counter()
# track how often to refresh the CSV in state (every 30 frames)
csv_refresh_every = 30

st.session_state.running = True

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# No download_button here — it lives above the loop, updated
# via st.session_state.last_csv every N frames
# ─────────────────────────────────────────────────────────────
while st.session_state.running:

    frame = cap_buf.get()
    if frame is None:
        time.sleep(0.01)
        continue

    # FPS
    fps_counter += 1
    now = time.time()
    dt  = now - fps_t0
    if dt >= 1.0:
        fps_est     = fps_counter / dt
        fps_counter = 0
        fps_t0      = now

    # Inference every N frames
    if frame_idx % int(infer_every) == 0:
        t0 = time.perf_counter()
        boxes, conf = onnx_detect(sess, inp_name, out_name,
                                  frame, int(img_size), float(conf_th),
                                  float(iou_th), int(topk))
        last_inf_ms = (time.perf_counter() - t0) * 1000.0
        last_det    = to_sv(boxes, conf)

    # Zone filter + track
    dets = last_det[np.arange(len(last_det))] if len(last_det) > 0 else sv.Detections.empty()
    mask = in_zone_mask(dets, SOURCE)
    dets = dets[mask]
    dets = byte_track.update_with_detections(detections=dets)

    # Homography + speed (Script 2 math — unchanged)
    points = dets.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    points = view_transformer.transform_points(points).astype(int)

    if dets.tracker_id is not None and len(dets.tracker_id) > 0:
        for tid, (_, y) in zip(dets.tracker_id, points):
            coordinates[int(tid)].append(int(y))

    labels      = []
    live_speeds = {}
    warmup      = fps_est / 2

    if dets.tracker_id is not None and len(dets.tracker_id) > 0:
        for tid in dets.tracker_id:
            tid  = int(tid)
            hist = coordinates[tid]
            if len(hist) < warmup:
                labels.append(f"#{tid}")
            else:
                dist   = abs(int(hist[-1]) - int(hist[0]))
                time_s = len(hist) / fps_est
                speed  = (dist / time_s) * 3.6
                speed_samples[tid].append(float(speed))
                labels.append(f"#{tid}  {int(speed)} km/h")
                live_speeds[tid] = int(speed)

    # Annotate
    ann = frame.copy()
    if show_poly:
        sv.draw_polygon(ann, polygon=SOURCE, color=sv.Color.RED)
    if show_trace:
        ann = trace_ann.annotate(scene=ann, detections=dets)
    ann = box_ann.annotate(scene=ann, detections=dets)
    ann = lbl_ann.annotate(scene=ann, detections=dets, labels=labels)

    elapsed  = time.perf_counter() - t0_wall
    live_fps = frames_done / elapsed if elapsed > 0 else 0.0

    cv2.putText(ann,
        f"FPS:{live_fps:.1f}  INF:{last_inf_ms:.0f}ms  EVERY:{infer_every}",
        (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 229, 255), 2)

    # Update slots only
    feed_slot.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_container_width=True)

    fps_class = "ok" if live_fps >= 20 else "warn"
    fps_slot.markdown(
        f'<div class="stat-card"><div class="stat-label">Live FPS</div>'
        f'<div class="stat-value {fps_class}">{live_fps:.1f}</div></div>',
        unsafe_allow_html=True)
    inf_slot.markdown(
        f'<div class="stat-card"><div class="stat-label">Inference ms</div>'
        f'<div class="stat-value">{last_inf_ms:.1f}</div></div>',
        unsafe_allow_html=True)

    if live_speeds:
        rows = "".join(
            f'<div class="speed-row"><span class="tid">VHCL #{t}</span>'
            f'<span class="kmh {"fast" if s > 80 else ""}">{s} km/h</span></div>'
            for t, s in sorted(live_speeds.items()))
        speed_slot.markdown(rows, unsafe_allow_html=True)
    else:
        speed_slot.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;color:#1e3a50;'
            'font-size:0.75rem;letter-spacing:0.15em;">AWAITING DETECTIONS…</div>',
            unsafe_allow_html=True)

    # Update CSV in session state every N frames (not every frame)
    # The download_button rendered above the loop picks this up on next script run
    if speed_samples and frame_idx % csv_refresh_every == 0:
        st.session_state.last_csv = build_csv(speed_samples)

    frame_idx   += 1
    frames_done += 1

# ─────────────────────────────────────────────────────────────
# AFTER STOP
# ─────────────────────────────────────────────────────────────
cap_stop.set()
if speed_samples:
    st.session_state.last_csv = build_csv(speed_samples)

feed_slot.markdown("""
<div style="background:#0e1520;border:1px solid #1a2840;border-radius:4px;
            padding:3rem;text-align:center;font-family:'Share Tech Mono',monospace;
            color:#1e3a50;font-size:1rem;letter-spacing:0.15em;">
    ◈ STOPPED
</div>""", unsafe_allow_html=True)