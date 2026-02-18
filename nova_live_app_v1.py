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

import base64
import csv
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

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
.db-table-wrap {
    overflow-x: auto;
    margin-top: 0.4rem;
}
.db-table {
    width: 100%; border-collapse: collapse;
    font-family: 'Share Tech Mono', monospace; font-size: 0.78rem;
}
.db-table th {
    background: #0b1826; color: #3a8fb5;
    font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase;
    padding: 0.45rem 0.7rem; border-bottom: 1px solid #1a2840;
    text-align: left; white-space: nowrap;
}
.db-table td {
    padding: 0.35rem 0.7rem; border-bottom: 1px solid #111d2a;
    color: #c8d6e5; white-space: nowrap;
}
.db-table tr:hover td { background: #0e1825; }
.db-table .td-id   { color: #3a8fb5; }
.db-table .td-avg  { color: #00e5ff; }
.db-table .td-max  { color: #ffb300; }
.db-table .td-min  { color: #00e676; }
.db-table .td-ok   { color: #00e676; font-weight: 600; }
.db-table .td-spd  { color: #ff5252; font-weight: 600; }
.db-table .no-data {
    text-align: center; color: #1e3a50; font-size: 0.75rem;
    letter-spacing: 0.18em; padding: 1.2rem;
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
# CSV DATABASE PATH
# ─────────────────────────────────────────────────────────────
CSV_DB_PATH    = "speed_summary.csv"
CSV_FIELDNAMES = ["ID", "Entry_Time", "Exit_Time", "Average_Speed", "Max", "Min", "Remarks"]

REPLAY_DIR     = os.path.join("static", "replays")
os.makedirs(REPLAY_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# CSV DATABASE FUNCTIONS
# ─────────────────────────────────────────────────────────────
# Module-level lock for thread-safe CSV writes
_csv_lock = threading.Lock()


def init_csv_db():
    """Create CSV with headers if it doesn't exist."""
    if not os.path.exists(CSV_DB_PATH):
        with open(CSV_DB_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()


def upsert_csv_row(tid, entry_time, exit_time, samples, speed_threshold):
    """
    Write or update a single vehicle row in the CSV database.
    Reads all rows, replaces the one matching tid (or appends), then rewrites.
    Thread-safe via a simple file lock approach using a module-level lock.
    """
    if not samples:
        return

    avg_speed = round(sum(samples) / len(samples), 2)
    max_speed = round(max(samples), 2)
    min_speed = round(min(samples), 2)
    remarks   = 1 if avg_speed > speed_threshold else 0

    new_row = {
        "ID":            tid,
        "Entry_Time":    entry_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Exit_Time":     exit_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Average_Speed": avg_speed,
        "Max":           max_speed,
        "Min":           min_speed,
        "Remarks":       remarks,
    }

    with _csv_lock:
        rows = []
        updated = False

        # Read existing rows
        if os.path.exists(CSV_DB_PATH):
            with open(CSV_DB_PATH, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("ID") == str(tid):
                        rows.append(new_row)
                        updated = True
                    else:
                        rows.append(row)

        if not updated:
            rows.append(new_row)

        # Rewrite entire file
        with open(CSV_DB_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)


def read_csv_as_string():
    """Read the CSV database and return as a string for download."""
    if not os.path.exists(CSV_DB_PATH):
        return ""
    with open(CSV_DB_PATH, "r", newline="") as f:
        return f.read()


def _build_table_rows(rows):
    """Shared row builder for both table views."""
    body_rows = []
    for r in rows:
        remarks_val = str(r.get("Remarks", "0")).strip()
        rem_cell = ('<td class="td-spd">⚠ SPEEDING</td>'
                    if remarks_val == "1" else '<td class="td-ok">✔ NORMAL</td>')

        def _fmt(key, _r=r, decimals=1):
            try: return f"{float(_r.get(key, 0)):.{decimals}f}"
            except: return _r.get(key, "—")

        body_rows.append(
            f"<tr>"
            f"<td class='td-id'>#{r.get('ID','?')}</td>"
            f"<td>{r.get('Entry_Time','—')}</td>"
            f"<td>{r.get('Exit_Time','—')}</td>"
            f"<td class='td-avg'>{_fmt('Average_Speed')}</td>"
            f"{rem_cell}"
            f"</tr>"
        )
    return body_rows


_TABLE_HEADER = (
    "<table class='db-table'>"
    "<thead><tr>"
    "<th>ID</th><th>Entry Time</th><th>Exit Time</th>"
    "<th>Avg Speed (km/h)</th><th>Remarks</th>"
    "</tr></thead><tbody>"
)


def _load_sorted_rows(reverse=True):
    """Read CSV and return rows sorted by Exit_Time. reverse=True → newest first."""
    if not os.path.exists(CSV_DB_PATH):
        return None
    with _csv_lock:
        with open(CSV_DB_PATH, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        return None

    def _exit_key(r):
        try: return datetime.strptime(r["Exit_Time"], "%Y-%m-%d %H:%M:%S")
        except: return datetime.min

    return sorted(rows, key=_exit_key, reverse=reverse)


def render_db_table_html(speed_threshold: float) -> str:
    """Live table — 10 most recent records, newest first."""
    rows = _load_sorted_rows(reverse=True)
    if rows is None:
        return '<div class="no-data">◈ NO DATA YET</div>'

    if not rows:
        return '<div class="no-data">◈ NO DATA YET</div>'

    total     = len(rows)
    rows      = rows[:10]
    body_rows = _build_table_rows(rows)

    return (
        f'<div class="db-table-wrap">'
        f'{_TABLE_HEADER}{"".join(body_rows)}</tbody></table>'
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.62rem;'
        f'color:#3a5068;letter-spacing:0.15em;text-align:right;padding-top:4px;">'
        f'SHOWING {len(rows)} OF {total} RECORDS</div>'
        f'</div>'
    )


def render_full_table_html(speed_threshold: float) -> str:
    """Full table — ALL records, oldest first."""
    rows = _load_sorted_rows(reverse=False)
    if rows is None:
        return '<div class="no-data">◈ NO DATA YET</div>'

    if not rows:
        return '<div class="no-data">◈ NO DATA YET</div>'

    total     = len(rows)
    body_rows = _build_table_rows(rows)

    return (
        f'<div class="db-table-wrap">'
        f'{_TABLE_HEADER}{"".join(body_rows)}</tbody></table>'
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.62rem;'
        f'color:#3a5068;letter-spacing:0.15em;text-align:right;padding-top:4px;">'
        f'ALL {total} RECORDS · OLDEST → NEWEST</div>'
        f'</div>'
    )


def _find_replay_for_tid(tid: str):
    """Return (absolute_path, filename) for a replay matching tracker id, or (None, None)."""
    if not os.path.exists(REPLAY_DIR):
        return None, None
    prefix = f"replay_id{tid}_"
    for fname in os.listdir(REPLAY_DIR):
        if fname.startswith(prefix) and fname.endswith(".mp4"):
            return os.path.join(REPLAY_DIR, fname), fname
    return None, None


def render_incidents_table_html() -> str:
    """
    Speeding-only table, oldest → newest.
    ▶ PLAY button triggers a direct browser download of the replay mp4.
    """
    rows = _load_sorted_rows(reverse=False)
    if rows is None:
        return '<div class="no-data">◈ NO SPEEDING INCIDENTS YET</div>'

    rows = [r for r in rows if str(r.get("Remarks", "0")).strip() == "1"]
    if not rows:
        return '<div class="no-data">◈ NO SPEEDING INCIDENTS YET</div>'

    header = (
        "<table class='db-table'>"
        "<thead><tr>"
        "<th>ID</th><th>Entry Time</th><th>Exit Time</th>"
        "<th>Avg Speed (km/h)</th><th>Replay</th>"
        "</tr></thead><tbody>"
    )

    body_rows = []
    for r in rows:
        tid = r.get("ID", "?")

        def _fmt(key, _r=r, d=1):
            try: return f"{float(_r.get(key, 0)):.{d}f}"
            except: return _r.get(key, "—")

        fpath, fname = _find_replay_for_tid(str(tid))
        if fpath and os.path.exists(fpath):
            with open(fpath, "rb") as vf:
                b64 = base64.b64encode(vf.read()).decode()
            replay_cell = (
                f'<td><a href="data:video/mp4;base64,{b64}" download="{fname}" '
                f'style="font-family:\'Share Tech Mono\',monospace;color:#ff5252;'
                f'text-decoration:none;letter-spacing:0.1em;font-size:0.78rem;">'
                f'⬇ DOWNLOAD</a></td>'
            )
        else:
            replay_cell = '<td style="color:#3a5068;font-size:0.72rem;font-family:\'Share Tech Mono\',monospace;">NO REPLAY</td>'

        body_rows.append(
            f"<tr>"
            f"<td class='td-id'>#{tid}</td>"
            f"<td>{r.get('Entry_Time','—')}</td>"
            f"<td>{r.get('Exit_Time','—')}</td>"
            f"<td class='td-avg'>{_fmt('Average_Speed')}</td>"
            f"{replay_cell}"
            f"</tr>"
        )

    total = len(rows)
    return (
        f'<div class="db-table-wrap">'
        f'{header}{"".join(body_rows)}</tbody></table>'
        f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.62rem;'
        f'color:#ff5252;letter-spacing:0.15em;text-align:right;padding-top:4px;">'
        f'⚠ {total} SPEEDING INCIDENT{"S" if total != 1 else ""} · OLDEST → NEWEST</div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────
# REPLAY WRITER
# ─────────────────────────────────────────────────────────────
def save_replay(frames_snapshot, tid, avg_speed, fps, frame_size):
    """
    Write a list of annotated BGR frames to static/replays/ as an H.264 mp4
    so Chrome can play it natively via the /app/static/ URL.
    Runs in a daemon thread so it never blocks the main loop.
    """
    def _write():
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(REPLAY_DIR,
                                f"replay_id{tid}_{ts}_{int(avg_speed)}kmh.mp4")
        h, w = frame_size

        # Try H.264 first (Chrome-compatible), fall back to mp4v
        for fourcc_str in ("avc1", "H264", "X264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out    = cv2.VideoWriter(filename, fourcc, fps, (w, h))
            if out.isOpened():
                break
            out.release()

        for f in frames_snapshot:
            out.write(f)
        out.release()

    threading.Thread(target=_write, daemon=True).start()
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
# CORE ENGINE
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

def to_sv(boxes, conf):
    if boxes.shape[0] == 0: return sv.Detections.empty()
    return sv.Detections(xyxy=boxes, confidence=conf,
                         class_id=np.zeros(len(boxes), dtype=int))

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

# ─────────────────────────────────────────────────────────────
# SESSION STATE
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
    if st.button("Scan Cameras", use_container_width=True):
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
    preset_name     = st.selectbox("Source preset", ["middle", "long", "short"])
    speed_threshold = st.number_input("Speed threshold (km/h)", min_value=10, max_value=300,
                                      value=80, step=5,
                                      help="Vehicles exceeding this speed are flagged (Remarks=1)")

    st.markdown('<div class="sec-hdr">Display</div>', unsafe_allow_html=True)
    show_poly  = st.checkbox("Show ROI polygon",   value=True)
    show_trace = st.checkbox("Show vehicle trace", value=True)

    st.markdown('<div class="sec-hdr">Controls</div>', unsafe_allow_html=True)
    col_s, col_e = st.columns(2)
    btn_start = col_s.button("▶ START", type="primary", use_container_width=True)
    btn_stop  = col_e.button("⏹ STOP",                  use_container_width=True)

# ─────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="nova-title">NOVA LIVE</div>', unsafe_allow_html=True)
st.markdown('<div class="nova-sub">VEHICLE SPEED DETECTION · ONNX DirectML · ByteTrack</div>',
            unsafe_allow_html=True)

col_feed, col_stats = st.columns([2.4, 1], gap="large")
with col_feed:
    feed_slot = st.empty()
    st.markdown('<div class="sec-hdr">Speed Database — Latest Records</div>',
                unsafe_allow_html=True)
    table_slot = st.empty()
    with st.expander("📋 View Full Database", expanded=False):
        full_table_slot = st.empty()

with col_stats:
    st.markdown('<div class="sec-hdr">System</div>', unsafe_allow_html=True)
    fps_slot    = st.empty()
    inf_slot    = st.empty()
    prov_slot   = st.empty()
    cam_slot    = st.empty()
    st.markdown('<div class="sec-hdr">Live Speeds</div>', unsafe_allow_html=True)
    speed_slot  = st.empty()
    st.markdown('<div class="sec-hdr">Replays</div>', unsafe_allow_html=True)
    replay_slot = st.empty()
    st.markdown('<div class="sec-hdr">Export</div>', unsafe_allow_html=True)
    export_slot = st.empty()

# ── Speeding Incidents Table (full width, below columns) ───
st.markdown('<div class="sec-hdr">⚠ Speeding Incidents — Click to Download Replay</div>',
            unsafe_allow_html=True)
incidents_slot = st.empty()

# Render export button
if st.session_state.last_csv:
    export_slot.download_button(
        "⬇ Download speed_summary.csv",
        data=st.session_state.last_csv,
        file_name="speed_summary.csv",
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
# INITIALIZE CSV DATABASE
# ─────────────────────────────────────────────────────────────
init_csv_db()

# ─────────────────────────────────────────────────────────────
# START CAPTURE + BUILD OBJECTS ONCE
# ─────────────────────────────────────────────────────────────
cap_buf, cap_stop = start_capture(int(cam_index), int(cam_w), int(cam_h))
time.sleep(0.15)

byte_track       = sv.ByteTrack(frame_rate=fps_live)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
coordinates      = defaultdict(lambda: deque(maxlen=int(fps_live)))
speed_samples    = defaultdict(list)

# ── Tracking entry/exit times ──────────────────────────────
# entry_times[tid]  = datetime when tid first appeared in ROI
# exit_times[tid]   = datetime of the most recent frame tid was seen in ROI
# active_ids        = set of tracker IDs currently visible this frame
entry_times      = {}   # tid -> datetime
exit_times       = {}   # tid -> datetime (updated every frame tid is visible)
prev_active_ids  = set()

thickness  = sv.calculate_optimal_line_thickness(resolution_wh=(int(cam_w), int(cam_h)))
text_scale = sv.calculate_optimal_text_scale(resolution_wh=(int(cam_w), int(cam_h)))

box_ann   = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
lbl_ann   = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness,
                               text_position=sv.Position.BOTTOM_CENTER,
                               color_lookup=sv.ColorLookup.TRACK)
trace_ann = sv.TraceAnnotator(thickness=thickness, trace_length=int(fps_live * 2),
                               position=sv.Position.BOTTOM_CENTER,
                               color_lookup=sv.ColorLookup.TRACK)

frame_idx        = 0
last_det         = sv.Detections.empty()
last_inf_ms      = 0.0
fps_counter      = 0
fps_t0           = time.time()
fps_est          = fps_live
frames_done      = 0
t0_wall          = time.perf_counter()
csv_refresh_every = 30   # update download CSV every N frames

# ── Replay buffer ──────────────────────────────────────────
REPLAY_SECONDS   = 8
replay_buffer    = deque()          # holds (annotated_bgr_frame,) for last 8 s
replay_count     = 0                # how many replays saved this session
replayed_ids     = set()            # tracker IDs already saved (one replay per vehicle)

st.session_state.running = True

# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
while st.session_state.running:

    frame = cap_buf.get()
    if frame is None:
        time.sleep(0.01)
        continue

    now_dt = datetime.now()   # local machine time for this frame

    # FPS
    fps_counter += 1
    now_ts = time.time()
    dt     = now_ts - fps_t0
    if dt >= 1.0:
        fps_est     = fps_counter / dt
        fps_counter = 0
        fps_t0      = now_ts

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

    # ── Entry / exit time tracking ─────────────────────────
    current_active_ids = set()
    if dets.tracker_id is not None and len(dets.tracker_id) > 0:
        for tid in dets.tracker_id:
            tid = int(tid)
            current_active_ids.add(tid)
            if tid not in entry_times:
                # Vehicle just entered ROI for the first time
                entry_times[tid] = now_dt
            # Always update exit time while vehicle is visible
            exit_times[tid] = now_dt

    # Detect vehicles that just left the ROI this frame
    just_left = prev_active_ids - current_active_ids
    for tid in just_left:
        # Vehicle exited — write/update its final row in the CSV database
        if speed_samples[tid]:
            avg_spd = sum(speed_samples[tid]) / len(speed_samples[tid])
            upsert_csv_row(
                tid             = tid,
                entry_time      = entry_times.get(tid, now_dt),
                exit_time       = exit_times.get(tid, now_dt),
                samples         = speed_samples[tid],
                speed_threshold = float(speed_threshold),
            )
            # Save replay if vehicle was speeding and not yet saved
            if avg_spd > float(speed_threshold) and tid not in replayed_ids:
                replayed_ids.add(tid)
                replay_count += 1
                save_replay(
                    frames_snapshot = list(replay_buffer),
                    tid             = tid,
                    avg_speed       = avg_spd,
                    fps             = max(fps_est, 1.0),
                    frame_size      = (frame.shape[0], frame.shape[1]),
                )

    prev_active_ids = current_active_ids

    # ── Homography + speed ─────────────────────────────────
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

    # ── Annotate ───────────────────────────────────────────
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

    # ── Rolling replay buffer (keep last REPLAY_SECONDS of annotated frames) ──
    replay_buffer.append(ann.copy())
    max_buf = int(fps_est * REPLAY_SECONDS) if fps_est > 0 else int(fps_live * REPLAY_SECONDS)
    while len(replay_buffer) > max_buf:
        replay_buffer.popleft()

    # ── Update UI slots ────────────────────────────────────
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
        speed_rows = []
        for t, s in sorted(live_speeds.items()):
            cls = "fast" if s > speed_threshold else ""
            speed_rows.append(
                f'<div class="speed-row"><span class="tid">VHCL #{t}</span>'
                f'<span class="kmh {cls}">{s} km/h</span></div>'
            )
        speed_slot.markdown("".join(speed_rows), unsafe_allow_html=True)
    else:
        speed_slot.markdown(
            '<div style="font-family:\'Share Tech Mono\',monospace;color:#1e3a50;'
            'font-size:0.75rem;letter-spacing:0.15em;">AWAITING DETECTIONS…</div>',
            unsafe_allow_html=True)

    replay_slot.markdown(
        f'<div class="stat-card"><div class="stat-label">Replays Saved</div>'
        f'<div class="stat-value {"warn" if replay_count > 0 else ""}">'
        f'{"🎬 " if replay_count > 0 else ""}{replay_count}</div>'
        f'<div style="font-size:0.62rem;color:#3a5068;letter-spacing:0.12em;margin-top:2px;">'
        f'{REPLAY_DIR}</div></div>',
        unsafe_allow_html=True)

    # Refresh download CSV and database table every N frames
    if frame_idx % csv_refresh_every == 0:
        st.session_state.last_csv = read_csv_as_string()
        table_slot.markdown(render_db_table_html(float(speed_threshold)),
                            unsafe_allow_html=True)
        full_table_slot.markdown(render_full_table_html(float(speed_threshold)),
                                 unsafe_allow_html=True)
        incidents_slot.markdown(render_incidents_table_html(), unsafe_allow_html=True)

    frame_idx   += 1
    frames_done += 1

# ─────────────────────────────────────────────────────────────
# AFTER STOP — flush all still-active vehicles to CSV
# ─────────────────────────────────────────────────────────────
cap_stop.set()

flush_time = datetime.now()
for tid in list(prev_active_ids):
    if speed_samples[tid]:
        upsert_csv_row(
            tid             = tid,
            entry_time      = entry_times.get(tid, flush_time),
            exit_time       = exit_times.get(tid, flush_time),
            samples         = speed_samples[tid],
            speed_threshold = float(speed_threshold),
        )

# Final CSV snapshot for download button
st.session_state.last_csv = read_csv_as_string()
table_slot.markdown(render_db_table_html(float(speed_threshold)), unsafe_allow_html=True)
full_table_slot.markdown(render_full_table_html(float(speed_threshold)), unsafe_allow_html=True)
incidents_slot.markdown(render_incidents_table_html(), unsafe_allow_html=True)

feed_slot.markdown("""
<div style="background:#0e1520;border:1px solid #1a2840;border-radius:4px;
            padding:3rem;text-align:center;font-family:'Share Tech Mono',monospace;
            color:#1e3a50;font-size:1rem;letter-spacing:0.15em;">
    ◈ STOPPED
</div>""", unsafe_allow_html=True)