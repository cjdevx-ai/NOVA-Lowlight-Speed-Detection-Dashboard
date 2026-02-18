"""
NOVA Dashboard - Full Integrated System
Author: Clarence Jay Fetalino
Year: 2026
"""

import csv
import io
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

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NOVA · Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# COLOR SYSTEM & CSS
# ═══════════════════════════════════════════════════════════
COLORS = {
    "cyan":     "#00E5FF",
    "bg_dark":  "#0F172A",
    "red":      "#FF4D4F",
    "charcoal": "#2B2D42",
    "white":    "#FFFFFF",
    "gray":     "#64748b",
}

def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: {COLORS["bg_dark"]}; color: {COLORS["white"]}; }}
    footer {{visibility: hidden;}}
    
    section[data-testid="stSidebar"] {{ background: {COLORS["charcoal"]} !important; border-right: 2px solid {COLORS["cyan"]}33; }}
    
    .nova-header {{
        background: {COLORS["charcoal"]}; padding: 1rem 2rem; border-bottom: 2px solid {COLORS["cyan"]};
        display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem; border-radius: 8px;
    }}
    .nova-header-title {{ font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: {COLORS["cyan"]}; letter-spacing: 0.05em; text-transform: uppercase; }}
    
    .kpi-card {{
        background: {COLORS["charcoal"]}; border: 1px solid {COLORS["cyan"]}33; border-left: 4px solid {COLORS["cyan"]};
        padding: 1.2rem; border-radius: 6px; margin-bottom: 1rem;
    }}
    .kpi-label {{ font-size: 0.75rem; color: {COLORS["gray"]}; text-transform: uppercase; margin-bottom: 0.3rem; }}
    .kpi-value {{ font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: {COLORS["cyan"]}; }}
    .kpi-value.alert {{ color: {COLORS["red"]}; }}

    .sec-hdr {{ font-size: 0.68rem; letter-spacing: 0.22em; color: {COLORS["gray"]}; text-transform: uppercase; border-bottom: 1px solid #1a2840; padding-bottom: 3px; margin: 0.9rem 0 0.5rem 0; }}
    .prov-gpu {{ color: #00e676; font-weight: bold; }}
    .prov-cpu {{ color: #ffb300; font-weight: bold; }}
    
    .speed-row {{ 
        display: flex; justify-content: space-between; padding: 0.4rem 0.8rem; 
        background: #1e293b; border: 1px solid {COLORS["cyan"]}22; margin-bottom: 4px; border-radius: 4px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;
    }}
    .kmh-high {{ color: {COLORS["red"]}; font-weight: bold; }}

    .log-table {{ width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }}
    .log-table th {{ background: {COLORS["charcoal"]}; color: {COLORS["cyan"]}; padding: 0.8rem; text-align: left; border-bottom: 2px solid {COLORS["cyan"]}; }}
    .log-table td {{ padding: 0.7rem 0.8rem; border-bottom: 1px solid {COLORS["cyan"]}22; }}
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "current_tab": "Live",
        "running": False,
        "cameras": [],
        "cam_scanned": False,
        "cam_index": 0,
        "cam_w": 1280,
        "cam_h": 720,
        "onnx_path": r"G:\cla_projects\NOVA Lowlight Speed Detection Dashboard\yolo11nano_1k_320\best.onnx",
        "conf_th": 0.40,
        "iou_th": 0.50,
        "topk": 120,
        "img_size": 320,
        "infer_every": 2,
        "preset_name": "middle",
        "show_poly": True,
        "show_trace": True,
        "total_vehicles": 0,
        "speeding_count": 0,
        "avg_speed": 0.0,
        "vehicle_logs": [],
        "last_csv": "",
        "speed_samples": defaultdict(list),
        "logged_ids": set()
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ═══════════════════════════════════════════════════════════
# CORE ENGINE CLASSES & MATH
# ═══════════════════════════════════════════════════════════
SOURCE_PRESETS = {
    "long":   np.array([[38, 56],   [345, 9],  [1207, 367], [370, 715]], dtype=np.int32),
    "short":  np.array([[140, 230], [661, 132], [1207, 367], [370, 715]], dtype=np.int32),
    "middle": np.array([[128, 151], [461, 90],  [1137, 397], [370, 715]], dtype=np.int32),
}
TARGET_WIDTH, TARGET_HEIGHT = 4, 17
TARGET = np.array([[0,0], [TARGET_WIDTH-1,0], [TARGET_WIDTH-1,TARGET_HEIGHT-1], [0,TARGET_HEIGHT-1]], dtype=np.float32)

class ViewTransformer:
    def __init__(self, source, target):
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))
    def transform_points(self, pts):
        if pts.size == 0: return pts
        return cv2.perspectiveTransform(pts.reshape(-1,1,2).astype(np.float32), self.m).reshape(-1,2)

def letterbox(im, new_shape, color=(114,114,114)):
    h,w = im.shape[:2]; r = min(new_shape/h, new_shape/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = new_shape-nw, new_shape-nh
    l, r2 = pw//2, pw-pw//2; t, b = ph//2, ph-ph//2
    return cv2.copyMakeBorder(resized, t, b, l, r2, cv2.BORDER_CONSTANT, value=color), r, l, t

def iou_xyxy(a, b):
    xx1=np.maximum(a[0],b[:,0]); yy1=np.maximum(a[1],b[:,1])
    xx2=np.minimum(a[2],b[:,2]); yy2=np.minimum(a[3],b[:,3])
    inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter+1e-9)

def nms(boxes, scores, iou_th=0.5):
    order=scores.argsort()[::-1]; keep=[]
    while order.size>0:
        i=order[0]; keep.append(i)
        if order.size==1: break
        order=order[1:][iou_xyxy(boxes[i],boxes[order[1:]])<iou_th]
    return keep

def in_zone_mask(dets, polygon):
    if len(dets)==0: return np.array([],dtype=bool)
    cx=(dets.xyxy[:,0]+dets.xyxy[:,2])/2; cy=(dets.xyxy[:,1]+dets.xyxy[:,3])/2
    pts=np.stack([cx,cy],axis=1).astype(np.float32)
    return np.array([cv2.pointPolygonTest(polygon,(float(p[0]),float(p[1])),False)>=0 for p in pts],dtype=bool)

class LatestFrame:
    def __init__(self):
        self._lock=threading.Lock(); self._frame=None
    def set(self, f):
        with self._lock: self._frame=f
    def get(self):
        with self._lock: return self._frame

def start_capture(cam_index, width, height):
    buf=LatestFrame(); stop=threading.Event()
    def _loop():
        cap=cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened(): cap=cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        while not stop.is_set():
            ok, frame=cap.read()
            if ok and frame is not None: buf.set(frame)
            else: time.sleep(0.005)
        cap.release()
    threading.Thread(target=_loop, daemon=True).start()
    return buf, stop

@st.cache_resource
def get_ort_session(path):
    if not os.path.exists(path): return None, "", "", False
    so = ort.SessionOptions()
    avail = ort.get_available_providers()
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if "DmlExecutionProvider" in avail else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name, ("Dml" in str(providers))

def onnx_detect(sess, inp_name, out_name, frame_bgr, img_size, conf_th, iou_th, topk):
    img,r,px,py=letterbox(frame_bgr, img_size)
    x=cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    x=np.transpose(x,(2,0,1))[None]
    y=sess.run([out_name],{inp_name:x})[0]
    y=np.squeeze(y)
    if y.ndim==2 and y.shape[0]<y.shape[1]: y=y.T
    xywh=y[:, 0:4]; conf=y[:, 4:].max(axis=1); m=conf>=conf_th
    if not np.any(m): return np.zeros((0,4),np.float32), np.zeros((0,),np.float32)
    xywh,conf=xywh[m],conf[m]
    cx,cy,w,h=xywh[:,0],xywh[:,1],xywh[:,2],xywh[:,3]
    boxes=np.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],axis=1)
    boxes[:,[0,2]]-=px; boxes[:,[1,3]]-=py; boxes/=r
    keep=nms(boxes,conf,iou_th)
    return boxes[keep].astype(np.float32), conf[keep].astype(np.float32)

def to_sv(boxes, conf):
    if boxes.shape[0]==0: return sv.Detections.empty()
    return sv.Detections(xyxy=boxes, confidence=conf, class_id=np.zeros(len(boxes),dtype=int))

def discover_cameras(max_test=8):
    found = []
    for idx in range(max_test):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened(): continue
        ok, _ = cap.read()
        if ok:
            found.append({
                "index": idx, "label": f"Camera {idx}", 
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            })
        cap.release()
    return found

def build_csv(samples):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Tracker ID", "Mean Speed (km/h)", "Max Speed (km/h)", "Samples"])
    for tid, s_list in samples.items():
        if s_list:
            writer.writerow([tid, round(sum(s_list)/len(s_list), 2), round(max(s_list), 2), len(s_list)])
    return output.getvalue()

# ═══════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════
def render_header():
    st.markdown(f"""
    <div class="nova-header">
        <span class="nova-header-title">NIGHTTIME OBSERVATION AND VEHICLE ANALYSIS</span>
        <div style="font-family:'JetBrains Mono'; font-size:0.8rem;">
            STATUS: <span style="color:{COLORS['cyan'] if not st.session_state.running else COLORS['red']}">
            {'● STANDBY' if not st.session_state.running else '● LIVE'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("## Navigation")
        tabs = ["Live", "Analytics", "Logs", "Settings", "Meet Our Team", "Help"]
        for tab in tabs:
            if st.button(tab, use_container_width=True, type="primary" if st.session_state.current_tab == tab else "secondary"):
                st.session_state.current_tab = tab
                st.rerun()

# ═══════════════════════════════════════════════════════════
# TAB FUNCTIONS
# ═══════════════════════════════════════════════════════════

def tab_live():
    col_feed, col_stats = st.columns([2.5, 1], gap="medium")
    
    with col_feed:
        feed_slot = st.empty()
    
    with col_stats:
        st.markdown('<div class="sec-hdr">System Status</div>', unsafe_allow_html=True)
        fps_slot = st.empty()
        inf_slot = st.empty()
        prov_slot = st.empty()
        
        st.markdown('<div class="sec-hdr">Live Speed Log</div>', unsafe_allow_html=True)
        speed_slot = st.empty()
        
        st.markdown('<div class="sec-hdr">Controls</div>', unsafe_allow_html=True)
        btn_start = st.button("▶ START ENGINE", type="primary", use_container_width=True)
        btn_stop = st.button("⏹ STOP ENGINE", use_container_width=True)
        
        export_slot = st.empty()
        if st.session_state.get("last_csv"):
            export_slot.download_button("⬇ Download Logs", st.session_state.last_csv, "nova_speeds.csv", "text/csv", use_container_width=True)

    if btn_stop:
        st.session_state.running = False

    if btn_start or st.session_state.running:
        st.session_state.running = True
        sess, inp_name, out_name, gpu = get_ort_session(st.session_state.onnx_path)
        if sess is None:
            st.error("Model Error. Check Settings."); st.session_state.running = False; return

        cap_buf, cap_stop = start_capture(int(st.session_state.cam_index), int(st.session_state.cam_w), int(st.session_state.cam_h))
        byte_track = sv.ByteTrack(frame_rate=30)
        SOURCE = SOURCE_PRESETS[st.session_state.preset_name]
        view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
        coordinates = defaultdict(lambda: deque(maxlen=30))
        
        box_ann = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
        lbl_ann = sv.LabelAnnotator(text_scale=0.5, color_lookup=sv.ColorLookup.TRACK)
        trace_ann = sv.TraceAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)

        frame_idx = 0
        t0_wall = time.perf_counter()
        inf_ms = 0.0

        while st.session_state.running:
            frame = cap_buf.get()
            if frame is None:
                time.sleep(0.005); continue

            if frame_idx % int(st.session_state.infer_every) == 0:
                t_inf = time.perf_counter()
                bx, cf = onnx_detect(sess, inp_name, out_name, frame, st.session_state.img_size, st.session_state.conf_th, st.session_state.iou_th, st.session_state.topk)
                inf_ms = (time.perf_counter() - t_inf) * 1000
                dets = to_sv(bx, cf)

            mask = in_zone_mask(dets, SOURCE)
            tracked = byte_track.update_with_detections(detections=dets[mask])
            
            labels = []
            live_speeds = {}
            
            if tracked.tracker_id is not None:
                coords = tracked.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                trans_coords = view_transformer.transform_points(coords).astype(int)
                
                for tid, (_, y) in zip(tracked.tracker_id, trans_coords):
                    tid = int(tid)
                    coordinates[tid].append(y)
                    hist = coordinates[tid]
                    
                    if len(hist) > 10:
                        dist = abs(hist[-1] - hist[0])
                        speed = (dist / (len(hist)/30.0)) * 3.6
                        st.session_state.speed_samples[tid].append(speed)
                        labels.append(f"#{tid} {int(speed)} km/h")
                        live_speeds[tid] = int(speed)
                        
                        if tid not in st.session_state.logged_ids and len(hist) >= 25:
                            avg_s = sum(st.session_state.speed_samples[tid]) / len(st.session_state.speed_samples[tid])
                            remark = "Speeding" if avg_s > 60 else "Normal"
                            st.session_state.vehicle_logs.insert(0, {"id": tid, "in": datetime.now().strftime("%H:%M:%S"), "speed": int(avg_s), "rem": remark})
                            st.session_state.logged_ids.add(tid)
                            st.session_state.total_vehicles += 1
                            if remark == "Speeding": st.session_state.speeding_count += 1
                    else:
                        labels.append(f"#{tid}")

            ann_frame = frame.copy()
            if st.session_state.show_poly: sv.draw_polygon(ann_frame, SOURCE, sv.Color.from_hex("#00E5FF"))
            if st.session_state.show_trace: ann_frame = trace_ann.annotate(ann_frame, tracked)
            ann_frame = box_ann.annotate(ann_frame, tracked)
            ann_frame = lbl_ann.annotate(ann_frame, tracked, labels=labels)

            feed_slot.image(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            elapsed = time.perf_counter() - t0_wall
            curr_fps = frame_idx / elapsed if elapsed > 0 else 0
            
            fps_slot.markdown(f'<div class="kpi-card"><div class="kpi-label">System FPS</div><div class="kpi-value">{curr_fps:.1f}</div></div>', unsafe_allow_html=True)
            inf_slot.markdown(f'<div class="kpi-card"><div class="kpi-label">Inference</div><div class="kpi-value">{inf_ms:.1f}ms</div></div>', unsafe_allow_html=True)
            prov_txt = "GPU (DML)" if gpu else "CPU"
            prov_slot.markdown(f'<div class="kpi-card"><div class="kpi-label">Engine</div><div class="kpi-value" style="font-size:0.8rem">{prov_txt}</div></div>', unsafe_allow_html=True)

            if live_speeds:
                speed_html = "".join([f'<div class="speed-row"><span>ID #{t}</span><span class="{"kmh-high" if s > 60 else ""}">{s} km/h</span></div>' for t, s in sorted(live_speeds.items())])
                speed_slot.markdown(speed_html, unsafe_allow_html=True)

            if frame_idx % 30 == 0:
                st.session_state.last_csv = build_csv(st.session_state.speed_samples)

            frame_idx += 1
        cap_stop.set()
        st.rerun()
    else:
        feed_slot.markdown('<div style="background:#0F172A; border:2px dashed #1E293B; border-radius:8px; padding:100px; text-align:center; color:#334155;">◈ SYSTEM STANDBY</div>', unsafe_allow_html=True)

def tab_settings():
    st.markdown("## System Settings")
    with st.expander("🎥 Camera Source", expanded=True):
        if st.button("🔍 Scan Hardware"): st.session_state.cameras = discover_cameras(); st.session_state.cam_scanned = True
        if st.session_state.cam_scanned and st.session_state.cameras:
            opts = {f"[{c['index']}] {c['label']}": c['index'] for c in st.session_state.cameras}
            st.session_state.cam_index = st.selectbox("Select Camera", list(opts.keys()))
            st.session_state.cam_index = opts[st.session_state.cam_index]
        else:
            st.session_state.cam_index = st.number_input("Index", 0, 10, st.session_state.cam_index)
        st.session_state.cam_w = st.selectbox("Width", [640, 960, 1280, 1920], index=2)
        st.session_state.cam_h = st.selectbox("Height", [480, 540, 720, 1080], index=2)

    with st.expander("🤖 Model"):
        st.session_state.onnx_path = st.text_input("Path", st.session_state.onnx_path)
        st.session_state.conf_th = st.slider("Conf", 0.05, 0.95, st.session_state.conf_th)
        st.session_state.iou_th = st.slider("IOU", 0.10, 0.90, st.session_state.iou_th)
        st.session_state.img_size = st.selectbox("Size", [320, 416, 512, 640], index=0)
        st.session_state.infer_every = st.slider("Infer Every N", 1, 8, st.session_state.infer_every)

    with st.expander("🎯 Zone & ROI"):
        st.session_state.preset_name = st.selectbox("Preset", ["middle", "long", "short"])
        st.session_state.show_poly = st.checkbox("Show ROI", st.session_state.show_poly)
        st.session_state.show_trace = st.checkbox("Show Trace", st.session_state.show_trace)

def tab_logs():
    st.markdown("## Vehicle Logs")
    if not st.session_state.vehicle_logs:
        st.info("No vehicles detected yet.")
        return
    html = '<table class="log-table"><thead><tr><th>ID</th><th>Entry</th><th>Avg Speed</th><th>Remark</th></tr></thead><tbody>'
    for l in st.session_state.vehicle_logs:
        color = COLORS["red"] if l["rem"] == "Speeding" else COLORS["cyan"]
        html += f"<tr><td>{l['id']}</td><td>{l['in']}</td><td>{l['speed']} km/h</td><td style='color:{color}'>{l['rem']}</td></tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════
inject_css()
render_sidebar()
render_header()

if st.session_state.current_tab == "Live": tab_live()
elif st.session_state.current_tab == "Settings": tab_settings()
elif st.session_state.current_tab == "Logs": tab_logs()
else: st.info(f"Page '{st.session_state.current_tab}' is under construction.")