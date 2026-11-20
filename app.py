"""Diabetes Risk Screener - 2-step Streamlit app (BITSF464 project prototype).
Step 1: clinical measurements (same fields as the dataset) -> trained model risk.
Step 2: face scan (skin colour, facial blood-flow proxy, skin/collagen texture) -> experimental adjustment.
Run:  streamlit run app.py
"""
import os
import threading
import cv2
import numpy as np
import pandas as pd
import streamlit as st
try:  # live camera with face tracking (pip install streamlit-webrtc)
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    HAS_WEBRTC = True
except Exception:
    HAS_WEBRTC = False
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Diabetes Risk Screener", page_icon="🩺", layout="centered")

DATA_FILE = "synthetic_diabetes_dataset.xlsx"
FEATURES = ["Age", "BMI", "Blood_Pressure", "Insulin", "Diabetes_Pedigree_Function"]
FACE_MAX_ADJUST = 0.05  # face scan can move the final risk by at most +/- 5 percentage points

st.session_state.setdefault("step", 1)
st.session_state.setdefault("clinical", None)


# ---------------------------------------------------------------- model
@st.cache_resource
def train_model(df: pd.DataFrame, features: tuple):
    X, y = df[list(features)], df["Outcome"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # k = 5, as in the notebook
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
    model.fit(X, y)
    return model, auc


def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_excel(DATA_FILE, sheet_name="Patient Data")
    up = st.sidebar.file_uploader("Upload synthetic_diabetes_dataset.xlsx", type="xlsx")
    return pd.read_excel(up, sheet_name="Patient Data") if up else None


# ---------------------------------------------------------------- face analysis
ROIS = {"Forehead": (0.28, 0.06, 0.72, 0.24),
        "Left cheek": (0.14, 0.52, 0.38, 0.74),
        "Right cheek": (0.62, 0.52, 0.86, 0.74)}


CASCADE_FILES = ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"]
TILTS = [0, -15, 15, -30, 30]          # head tilts (degrees) the detector will try
_CASCADES = {}


def _cascade(name):
    if name not in _CASCADES:
        _CASCADES[name] = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    return _CASCADES[name]


def detector_ready():
    """True only if OpenCV can really load a face detector (OpenCV 5.x ships without the face-detector files)."""
    try:
        return any(not _cascade(n).empty() for n in CASCADE_FILES)
    except Exception:
        return False


def _rot(shape, angle):
    h, w = shape[:2]
    return cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)


def find_face(gray, tilts=TILTS, min_size=60):
    """Find a face, also when the head is tilted. Returns (box, angle) where the box is in the coordinates of
    the image rotated by `angle` (so that image shows the face upright), or None if no face is found."""
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    for ang in tilts:
        g = gray if ang == 0 else cv2.warpAffine(gray, _rot(gray.shape, ang), (w, h), borderMode=cv2.BORDER_REPLICATE)
        for name in CASCADE_FILES:
            casc = _cascade(name)
            if casc.empty():
                continue
            for scale, neighbours in [(1.1, 5), (1.05, 3)]:
                faces = casc.detectMultiScale(g, scale, neighbours, minSize=(min_size, min_size))
                if len(faces):
                    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3])), ang
    return None


def analyse_face(bgr):
    """Analyse forehead and both cheeks. Returns None when no face is found (nothing is guessed)."""
    big = max(bgr.shape[:2])
    if big > 960:  # shrink very large camera frames
        bgr = cv2.resize(bgr, None, fx=960 / big, fy=960 / big)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    res = find_face(gray)
    if res is None:
        return None
    (x, y, w, h), tilt = res
    if tilt:  # straighten the tilted head so the regions line up with the real forehead and cheeks
        bgr = cv2.warpAffine(bgr, _rot(bgr.shape, tilt), (bgr.shape[1], bgr.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, W - x), min(h, H - y)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(float)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    annot = bgr.copy()
    cv2.rectangle(annot, (x, y), (x + w, y + h), (255, 160, 0), 2)

    stats = {}
    for name, (x0, y0, x1, y1) in ROIS.items():
        xs, xe, ys, ye = x + int(x0 * w), x + int(x1 * w), y + int(y0 * h), y + int(y1 * h)
        patch = smooth[ys:ye, xs:xe]
        stats[name] = dict(L=L[ys:ye, xs:xe].mean(), a=a[ys:ye, xs:xe].mean(), b=b[ys:ye, xs:xe].mean(),
                           lap=cv2.Laplacian(patch, cv2.CV_64F).var(),
                           edge=(cv2.Canny(patch, 60, 140) > 0).mean())
        cv2.rectangle(annot, (xs, ys), (xe, ye), (0, 220, 0), 2)
        cv2.putText(annot, name, (xs, max(ys - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

    m = lambda k: float(np.mean([s[k] for s in stats.values()]))
    Lm, am, bm, lap, edge = m("L"), m("a"), m("b"), m("lap"), m("edge")
    ita = float(np.degrees(np.arctan2(Lm - 50, bm)))  # Individual Typology Angle
    tone = ("Very light" if ita > 55 else "Light" if ita > 41 else "Intermediate" if ita > 28
            else "Tan" if ita > 10 else "Brown" if ita > -30 else "Dark")
    cheek_a = (stats["Left cheek"]["a"] + stats["Right cheek"]["a"]) / 2
    clip = lambda v: float(np.clip(v, 0, 1))

    # Heuristic, UNCALIBRATED sub-scores (0-100, higher = more "risk-associated" appearance)
    yellow = clip((bm - 16) / 14) * 100                                    # sallow / yellowish skin
    flow = clip(0.7 * abs(am - 14) / 10 + 0.3 * abs(cheek_a - stats["Forehead"]["a"]) / 8) * 100  # unusual redness/pallor
    collagen = (0.5 * clip((lap - 30) / 270) + 0.5 * clip(edge / 0.12)) * 100  # rough texture / lines
    return dict(found=True, tilt=tilt, image=cv2.cvtColor(annot, cv2.COLOR_BGR2RGB), L=Lm, a=am, b=bm, ita=ita, tone=tone,
                lap=lap, edge=edge, yellow=yellow, flow=flow, collagen=collagen,
                face_score=(yellow + flow + collagen) / 3)


if HAS_WEBRTC:
    class FaceTracker(VideoProcessorBase):
        """Runs on every video frame: finds the face (also when tilted), follows it and draws the analysed regions."""

        def __init__(self):
            self.lock = threading.Lock()
            self.raw = None        # latest un-annotated frame (used when you press "Analyse")
            self.polys = None      # smoothed outlines: face + 3 regions, each 4 corner points
            self.misses = 0
            self.n = 0

        @staticmethod
        def _outlines(box, ang, small_shape, scale):
            x, y, w, h = box
            quads = [(0, 0, 1, 1)] + [tuple(r) for r in ROIS.values()]
            Minv = cv2.invertAffineTransform(_rot(small_shape, ang))
            out = []
            for x0, y0, x1, y1 in quads:
                pts = np.array([[x + x0 * w, y + y0 * h], [x + x1 * w, y + y0 * h],
                                [x + x1 * w, y + y1 * h], [x + x0 * w, y + y1 * h]], np.float32)
                out.append(cv2.transform(pts[None], Minv)[0] / scale)   # back to the original, tilted image
            return np.array(out)

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            with self.lock:
                self.raw = img.copy()
            self.n += 1
            H, W = img.shape[:2]
            scale = 400 / max(H, W) if max(H, W) > 400 else 1.0
            small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # upright search every frame; the slower tilt search only every 3rd frame
            res = find_face(gray, tilts=TILTS if self.n % 3 == 0 else [0], min_size=40)
            if res is not None:
                self.misses = 0
                new = self._outlines(res[0], res[1], gray.shape, scale)
                self.polys = new if self.polys is None else 0.55 * self.polys + 0.45 * new   # smooth movement
            else:
                self.misses += 1
                if self.misses > 20:   # lost the face for about a second
                    self.polys = None
            if self.polys is not None:
                for i, quad in enumerate(self.polys):
                    col = (255, 160, 0) if i == 0 else (0, 220, 0)
                    cv2.polylines(img, [quad.astype(np.int32)], True, col, 2)
                msg, col = "Face tracked - press Analyse", (0, 200, 0)
            else:
                msg, col = "No face found - face the camera, whole face in view", (0, 120, 255)
            cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")


def band(p):
    return ("Low", "🟢") if p < 0.30 else ("Moderate", "🟠") if p < 0.60 else ("High", "🔴")


# ---------------------------------------------------------------- sidebar / navigation
df = load_data()
st.sidebar.title("🩺 Diabetes Risk Screener")
done = st.session_state.clinical is not None
st.sidebar.markdown(f"**1. Clinical measurements** {'✅' if done else '⏳'}")
st.sidebar.markdown(f"**2. Face scan** {'🔓' if done else '🔒 (complete step 1 first)'}")
if st.sidebar.button("Go to Step 1"):
    st.session_state.step = 1
if st.sidebar.button("Go to Step 2", disabled=not done):
    st.session_state.step = 2

# ---- Demo mode (sidebar): preset patients so the app can be shown without typing values or using a camera
DEMO_PATIENTS = {
    "Low-risk example": dict(Age=28.0, Height_cm=172.0, Weight_kg=65.0, BMI=21.97, Blood_Pressure=64.0, Insulin=60.0,
                             Diabetes_Pedigree_Function=0.20),
    "Moderate-risk example": dict(Age=50.0, Height_cm=168.0, Weight_kg=78.0, BMI=27.64, Blood_Pressure=74.0, Insulin=95.0,
                                  Diabetes_Pedigree_Function=0.45),
    "High-risk example": dict(Age=62.0, Height_cm=158.0, Weight_kg=90.0, BMI=36.05, Blood_Pressure=88.0, Insulin=200.0,
                              Diabetes_Pedigree_Function=0.90),
}
DEMO_FACES = {  # SIMULATED face-scan outputs (no photo analysed)
    "Low-risk example": dict(L=68.0, a=13.5, b=15.0, lap=45.0, edge=0.03, yellow=10.0, flow=15.0, collagen=15.0),
    "Moderate-risk example": dict(L=58.0, a=16.0, b=19.0, lap=120.0, edge=0.06, yellow=40.0, flow=45.0, collagen=45.0),
    "High-risk example": dict(L=50.0, a=22.0, b=25.0, lap=240.0, edge=0.10, yellow=75.0, flow=70.0, collagen=80.0),
}
with st.sidebar.expander("🎬 Demo mode", expanded=True):
    demo_pick = st.selectbox("Demo patient", list(DEMO_PATIENTS), index=1)
    if st.button("Load demo patient"):
        st.session_state.clinical = dict(DEMO_PATIENTS[demo_pick])
        st.session_state.demo_level = demo_pick
        st.session_state.step = 1
        st.rerun()
    st.checkbox("Use simulated face scan in Step 2 (no photo)", key="demo_face")
    st.caption("Demo values are made up for presentation. The simulated face scan is not a real measurement.")

if df is None:
    st.info("Place `synthetic_diabetes_dataset.xlsx` next to `app.py`, or upload it in the sidebar.")
    st.stop()
model, cv_auc = train_model(df, tuple(FEATURES))
st.sidebar.caption(f"Model: Logistic Regression trained on {len(df)} synthetic patients. "
                   f"5-fold stratified CV ROC-AUC = {cv_auc:.3f}")

if st.session_state.step == 2 and not done:
    st.session_state.step = 1  # hard lock: step 2 needs step 1 values

# ---------------------------------------------------------------- STEP 1
if st.session_state.step == 1:
    st.title("Step 1 of 2 – Clinical measurements")
    st.caption("Enter your measurements (nothing is pre-filled). BMI is calculated when you save.")
    prev = st.session_state.clinical or {}
    pv = lambda k: float(prev[k]) if k in prev else None
    with st.form("clinical_form"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", 18.0, 100.0, pv("Age"), 0.5, format="%.1f", placeholder="e.g. 45.0")
        height = c2.number_input("Height (cm)", 100.0, 230.0, pv("Height_cm"), 0.5, format="%.1f", placeholder="e.g. 170.0")
        weight = c1.number_input("Weight (kg)", 20.0, 300.0, pv("Weight_kg"), 0.5, format="%.1f", placeholder="e.g. 70.0")
        bp = c2.number_input("Diastolic blood pressure (mm Hg)", 40.0, 140.0, pv("Blood_Pressure"), 0.5,
                             format="%.1f", placeholder="e.g. 75.0")
        insulin = c1.number_input("Insulin (mu U/ml)", 0.0, 900.0, pv("Insulin"), 0.5, format="%.1f",
                                  placeholder="e.g. 90.0")
        dpf = c2.number_input("Diabetes Pedigree Function", 0.0, 2.5, pv("Diabetes_Pedigree_Function"), 0.05,
                              format="%.2f", placeholder="e.g. 0.45",
                              help="Family-history score. Higher = stronger family history of diabetes.")
        submitted = st.form_submit_button("Save measurements & calculate BMI")
    if submitted:
        if None in (age, height, weight, bp, insulin, dpf):
            st.error("Please fill in all six fields.")
        else:
            bmi = weight / (height / 100) ** 2
            st.session_state.clinical = dict(Age=age, Height_cm=height, Weight_kg=weight, BMI=round(bmi, 2),
                                             Blood_Pressure=bp, Insulin=insulin, Diabetes_Pedigree_Function=dpf)
            st.rerun()
    if done:
        st.success("Measurements saved. Step 2 is now unlocked.")
        st.metric("Calculated BMI", f"{st.session_state.clinical['BMI']:.2f}")
        if st.button("Continue to Step 2 → Face scan", type="primary"):
            st.session_state.step = 2
            st.rerun()

# ---------------------------------------------------------------- STEP 2
else:
    clin = st.session_state.clinical
    st.title("Step 2 of 2 – Face scan")
    st.caption("Use even, natural light: face the camera, no filters, no makeup, glasses off.")

    x_row = pd.DataFrame([{f: clin[f] for f in FEATURES}])
    p_model = float(model.predict_proba(x_row)[0, 1])

    demo_on = st.session_state.get("demo_face", False)
    photo, live = None, False
    if not demo_on:
        modes = ["Live camera (follows your face)", "Photo / upload"] if HAS_WEBRTC else ["Photo / upload"]
        mode = st.radio("Camera mode", modes, horizontal=True)
        if not HAS_WEBRTC:
            st.info("Live face tracking needs one extra package: run `python3 -m pip install streamlit-webrtc` "
                    "and restart the app. Until then, use a photo.")
        if not detector_ready():
            st.error(f"The face detector is not available (OpenCV {cv2.__version__}). OpenCV 5 does not include the "
                     "face-detector files. In Terminal run:  python3 -m pip uninstall -y opencv-python "
                     "opencv-python-headless  and then  python3 -m pip install \"opencv-python-headless<5\"  "
                     "and restart the app. Until then nothing can be analysed (demo mode still works).")
        live = mode.startswith("Live")
        if live:
            ctx = webrtc_streamer(key="face-tracker", video_processor_factory=FaceTracker,
                                  media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                                  async_processing=True,
                                  rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            st.caption("Click START, allow the camera, and keep your whole face in view. The green boxes follow your "
                       "face; when they sit on your forehead and cheeks, hold still and press Analyse.")
            if ctx.state.playing and st.button("📸 Analyse current frame", type="primary"):
                vp = ctx.video_processor
                raw = None
                if vp is not None:
                    with vp.lock:
                        raw = None if vp.raw is None else vp.raw.copy()
                if raw is None:
                    st.warning("No camera frame received yet. Wait a second and try again.")
                else:
                    try:
                        res = analyse_face(raw)
                        st.session_state["live_face"] = res
                        if res is None:
                            st.warning("No face found in that frame. Move so your whole face is inside the picture "
                                       "(not cut off at the edge), face the camera, and press Analyse again.")
                    except Exception as e:
                        st.session_state["live_face"] = None
                        st.error(f"Could not analyse this frame ({e}). Try again.")
        else:
            photo = st.camera_input("Take a photo") or st.file_uploader("…or upload a face photo", type=["jpg", "jpeg", "png"])
    face = None
    if demo_on:
        d = dict(DEMO_FACES[st.session_state.get("demo_level", "Moderate-risk example")])
        d["ita"] = float(np.degrees(np.arctan2(d["L"] - 50, d["b"])))
        d["tone"] = ("Very light" if d["ita"] > 55 else "Light" if d["ita"] > 41 else "Intermediate" if d["ita"] > 28
                     else "Tan" if d["ita"] > 10 else "Brown" if d["ita"] > -30 else "Dark")
        d["face_score"] = (d["yellow"] + d["flow"] + d["collagen"]) / 3
        d["image"] = None
        face = d
        st.info("🎬 Demo mode: showing SIMULATED face-scan values (no photo analysed).")
    elif live:
        face = st.session_state.get("live_face")
    elif photo is not None:
        try:
            bgr = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("image could not be read")
            face = analyse_face(bgr)
            if face is None:
                st.error("No face detected in this photo, so nothing was analysed. Retake it with your whole face "
                         "inside the picture (not cut off), facing the camera, in even light.")
        except Exception as e:
            face = None
            st.error(f"Could not process this image ({e}). Retake the photo or upload a JPG/PNG.")

    if face is not None:
        if face["image"] is not None:
            cap = "Regions analysed: forehead and both cheeks"
            if face.get("tilt"):
                cap += f" (head tilt of about {abs(face['tilt'])}° straightened)"
            st.image(face["image"], caption=cap, use_container_width=True)
        t1, t2, t3 = st.tabs(["🎨 Skin colour", "🩸 Facial blood flow", "🧴 Skin & collagen"])
        with t1:
            st.metric("Skin tone (ITA)", f"{face['tone']}  ({face['ita']:.0f}°)")
            c1, c2 = st.columns(2)
            c1.metric("Lightness L*", f"{face['L']:.1f}")
            c2.metric("Yellowness b*", f"{face['b']:.1f}")
            st.progress(int(face["yellow"]), text=f"Yellowish / sallow tone indicator: {face['yellow']:.0f}/100")
        with t2:
            st.metric("Redness a* (perfusion proxy)", f"{face['a']:.1f}", help="Typical skin is roughly 10-18.")
            st.progress(int(face["flow"]), text=f"Unusual redness / pallor indicator: {face['flow']:.0f}/100")
            st.caption("Estimated from skin colour in a single photo. True facial blood-flow analysis (rPPG) needs video.")
        with t3:
            c1, c2 = st.columns(2)
            c1.metric("Texture roughness", f"{face['lap']:.0f}")
            c2.metric("Fine-line / edge density", f"{face['edge'] * 100:.1f}%")
            st.progress(int(face["collagen"]), text=f"Skin texture / collagen-change indicator: {face['collagen']:.0f}/100")

        adj = (face["face_score"] - 50) / 50 * FACE_MAX_ADJUST
        p_final = float(np.clip(p_model + adj, 0, 1))
        label, icon = band(p_final)

        st.divider()
        st.subheader("Approximate diabetes risk")
        c1, c2, c3 = st.columns(3)
        c1.metric("Clinical model (Step 1)", f"{p_model * 100:.0f}%")
        c2.metric("Face scan adjustment", f"{adj * 100:+.1f} pts")
        c3.metric("Combined estimate", f"{p_final * 100:.0f}%")
        st.progress(int(p_final * 100), text=f"{icon} {label} risk")

        coefs = model[-1].coef_[0]
        contrib = coefs * model[0].transform(x_row)[0]
        top = pd.DataFrame({"Feature": FEATURES, "Your value": x_row.iloc[0].values,
                            "Effect on risk": np.where(contrib > 0, "↑ raises", "↓ lowers"),
                            "Strength": np.abs(contrib).round(2)}).sort_values("Strength", ascending=False)
        st.markdown("**Top factors behind the clinical-model risk**")
        st.dataframe(top, hide_index=True, use_container_width=True)

    st.warning("**Not a medical diagnosis.** The clinical model is trained on a *synthetic* dataset. "
               "The face-scan indicators are simple, uncalibrated image heuristics: no dataset in this project links "
               "facial features to diabetes, so they can only nudge the result by ±5 points and are affected by lighting, "
               "camera and skin tone. Confirm any concern with a fasting glucose or HbA1c test from a doctor.")
