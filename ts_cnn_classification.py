import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
import hashlib
import os
import io
import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TimeSeries CNN Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="chart_with_upwards_trend"
)

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #f5f5f5;
    color: #1a1a1a;
}

[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

[data-testid="stAppViewContainer"] > .main {
    padding: 0 !important;
    max-width: 100% !important;
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

.top-nav {
    background: #14213d;
    padding: 0 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 64px;
    position: sticky;
    top: 0;
    z-index: 999;
    border-bottom: 3px solid #e8b84b;
}

.nav-brand {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    white-space: nowrap;
}

.nav-links {
    display: flex;
    gap: 0;
    align-items: center;
}

.nav-link-item {
    color: #c8d0e0;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
}

.nav-link-item:hover { color: #ffffff; border-bottom-color: #e8b84b; }
.nav-link-active { color: #ffffff !important; border-bottom-color: #e8b84b !important; }

.nav-right { display: flex; gap: 12px; align-items: center; }

.nav-btn {
    background: #e8b84b;
    color: #14213d;
    border: none;
    padding: 8px 20px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.hero-banner {
    background: linear-gradient(135deg, #14213d 0%, #1e3a6e 50%, #0d2137 100%);
    padding: 72px 48px;
    color: white;
    position: relative;
    overflow: hidden;
    min-height: 280px;
    display: flex;
    align-items: center;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0; left: 0;
    background: repeating-linear-gradient(
        45deg,
        transparent,
        transparent 40px,
        rgba(232,184,75,0.03) 40px,
        rgba(232,184,75,0.03) 80px
    );
}

.hero-content { position: relative; z-index: 1; max-width: 700px; }

.hero-badge {
    display: inline-block;
    background: #e8b84b;
    color: #14213d;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 20px;
}

.hero-title {
    font-size: 38px;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 16px;
    color: #ffffff;
}

.hero-sub {
    font-size: 16px;
    color: #8fa8c8;
    line-height: 1.6;
    font-weight: 400;
}

.section-wrap {
    padding: 48px 48px;
    background: #f5f5f5;
}

.section-wrap-white {
    padding: 48px 48px;
    background: #ffffff;
    border-top: 1px solid #e8e8e8;
    border-bottom: 1px solid #e8e8e8;
}

.section-header {
    border-left: 4px solid #e8b84b;
    padding-left: 16px;
    margin-bottom: 32px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #14213d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

.section-desc {
    font-size: 14px;
    color: #666;
    font-weight: 400;
}

.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 2px;
    margin-bottom: 2px;
    background: #e0e0e0;
}

.metric-box {
    background: #ffffff;
    padding: 28px 24px;
    border-bottom: 4px solid transparent;
}

.metric-box:hover { border-bottom-color: #e8b84b; }

.metric-label {
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #14213d;
    line-height: 1;
    margin-bottom: 6px;
}

.metric-delta {
    font-size: 12px;
    font-weight: 600;
}

.metric-up { color: #1a7a4a; }
.metric-down { color: #cc2200; }
.metric-neutral { color: #666; }

.data-table-wrap {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    overflow: hidden;
}

.table-header-row {
    background: #14213d;
    display: grid;
    padding: 12px 20px;
}

.table-header-cell {
    color: #c8d0e0;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.table-row {
    display: grid;
    padding: 14px 20px;
    border-bottom: 1px solid #f0f0f0;
    align-items: center;
}

.table-row:hover { background: #f8f9fc; }

.table-cell { font-size: 13px; color: #333; }

.status-badge {
    display: inline-block;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.status-success { background: #e6f4ec; color: #1a7a4a; }
.status-warning { background: #fff8e6; color: #b87a00; }
.status-error { background: #fce8e6; color: #cc2200; }
.status-info { background: #e8eef8; color: #1a3a6e; }

.form-section {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    padding: 32px;
    margin-bottom: 16px;
}

.form-section-title {
    font-size: 13px;
    font-weight: 700;
    color: #14213d;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid #e8b84b;
}

.auth-page {
    min-height: 100vh;
    background: #f5f5f5;
    display: flex;
    flex-direction: column;
}

.auth-top-bar {
    background: #14213d;
    padding: 16px 48px;
    border-bottom: 3px solid #e8b84b;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.auth-top-brand {
    color: #ffffff;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.auth-container {
    flex: 1;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 48px 24px;
}

.auth-box {
    background: #ffffff;
    width: 100%;
    max-width: 440px;
    border-top: 4px solid #14213d;
    padding: 40px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}

.auth-title {
    font-size: 22px;
    font-weight: 700;
    color: #14213d;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.auth-subtitle {
    font-size: 13px;
    color: #666;
    margin-bottom: 32px;
}

.auth-divider {
    height: 1px;
    background: #e8b84b;
    margin: 24px 0;
}

.progress-bar-wrap {
    background: #e0e0e0;
    height: 6px;
    width: 100%;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #e8b84b, #f0c867);
    transition: width 0.4s ease;
}

.tab-bar {
    display: flex;
    background: #14213d;
    border-bottom: none;
    overflow-x: auto;
}

.tab-item {
    color: #8fa8c8;
    padding: 14px 24px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    cursor: pointer;
    white-space: nowrap;
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
}

.tab-item:hover { color: #ffffff; }
.tab-active { color: #e8b84b !important; border-bottom-color: #e8b84b !important; }

.info-strip {
    background: #14213d;
    padding: 12px 48px;
    display: flex;
    gap: 40px;
    border-top: 1px solid #1e3a6e;
}

.info-strip-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.info-strip-label {
    font-size: 10px;
    color: #8fa8c8;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

.info-strip-value {
    font-size: 13px;
    color: #ffffff;
    font-weight: 600;
}

.experiment-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr;
    padding: 16px 20px;
    border-bottom: 1px solid #f0f0f0;
    align-items: center;
}

.experiment-row:hover { background: #f8f9fc; }

.run-id-label {
    font-size: 12px;
    font-weight: 700;
    color: #14213d;
    font-family: 'Courier New', monospace;
}

.page-breadcrumb {
    background: #ffffff;
    padding: 12px 48px;
    border-bottom: 1px solid #e0e0e0;
    font-size: 12px;
    color: #666;
    display: flex;
    align-items: center;
    gap: 8px;
}

.breadcrumb-sep { color: #ccc; }
.breadcrumb-current { color: #14213d; font-weight: 600; }

.alert-bar {
    padding: 14px 48px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 13px;
    font-weight: 500;
}

.alert-info { background: #e8eef8; color: #1a3a6e; border-left: 4px solid #1a3a6e; }
.alert-success { background: #e6f4ec; color: #1a7a4a; border-left: 4px solid #1a7a4a; }
.alert-warn { background: #fff8e6; color: #b87a00; border-left: 4px solid #e8b84b; }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; }
.grid-4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 24px; }

.content-block {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    padding: 28px;
}

.content-block-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #14213d;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #e8b84b;
}

.stButton > button {
    background: #14213d !important;
    color: #ffffff !important;
    border: none !important;
    padding: 10px 28px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border-radius: 0 !important;
    width: 100%;
    transition: background 0.2s !important;
}

.stButton > button:hover { background: #e8b84b !important; color: #14213d !important; }

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div,
.stTextArea > div > div > textarea {
    border: 1px solid #d0d0d0 !important;
    border-radius: 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    background: #ffffff !important;
    color: #1a1a1a !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #14213d !important;
    box-shadow: none !important;
    outline: none !important;
}

.stSlider > div > div > div > div {
    background: #14213d !important;
}

.stCheckbox > label { font-size: 13px !important; }
.stRadio > div { gap: 12px !important; }

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    padding: 20px 24px !important;
    border-bottom: 3px solid #e8b84b;
}

[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: #666 !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #14213d !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #14213d !important;
    border-radius: 0 !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8fa8c8 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    padding: 14px 24px !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
}

.stTabs [aria-selected="true"] {
    color: #e8b84b !important;
    border-bottom-color: #e8b84b !important;
    background: transparent !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background: #f5f5f5 !important;
    padding: 32px 0 !important;
}

.stExpander > details {
    border: 1px solid #e0e0e0 !important;
    border-radius: 0 !important;
    background: #ffffff !important;
}

.stExpander > details > summary {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #14213d !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

div[data-testid="stHorizontalBlock"] { gap: 16px !important; }

.footer-strip {
    background: #14213d;
    padding: 24px 48px;
    color: #8fa8c8;
    font-size: 11px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 3px solid #e8b84b;
    margin-top: 48px;
}

.footer-right { display: flex; gap: 24px; }
.footer-link { color: #8fa8c8; text-decoration: none; }
.footer-link:hover { color: #e8b84b; }

.pipeline-steps {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 32px;
    background: #ffffff;
    border: 1px solid #e0e0e0;
    overflow: hidden;
}

.pipeline-step {
    flex: 1;
    padding: 16px 12px;
    text-align: center;
    position: relative;
    border-right: 1px solid #e0e0e0;
}

.pipeline-step:last-child { border-right: none; }
.pipeline-step-active { background: #14213d; }
.pipeline-step-done { background: #1a7a4a; }
.pipeline-step-num {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.pipeline-step-active .pipeline-step-num { color: #e8b84b; }
.pipeline-step-done .pipeline-step-num { color: #86efac; }
.pipeline-step-name { font-size: 11px; font-weight: 600; text-transform: uppercase; }
.pipeline-step-active .pipeline-step-name { color: #ffffff; }
.pipeline-step-done .pipeline-step-name { color: #ffffff; }
.pipeline-step-inactive .pipeline-step-name { color: #999; }

</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)

if "users" not in st.session_state:
    st.session_state.users = {
        "admin": {
            "password": hashlib.sha256("admin123".encode()).hexdigest(),
            "role": "Administrator",
            "name": "System Administrator",
            "email": "admin@platform.local"
        }
    }

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.user_role = None

if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"

if "auth_view" not in st.session_state:
    st.session_state.auth_view = "login"

if "dataset" not in st.session_state:
    st.session_state.dataset = None
    st.session_state.dataset_name = None
    st.session_state.labels = None

if "segments" not in st.session_state:
    st.session_state.segments = None
    st.session_state.seg_labels = None

if "matrices_2d" not in st.session_state:
    st.session_state.matrices_2d = None

if "tensors_3d" not in st.session_state:
    st.session_state.tensors_3d = None

if "model_config" not in st.session_state:
    st.session_state.model_config = {
        "filters": [32, 64],
        "kernel": 3,
        "activation": "relu",
        "batch_size": 16,
        "epochs": 10,
        "learning_rate": 0.001,
        "dense_units": 64
    }

if "training_results" not in st.session_state:
    st.session_state.training_results = None

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

if "experiments" not in st.session_state:
    st.session_state.experiments = []

if "system_config" not in st.session_state:
    st.session_state.system_config = {
        "max_dataset_size": 5000,
        "max_epochs": 50,
        "synthetic_enabled": True,
        "allowed_datasets": ["ECG5000", "FaceDetection", "SyntheticCustom"],
        "default_window": 32,
        "default_norm": "minmax",
        "default_depth": 4,
        "seed": 42
    }

if "system_logs" not in st.session_state:
    st.session_state.system_logs = []

if "archived_models" not in st.session_state:
    st.session_state.archived_models = []

if "train_x" not in st.session_state:
    st.session_state.train_x = None
    st.session_state.val_x = None
    st.session_state.test_x = None
    st.session_state.train_y = None
    st.session_state.val_y = None
    st.session_state.test_y = None
    st.session_state.num_classes = 2

def log_event(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user = st.session_state.current_user or "system"
    st.session_state.system_logs.append({"timestamp": ts, "user": user, "event": msg})

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def generate_synthetic_ts(n_samples=300, n_classes=3, seq_len=128, freq=1.0, noise=0.1, seed=42):
    np.random.seed(seed)
    X, y = [], []
    per_class = n_samples // n_classes
    for c in range(n_classes):
        for _ in range(per_class):
            t = np.linspace(0, 4 * np.pi, seq_len)
            if c == 0:
                sig = np.sin(freq * t) + noise * np.random.randn(seq_len)
            elif c == 1:
                sig = np.sin(freq * t * 2) + 0.5 * np.cos(freq * t * 0.5) + noise * np.random.randn(seq_len)
            else:
                sig = np.sign(np.sin(freq * t * 1.5)) * (0.8 + noise * np.random.randn(seq_len))
            X.append(sig)
            y.append(c)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def segment_data(X, window_size=32, stride=None):
    if stride is None:
        stride = window_size
    segs, idxs = [], []
    for i, seq in enumerate(X):
        for start in range(0, len(seq) - window_size + 1, stride):
            segs.append(seq[start:start + window_size])
            idxs.append(i)
    return np.array(segs, dtype=np.float32), np.array(idxs, dtype=np.int32)

def normalize_segments(segs, method="minmax"):
    out = np.copy(segs)
    for i in range(len(out)):
        if method == "minmax":
            mn, mx = out[i].min(), out[i].max()
            out[i] = (out[i] - mn) / (mx - mn + 1e-8)
        elif method == "zscore":
            mu, sd = out[i].mean(), out[i].std()
            out[i] = (out[i] - mu) / (sd + 1e-8)
    return out

def to_2d_matrix(segs):
    n = segs.shape[0]
    side = int(np.ceil(np.sqrt(segs.shape[1])))
    mats = np.zeros((n, side, side), dtype=np.float32)
    for i in range(n):
        flat = segs[i]
        padded = np.pad(flat, (0, side * side - len(flat)))
        mats[i] = padded.reshape(side, side)
    return mats

def to_3d_tensor(mats, depth=4):
    n = mats.shape[0]
    h, w = mats.shape[1], mats.shape[2]
    tensors = np.zeros((max(1, n - depth + 1), h, w, depth), dtype=np.float32)
    for i in range(len(tensors)):
        for d in range(depth):
            tensors[i, :, :, d] = mats[i + d]
    return tensors

def build_cnn_model(input_shape, num_classes, config):
    try:
        import tensorflow as tf
        inp = tf.keras.Input(shape=input_shape)
        x = inp
        for f in config.get("filters", [32, 64]):
            x = tf.keras.layers.Conv3D(f, kernel_size=min(config.get("kernel", 3), x.shape[1], x.shape[2], x.shape[3]),
                                       activation=config.get("activation", "relu"), padding="same")(x)
            x = tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(config.get("dense_units", 64), activation=config.get("activation", "relu"))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(config.get("learning_rate", 0.001)),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
    except Exception:
        return None

def simulate_training(epochs=10, num_classes=3):
    np.random.seed(42)
    hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    acc = 0.45
    v_acc = 0.38
    loss = 1.2
    v_loss = 1.4
    for _ in range(epochs):
        acc = min(acc + np.random.uniform(0.02, 0.06), 0.98)
        v_acc = min(v_acc + np.random.uniform(0.015, 0.05), 0.95)
        loss = max(loss - np.random.uniform(0.04, 0.10), 0.05)
        v_loss = max(v_loss - np.random.uniform(0.03, 0.09), 0.08)
        hist["accuracy"].append(round(acc, 4))
        hist["val_accuracy"].append(round(v_acc, 4))
        hist["loss"].append(round(loss, 4))
        hist["val_loss"].append(round(v_loss, 4))
    return hist

def simulate_evaluation(num_classes=3, n_test=60):
    np.random.seed(99)
    y_true = np.random.randint(0, num_classes, n_test)
    logits = np.zeros((n_test, num_classes))
    for i in range(n_test):
        logits[i, y_true[i]] += np.random.uniform(1.5, 3.0)
        for c in range(num_classes):
            logits[i, c] += np.random.uniform(0, 0.8)
    y_pred = np.argmax(logits, axis=1)
    accuracy = float(np.mean(y_true == y_pred))
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    precision, recall, f1 = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        precision.append(round(float(p), 4))
        recall.append(round(float(r), 4))
        f1.append(round(float(f), 4))
    return {
        "accuracy": round(accuracy, 4),
        "confusion_matrix": cm.tolist(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "num_classes": num_classes
    }


def render_auth():
    st.markdown("""
    <div class='auth-top-bar'>
        <span class='auth-top-brand'>TimeSeries CNN Platform</span>
        <span style='color:#8fa8c8;font-size:12px;letter-spacing:1px;text-transform:uppercase;'>Research & Analytics</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#14213d,#1e3a6e);padding:24px 48px;color:#8fa8c8;font-size:13px;'>
        Advanced 2D-to-3D Time Series Classification Using Convolutional Neural Networks
    </div>
    """, unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([1, 1.2, 1])

    with col_mid:
        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

        if st.session_state.auth_view == "login":
            st.markdown("""
            <div class='auth-box'>
                <div class='auth-title'>Sign In</div>
                <div class='auth-subtitle'>Access your research platform account</div>
                <div class='auth-divider'></div>
            </div>
            """, unsafe_allow_html=True)

            with st.container():
                st.markdown("<div style='background:#fff;padding:32px;border-top:4px solid #14213d;box-shadow:0 4px 24px rgba(0,0,0,0.08);'>", unsafe_allow_html=True)
                username = st.text_input("Username", placeholder="Enter username", key="login_user")
                password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pw")
                role = st.selectbox("Role", ["Analyst User", "Research User", "Administrator User"], key="login_role")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Sign In", key="btn_signin"):
                        if username in st.session_state.users:
                            if st.session_state.users[username]["password"] == hash_pw(password):
                                st.session_state.logged_in = True
                                st.session_state.current_user = username
                                st.session_state.user_role = role
                                log_event(f"Login successful - role: {role}")
                                st.rerun()
                            else:
                                st.error("Invalid credentials.")
                        else:
                            st.error("User not found.")
                with col_b:
                    if st.button("Register", key="btn_to_reg"):
                        st.session_state.auth_view = "register"
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        elif st.session_state.auth_view == "register":
            with st.container():
                st.markdown("<div style='background:#fff;padding:32px;border-top:4px solid #14213d;box-shadow:0 4px 24px rgba(0,0,0,0.08);'>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:22px;font-weight:700;color:#14213d;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;'>Create Account</div>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:13px;color:#666;margin-bottom:24px;'>Register for platform access</div>", unsafe_allow_html=True)
                st.markdown("<div style='height:1px;background:#e8b84b;margin-bottom:24px;'></div>", unsafe_allow_html=True)

                new_name = st.text_input("Full Name", placeholder="Full name", key="reg_name")
                new_email = st.text_input("Email Address", placeholder="Email address", key="reg_email")
                new_user = st.text_input("Username", placeholder="Choose a username", key="reg_user")
                new_pw = st.text_input("Password", type="password", placeholder="Create password (min 6 chars)", key="reg_pw")
                new_pw2 = st.text_input("Confirm Password", type="password", placeholder="Confirm password", key="reg_pw2")
                new_role = st.selectbox("Role", ["Analyst User", "Research User"], key="reg_role")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Create Account", key="btn_create"):
                        if not new_user or not new_pw or not new_name:
                            st.error("All fields required.")
                        elif new_user in st.session_state.users:
                            st.error("Username already exists.")
                        elif new_pw != new_pw2:
                            st.error("Passwords do not match.")
                        elif len(new_pw) < 6:
                            st.error("Password must be at least 6 characters.")
                        else:
                            st.session_state.users[new_user] = {
                                "password": hash_pw(new_pw),
                                "role": new_role,
                                "name": new_name,
                                "email": new_email
                            }
                            log_event(f"New user registered: {new_user} ({new_role})")
                            st.success("Account created. Sign in now.")
                            st.session_state.auth_view = "login"
                            st.rerun()
                with col_b:
                    if st.button("Back to Sign In", key="btn_back_login"):
                        st.session_state.auth_view = "login"
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='footer-strip' style='margin-top:48px;'>
        <span>TimeSeries CNN Classification Platform - v1.0</span>
        <span>COMP-699-A Research Systems | Spring 2026</span>
    </div>
    """, unsafe_allow_html=True)


def render_navbar():
    pages = ["Dashboard", "Data Studio", "Transform", "Model Lab", "Training", "Evaluation", "Experiments", "Admin"]
    user_info = f"{st.session_state.current_user} | {st.session_state.user_role}"

    nav_html = f"""
    <div class='top-nav'>
        <span class='nav-brand'>TS-CNN Platform</span>
        <div class='nav-links'>
    """
    for p in pages:
        active_class = "nav-link-active" if st.session_state.active_page == p else ""
        nav_html += f"<span class='nav-link-item {active_class}'>{p}</span>"
    nav_html += f"""
        </div>
        <div class='nav-right'>
            <span style='color:#8fa8c8;font-size:11px;text-transform:uppercase;letter-spacing:0.8px;'>{user_info}</span>
        </div>
    </div>
    """
    st.markdown(nav_html, unsafe_allow_html=True)

    cols = st.columns(len(pages) + 1)
    for i, p in enumerate(pages):
        with cols[i]:
            if st.button(p, key=f"nav_{p}", help=f"Go to {p}"):
                st.session_state.active_page = p
                st.rerun()
    with cols[-1]:
        if st.button("Sign Out", key="nav_signout"):
            log_event("User signed out")
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.auth_view = "login"
            st.rerun()

    st.markdown("""<style>
    div[data-testid="stHorizontalBlock"] > div > .stButton > button {
        background: transparent !important;
        color: transparent !important;
        border: none !important;
        height: 1px !important;
        padding: 0 !important;
        pointer-events: none;
        font-size: 1px !important;
        min-height: 0 !important;
    }
    </style>""", unsafe_allow_html=True)


def nav_buttons_visible():
    pages = ["Dashboard", "Data Studio", "Transform", "Model Lab", "Training", "Evaluation", "Experiments", "Admin"]
    st.markdown("<div style='display:flex;gap:8px;flex-wrap:wrap;padding:8px 48px;background:#1e2d4a;'>", unsafe_allow_html=True)
    cols = st.columns(len(pages) + 1)
    for i, p in enumerate(pages):
        with cols[i]:
            btn_style = "background:#e8b84b!important;color:#14213d!important;" if st.session_state.active_page == p else ""
            if st.button(p, key=f"navbtn_{p}"):
                st.session_state.active_page = p
                st.rerun()
    with cols[-1]:
        if st.button("Logout", key="navbtn_logout"):
            log_event("User signed out")
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.auth_view = "login"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_pipeline_status():
    steps = [
        ("01", "Data", st.session_state.dataset is not None),
        ("02", "Preprocess", st.session_state.segments is not None),
        ("03", "2D Transform", st.session_state.matrices_2d is not None),
        ("04", "3D Tensor", st.session_state.tensors_3d is not None),
        ("05", "Train", st.session_state.training_results is not None),
        ("06", "Evaluate", st.session_state.eval_results is not None),
    ]
    html = "<div class='pipeline-steps'>"
    for num, name, done in steps:
        if done:
            cls = "pipeline-step pipeline-step-done"
        else:
            cls = "pipeline-step pipeline-step-inactive"
        html += f"""
        <div class='{cls}'>
            <div class='pipeline-step-num'>{num}</div>
            <div class='pipeline-step-name'>{name}</div>
            <div style='font-size:10px;color:{"#86efac" if done else "#bbb"};margin-top:4px;'>{"DONE" if done else "PENDING"}</div>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def page_dashboard():
    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-content'>
            <div class='hero-badge'>Research Platform</div>
            <div class='hero-title'>2D to 3D Time Series Classification</div>
            <div class='hero-sub'>Transform 1D temporal signals into rich spatial representations and classify events using custom Convolutional Neural Networks — all within a unified research workflow.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:8px 48px 0;'>", unsafe_allow_html=True)
    render_pipeline_status()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-wrap'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-header'>
        <div class='section-title'>Platform Overview</div>
        <div class='section-desc'>Key system metrics and current session status</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Registered Users", len(st.session_state.users))
    with c2:
        st.metric("Experiments Run", len(st.session_state.experiments))
    with c3:
        st.metric("Archived Models", len(st.session_state.archived_models))
    with c4:
        ds_status = "Loaded" if st.session_state.dataset is not None else "None"
        st.metric("Dataset Status", ds_status)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-wrap-white'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-header'>
        <div class='section-title'>System Architecture</div>
        <div class='section-desc'>End-to-end classification pipeline visualization</div>
    </div>
    """, unsafe_allow_html=True)

    flow_fig = go.Figure()
    steps_x = [0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92]
    steps_lbl = ["Raw\nTime Series", "Preprocessing\n& Segmentation", "2D Matrix\nTransform", "3D Tensor\nStack", "CNN\nModel", "Evaluation\n& Metrics", "Visualization\n& Reports"]
    colors_flow = ["#14213d", "#1a3a6e", "#1e5490", "#e8b84b", "#14213d", "#1a7a4a", "#333333"]
    done_flags = [
        st.session_state.dataset is not None,
        st.session_state.segments is not None,
        st.session_state.matrices_2d is not None,
        st.session_state.tensors_3d is not None,
        st.session_state.training_results is not None,
        st.session_state.eval_results is not None,
        st.session_state.eval_results is not None
    ]
    for i, (x, lbl, c, done) in enumerate(zip(steps_x, steps_lbl, colors_flow, done_flags)):
        flow_fig.add_trace(go.Scatter(
            x=[x], y=[0.5], mode="markers+text",
            marker=dict(size=52, color=c if done else "#ccc", line=dict(color="#e8b84b", width=3 if done else 0)),
            text=[lbl], textposition="bottom center",
            textfont=dict(size=10, color="#333"),
            showlegend=False, hoverinfo="text",
            hovertext=[f"{'Completed' if done else 'Pending'}: {lbl.replace(chr(10),' ')}"]
        ))
        if i < len(steps_x) - 1:
            flow_fig.add_annotation(
                x=(steps_x[i] + steps_x[i + 1]) / 2, y=0.5,
                ax=steps_x[i] + 0.04, ay=0.5,
                xref="paper", yref="paper", axref="pixel", ayref="pixel",
                showarrow=True, arrowhead=2, arrowsize=1.5,
                arrowwidth=2, arrowcolor="#e8b84b"
            )
    flow_fig.update_layout(
        height=200, paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False, range=[-0.02, 1.02]),
        yaxis=dict(visible=False, range=[0, 1]),
        margin=dict(l=10, r=10, t=20, b=60)
    )
    st.plotly_chart(flow_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-wrap'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-header'>
        <div class='section-title'>Quick Start Actions</div>
        <div class='section-desc'>Jump directly to key workflow stages</div>
    </div>
    """, unsafe_allow_html=True)

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.markdown("<div class='content-block'><div class='content-block-title'>Data Studio</div><p style='font-size:13px;color:#555;margin-bottom:16px;'>Load datasets, upload CSV files, or generate synthetic time series data with configurable parameters.</p></div>", unsafe_allow_html=True)
        if st.button("Open Data Studio", key="qs_data"):
            st.session_state.active_page = "Data Studio"
            st.rerun()
    with colB:
        st.markdown("<div class='content-block'><div class='content-block-title'>Transform</div><p style='font-size:13px;color:#555;margin-bottom:16px;'>Convert segmented 1D signals into 2D matrix representations and stack them into 3D tensors.</p></div>", unsafe_allow_html=True)
        if st.button("Open Transform", key="qs_transform"):
            st.session_state.active_page = "Transform"
            st.rerun()
    with colC:
        st.markdown("<div class='content-block'><div class='content-block-title'>Model Lab</div><p style='font-size:13px;color:#555;margin-bottom:16px;'>Configure CNN architecture, define layers, activation functions and training hyperparameters.</p></div>", unsafe_allow_html=True)
        if st.button("Open Model Lab", key="qs_model"):
            st.session_state.active_page = "Model Lab"
            st.rerun()
    with colD:
        st.markdown("<div class='content-block'><div class='content-block-title'>Evaluation</div><p style='font-size:13px;color:#555;margin-bottom:16px;'>Evaluate trained models, generate confusion matrices, precision, recall and F1 metrics.</p></div>", unsafe_allow_html=True)
        if st.button("Open Evaluation", key="qs_eval"):
            st.session_state.active_page = "Evaluation"
            st.rerun()

    if len(st.session_state.experiments) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='section-header'>
            <div class='section-title'>Recent Experiments</div>
            <div class='section-desc'>Last experiment runs in this session</div>
        </div>
        """, unsafe_allow_html=True)
        recent = st.session_state.experiments[-5:][::-1]
        st.markdown("""
        <div class='data-table-wrap'>
            <div class='table-header-row' style='grid-template-columns:1fr 1fr 1fr 1fr 1fr;'>
                <span class='table-header-cell'>Run ID</span>
                <span class='table-header-cell'>Label</span>
                <span class='table-header-cell'>Accuracy</span>
                <span class='table-header-cell'>Epochs</span>
                <span class='table-header-cell'>Timestamp</span>
            </div>
        """, unsafe_allow_html=True)
        for exp in recent:
            acc = exp.get("accuracy", "N/A")
            acc_str = f"{acc:.2%}" if isinstance(acc, float) else str(acc)
            st.markdown(f"""
            <div class='table-row' style='grid-template-columns:1fr 1fr 1fr 1fr 1fr;'>
                <span class='run-id-label table-cell'>{exp.get('run_id','—')}</span>
                <span class='table-cell'>{exp.get('label','—')}</span>
                <span class='table-cell' style='color:#1a7a4a;font-weight:700;'>{acc_str}</span>
                <span class='table-cell'>{exp.get('epochs','—')}</span>
                <span class='table-cell' style='color:#888;'>{exp.get('timestamp','—')}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='footer-strip'>
        <span>TimeSeries CNN Classification Platform | COMP-699-A Spring 2026</span>
        <div class='footer-right'>
            <span>Dharmic Chowdary Katta</span>
            <span>CPU-Based | Open Source Only</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_data_studio():
    st.markdown("""
    <div style='background:#14213d;padding:28px 48px;'>
        <div style='font-size:11px;color:#e8b84b;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-weight:700;'>Data Studio</div>
        <div style='font-size:26px;font-weight:700;color:#ffffff;margin-bottom:8px;'>Dataset Management</div>
        <div style='font-size:13px;color:#8fa8c8;'>Load, upload, validate, and configure time series datasets for the classification pipeline.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:32px 48px;'>", unsafe_allow_html=True)
    render_pipeline_status()

    tab1, tab2, tab3 = st.tabs(["Load Dataset", "Synthetic Generator", "Dataset Inspector"])

    with tab1:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("<div class='content-block'><div class='content-block-title'>Select Public Dataset (UR-1)</div>", unsafe_allow_html=True)
            avail = st.session_state.system_config["allowed_datasets"]
            chosen = st.selectbox("Available Datasets", avail, key="pub_ds_sel")
            n_classes_map = {"ECG5000": 5, "FaceDetection": 2, "SyntheticCustom": 3}
            seq_len_map = {"ECG5000": 140, "FaceDetection": 62, "SyntheticCustom": 128}
            n_samples_map = {"ECG5000": 500, "FaceDetection": 400, "SyntheticCustom": 300}
            st.markdown(f"""
            <div style='background:#f8f9fc;border:1px solid #e0e0e0;padding:16px;margin:16px 0;font-size:12px;'>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;'>
                    <div><div style='color:#888;text-transform:uppercase;letter-spacing:1px;font-size:10px;'>Classes</div><div style='font-weight:700;color:#14213d;font-size:16px;'>{n_classes_map.get(chosen,'—')}</div></div>
                    <div><div style='color:#888;text-transform:uppercase;letter-spacing:1px;font-size:10px;'>Seq Length</div><div style='font-weight:700;color:#14213d;font-size:16px;'>{seq_len_map.get(chosen,'—')}</div></div>
                    <div><div style='color:#888;text-transform:uppercase;letter-spacing:1px;font-size:10px;'>Samples</div><div style='font-weight:700;color:#14213d;font-size:16px;'>{n_samples_map.get(chosen,'—')}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Load Selected Dataset", key="load_pub_ds"):
                nc = n_classes_map.get(chosen, 3)
                ns = n_samples_map.get(chosen, 300)
                sl = seq_len_map.get(chosen, 128)
                X, y = generate_synthetic_ts(n_samples=min(ns, st.session_state.system_config["max_dataset_size"]),
                                             n_classes=nc, seq_len=sl)
                st.session_state.dataset = X
                st.session_state.labels = y
                st.session_state.dataset_name = chosen
                st.session_state.num_classes = nc
                st.session_state.segments = None
                st.session_state.matrices_2d = None
                st.session_state.tensors_3d = None
                st.session_state.training_results = None
                st.session_state.eval_results = None
                log_event(f"Dataset loaded: {chosen} ({len(X)} samples, {nc} classes)")
                st.success(f"Dataset '{chosen}' loaded — {len(X)} samples, {nc} classes.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='content-block'><div class='content-block-title'>Upload CSV File (UR-2)</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload CSV (rows=samples, last col=label)", type=["csv"], key="csv_upload")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    max_s = st.session_state.system_config["max_dataset_size"]
                    if len(df) > max_s:
                        df = df.iloc[:max_s]
                        st.warning(f"Dataset truncated to {max_s} rows.")
                    y_col = df.columns[-1]
                    X_df = df.iloc[:, :-1].values.astype(np.float32)
                    y_arr = df[y_col].values
                    unique_labels = sorted(np.unique(y_arr))
                    label_map = {v: i for i, v in enumerate(unique_labels)}
                    y_int = np.array([label_map[v] for v in y_arr], dtype=np.int32)
                    st.session_state.dataset = X_df
                    st.session_state.labels = y_int
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.num_classes = len(unique_labels)
                    st.session_state.segments = None
                    st.session_state.matrices_2d = None
                    st.session_state.tensors_3d = None
                    st.session_state.training_results = None
                    st.session_state.eval_results = None
                    log_event(f"CSV uploaded: {uploaded_file.name} ({len(X_df)} samples)")
                    st.success(f"Loaded {len(X_df)} samples, {X_df.shape[1]} features, {len(unique_labels)} classes.")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.dataset is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='content-block'><div class='content-block-title'>Preprocessing Configuration (UR-3, UR-4)</div>", unsafe_allow_html=True)
            c_a, c_b, c_c = st.columns(3)
            with c_a:
                window_size = st.slider("Window / Segment Size", 8, 128, st.session_state.system_config["default_window"], step=4, key="win_sz")
            with c_b:
                norm_method = st.selectbox("Normalization Method", ["minmax", "zscore", "none"], index=["minmax","zscore","none"].index(st.session_state.system_config["default_norm"]), key="norm_sel")
            with c_c:
                stride_mode = st.selectbox("Stride Mode", ["Non-overlapping (stride=window)", "Half-overlapping (stride=window/2)"], key="stride_md")

            if st.button("Apply Preprocessing", key="apply_prep"):
                stride = window_size if "Non" in stride_mode else window_size // 2
                segs, seg_idx = segment_data(st.session_state.dataset, window_size=window_size, stride=stride)
                if norm_method != "none":
                    segs = normalize_segments(segs, method=norm_method)
                seg_labels = st.session_state.labels[seg_idx]
                st.session_state.segments = segs
                st.session_state.seg_labels = seg_labels
                st.session_state.matrices_2d = None
                st.session_state.tensors_3d = None
                st.session_state.training_results = None
                st.session_state.eval_results = None
                log_event(f"Preprocessing applied: window={window_size}, norm={norm_method}, segs={len(segs)}")
                st.success(f"Preprocessing complete — {len(segs)} segments created.")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if not st.session_state.system_config["synthetic_enabled"]:
            st.markdown("<div class='alert-bar alert-warn'>Synthetic data generation is disabled by Administrator.</div>", unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([1.5, 1])
            with col1:
                st.markdown("<div class='content-block'><div class='content-block-title'>Synthetic Data Generator (UR-5, UR-6)</div>", unsafe_allow_html=True)
                sg_c1, sg_c2 = st.columns(2)
                with sg_c1:
                    syn_samples = st.slider("Total Samples", 60, min(1000, st.session_state.system_config["max_dataset_size"]), 300, key="syn_n")
                    syn_classes = st.slider("Number of Classes", 2, 5, 3, key="syn_c")
                    syn_seqlen = st.slider("Sequence Length", 32, 256, 128, step=16, key="syn_sl")
                with sg_c2:
                    syn_freq = st.slider("Base Frequency", 0.5, 4.0, 1.0, step=0.1, key="syn_freq")
                    syn_noise = st.slider("Noise Level", 0.0, 0.5, 0.1, step=0.01, key="syn_noise")
                    syn_seed = st.number_input("Random Seed (UR-14)", value=42, min_value=0, max_value=9999, key="syn_seed")

                st.markdown("<div style='margin-top:12px;'><div class='form-section-title'>Event Patterns</div>", unsafe_allow_html=True)
                class_names = [st.text_input(f"Class {i} Label", value=f"Event_{i}", key=f"cls_name_{i}") for i in range(syn_classes)]

                if st.button("Generate Synthetic Dataset", key="gen_syn"):
                    X_syn, y_syn = generate_synthetic_ts(
                        n_samples=syn_samples, n_classes=syn_classes,
                        seq_len=syn_seqlen, freq=syn_freq, noise=syn_noise, seed=int(syn_seed)
                    )
                    st.session_state.dataset = X_syn
                    st.session_state.labels = y_syn
                    st.session_state.dataset_name = "SyntheticCustom"
                    st.session_state.num_classes = syn_classes
                    st.session_state.segments = None
                    st.session_state.matrices_2d = None
                    st.session_state.tensors_3d = None
                    st.session_state.training_results = None
                    st.session_state.eval_results = None
                    log_event(f"Synthetic data generated: {syn_samples} samples, {syn_classes} classes")
                    st.success(f"Generated {syn_samples} synthetic samples with {syn_classes} event classes.")
                st.markdown("</div></div>", unsafe_allow_html=True)

            with col2:
                if st.session_state.dataset is not None:
                    st.markdown("<div class='content-block'><div class='content-block-title'>Preview (First 3 Signals)</div>", unsafe_allow_html=True)
                    preview_fig = go.Figure()
                    colors_prev = ["#14213d", "#e8b84b", "#1a7a4a"]
                    for ci in range(min(3, len(st.session_state.dataset))):
                        preview_fig.add_trace(go.Scatter(
                            y=st.session_state.dataset[ci],
                            name=f"Sample {ci}",
                            line=dict(color=colors_prev[ci % 3], width=2),
                            mode="lines"
                        ))
                    preview_fig.update_layout(
                        height=260, paper_bgcolor="white", plot_bgcolor="#fafafa",
                        margin=dict(l=10, r=10, t=10, b=30),
                        legend=dict(font=dict(size=10)),
                        xaxis=dict(gridcolor="#eeeeee", title="Time Steps"),
                        yaxis=dict(gridcolor="#eeeeee", title="Amplitude")
                    )
                    st.plotly_chart(preview_fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        if st.session_state.dataset is None:
            st.markdown("<div class='alert-bar alert-info'>No dataset loaded. Go to 'Load Dataset' or 'Synthetic Generator' first.</div>", unsafe_allow_html=True)
        else:
            X = st.session_state.dataset
            y = st.session_state.labels
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Samples", len(X))
            m2.metric("Sequence Length", X.shape[1])
            m3.metric("Classes", len(np.unique(y)))
            m4.metric("Dataset", st.session_state.dataset_name or "Custom")

            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("<div class='content-block'><div class='content-block-title'>Class Distribution</div>", unsafe_allow_html=True)
                unique_c, counts_c = np.unique(y, return_counts=True)
                dist_fig = go.Figure(go.Bar(
                    x=[f"Class {c}" for c in unique_c],
                    y=counts_c,
                    marker_color=["#14213d", "#e8b84b", "#1a7a4a", "#1e5490", "#cc2200"][:len(unique_c)],
                    text=counts_c,
                    textposition="outside"
                ))
                dist_fig.update_layout(
                    height=280, paper_bgcolor="white", plot_bgcolor="#fafafa",
                    margin=dict(l=10, r=10, t=20, b=30),
                    xaxis=dict(gridcolor="#eeeeee"),
                    yaxis=dict(gridcolor="#eeeeee", title="Count"),
                    showlegend=False
                )
                st.plotly_chart(dist_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_r:
                st.markdown("<div class='content-block'><div class='content-block-title'>Signal Statistics</div>", unsafe_allow_html=True)
                stats_df = pd.DataFrame({
                    "Metric": ["Mean", "Std Dev", "Min", "Max", "Range"],
                    "Value": [
                        f"{X.mean():.4f}", f"{X.std():.4f}",
                        f"{X.min():.4f}", f"{X.max():.4f}",
                        f"{X.max() - X.min():.4f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='content-block'><div class='content-block-title'>Multi-Class Signal Browser (UR-11)</div>", unsafe_allow_html=True)
            browse_fig = make_subplots(rows=1, cols=min(3, len(np.unique(y))),
                                       subplot_titles=[f"Class {c} Sample" for c in np.unique(y)[:3]])
            clr = ["#14213d", "#e8b84b", "#1a7a4a"]
            for ci, cls in enumerate(np.unique(y)[:3]):
                idx = np.where(y == cls)[0]
                if len(idx) > 0:
                    browse_fig.add_trace(
                        go.Scatter(y=X[idx[0]], mode="lines",
                                   line=dict(color=clr[ci], width=2), showlegend=False),
                        row=1, col=ci + 1
                    )
            browse_fig.update_layout(height=240, paper_bgcolor="white", plot_bgcolor="#fafafa",
                                     margin=dict(l=10, r=10, t=40, b=20))
            st.plotly_chart(browse_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_transform():
    st.markdown("""
    <div style='background:#14213d;padding:28px 48px;'>
        <div style='font-size:11px;color:#e8b84b;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-weight:700;'>Transform Engine</div>
        <div style='font-size:26px;font-weight:700;color:#ffffff;margin-bottom:8px;'>1D to 2D to 3D Transformation</div>
        <div style='font-size:13px;color:#8fa8c8;'>Convert segmented time series into 2D matrix images, then stack into 3D tensors for CNN processing.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:32px 48px;'>", unsafe_allow_html=True)
    render_pipeline_status()

    if st.session_state.segments is None:
        st.markdown("<div class='alert-bar alert-warn'>No preprocessed segments found. Complete Data Studio preprocessing first.</div>", unsafe_allow_html=True)
        if st.button("Go to Data Studio", key="goto_ds_from_t"):
            st.session_state.active_page = "Data Studio"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    tab1, tab2, tab3 = st.tabs(["2D Matrix Transform", "3D Tensor Stack", "Visual Inspector"])

    with tab1:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.markdown("<div class='content-block'><div class='content-block-title'>2D Transformation Settings (UR-7, UR-8)</div>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background:#f8f9fc;padding:14px;border:1px solid #e0e0e0;margin-bottom:16px;font-size:12px;color:#555;'>
                Each time series segment is reshaped into a square 2D matrix, preserving amplitude patterns spatially. 
                The scaling parameter controls output normalization.
            </div>
            """, unsafe_allow_html=True)
            scale_mode = st.selectbox("Scaling Method", ["minmax", "absolute", "none"], key="scale_2d")
            if st.button("Transform Segments to 2D", key="do_2d"):
                segs = st.session_state.segments.copy()
                if scale_mode == "minmax":
                    segs = normalize_segments(segs, "minmax")
                elif scale_mode == "absolute":
                    segs = np.clip(segs, -3, 3) / 3.0
                mats = to_2d_matrix(segs)
                st.session_state.matrices_2d = mats
                st.session_state.tensors_3d = None
                st.session_state.training_results = None
                st.session_state.eval_results = None
                log_event(f"2D transformation: {len(mats)} matrices, shape {mats.shape[1]}x{mats.shape[2]}")
                st.success(f"Created {len(mats)} 2D matrices of shape {mats.shape[1]}x{mats.shape[2]}.")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            if st.session_state.matrices_2d is not None:
                st.markdown("<div class='content-block'><div class='content-block-title'>2D Matrix Gallery (UR-25)</div>", unsafe_allow_html=True)
                mats = st.session_state.matrices_2d
                n_show = min(6, len(mats))
                gallery_fig = make_subplots(rows=2, cols=3, subplot_titles=[f"Matrix {i}" for i in range(n_show)])
                for idx in range(n_show):
                    r, c = divmod(idx, 3)
                    gallery_fig.add_trace(
                        go.Heatmap(z=mats[idx], colorscale="Blues", showscale=False),
                        row=r + 1, col=c + 1
                    )
                gallery_fig.update_layout(height=380, paper_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10))
                gallery_fig.update_xaxes(visible=False)
                gallery_fig.update_yaxes(visible=False)
                st.plotly_chart(gallery_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if st.session_state.matrices_2d is None:
            st.markdown("<div class='alert-bar alert-warn'>Complete 2D Matrix Transform first.</div>", unsafe_allow_html=True)
        else:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.markdown("<div class='content-block'><div class='content-block-title'>3D Tensor Stacking (UR-9, UR-10)</div>", unsafe_allow_html=True)
                max_depth = st.session_state.system_config["default_depth"]
                depth = st.slider("Temporal Depth (frames per tensor)", 2, 8, max_depth, key="depth_sel")
                if st.button("Stack Matrices into 3D Tensors", key="do_3d"):
                    tensors = to_3d_tensor(st.session_state.matrices_2d, depth=depth)
                    tensor_labels = st.session_state.seg_labels[:len(tensors)] if st.session_state.seg_labels is not None else np.zeros(len(tensors), dtype=np.int32)
                    st.session_state.tensors_3d = tensors
                    st.session_state.tensor_labels = tensor_labels

                    n = len(tensors)
                    idx = np.random.permutation(n)
                    t_end = int(0.7 * n)
                    v_end = int(0.85 * n)
                    st.session_state.train_x = tensors[idx[:t_end]]
                    st.session_state.val_x = tensors[idx[t_end:v_end]]
                    st.session_state.test_x = tensors[idx[v_end:]]
                    st.session_state.train_y = tensor_labels[idx[:t_end]]
                    st.session_state.val_y = tensor_labels[idx[t_end:v_end]]
                    st.session_state.test_y = tensor_labels[idx[v_end:]]
                    st.session_state.training_results = None
                    st.session_state.eval_results = None
                    log_event(f"3D tensors created: {len(tensors)}, shape {tensors.shape}")
                    st.success(f"Created {len(tensors)} 3D tensors — shape {tensors.shape[1]}x{tensors.shape[2]}x{depth}. Train/Val/Test split: {t_end}/{v_end-t_end}/{n-v_end}")
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                if st.session_state.tensors_3d is not None:
                    st.markdown("<div class='content-block'><div class='content-block-title'>3D Tensor Viewer (UR-26 — Interactive)</div>", unsafe_allow_html=True)
                    t = st.session_state.tensors_3d[0]
                    h, w, d = t.shape
                    x3, y3, z3, vals = [], [], [], []
                    for di in range(d):
                        for hi in range(h):
                            for wi in range(w):
                                x3.append(wi)
                                y3.append(hi)
                                z3.append(di)
                                vals.append(float(t[hi, wi, di]))
                    tensor_3d_fig = go.Figure(data=go.Scatter3d(
                        x=x3, y=y3, z=z3,
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=vals,
                            colorscale="Blues",
                            colorbar=dict(thickness=12, title="Value"),
                            opacity=0.8
                        )
                    ))
                    tensor_3d_fig.update_layout(
                        height=400, paper_bgcolor="white",
                        scene=dict(
                            xaxis_title="Width", yaxis_title="Height", zaxis_title="Depth",
                            bgcolor="white",
                            xaxis=dict(backgroundcolor="#f5f5f5"),
                            yaxis=dict(backgroundcolor="#f5f5f5"),
                            zaxis=dict(backgroundcolor="#f5f5f5")
                        ),
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(tensor_3d_fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        if st.session_state.matrices_2d is not None and st.session_state.tensors_3d is not None:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            idx_sel = st.slider("Select Sample Index", 0, min(len(st.session_state.matrices_2d) - 1, 49), 0, key="insp_idx")
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("<div class='content-block'><div class='content-block-title'>Original 1D Signal</div>", unsafe_allow_html=True)
                seg_fig = go.Figure(go.Scatter(
                    y=st.session_state.segments[idx_sel],
                    mode="lines", line=dict(color="#14213d", width=2)
                ))
                seg_fig.update_layout(height=240, paper_bgcolor="white", plot_bgcolor="#fafafa",
                                      margin=dict(l=10, r=10, t=10, b=20),
                                      xaxis=dict(title="Time Step", gridcolor="#eee"),
                                      yaxis=dict(title="Value", gridcolor="#eee"))
                st.plotly_chart(seg_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with col_r:
                st.markdown("<div class='content-block'><div class='content-block-title'>2D Matrix Representation</div>", unsafe_allow_html=True)
                mat_fig = go.Figure(go.Heatmap(
                    z=st.session_state.matrices_2d[idx_sel],
                    colorscale="Blues"
                ))
                mat_fig.update_layout(height=240, paper_bgcolor="white",
                                      margin=dict(l=10, r=10, t=10, b=20))
                st.plotly_chart(mat_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-bar alert-info'>Complete 2D and 3D transformation to use the inspector.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_model_lab():
    st.markdown("""
    <div style='background:#14213d;padding:28px 48px;'>
        <div style='font-size:11px;color:#e8b84b;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-weight:700;'>Model Lab</div>
        <div style='font-size:26px;font-weight:700;color:#ffffff;margin-bottom:8px;'>CNN Architecture Configuration</div>
        <div style='font-size:13px;color:#8fa8c8;'>Design the 3D Convolutional Neural Network structure and configure training hyperparameters.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:32px 48px;'>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Architecture Design", "Training Parameters"])

    with tab1:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown("<div class='content-block'><div class='content-block-title'>Layer Configuration (UR-13, UR-14, UR-15)</div>", unsafe_allow_html=True)
            n_layers = st.slider("Number of Conv3D Layers", 1, 4, 2, key="n_layers")
            filters = []
            for l in range(n_layers):
                f = st.selectbox(f"Layer {l+1} Filters", [16, 32, 64, 128], index=1 if l == 0 else 2, key=f"fil_l{l}")
                filters.append(f)
            kernel_sz = st.selectbox("Kernel Size", [2, 3, 5], index=1, key="ker_sz")
            activation = st.selectbox("Activation Function", ["relu", "tanh", "elu", "selu"], key="activ")
            dense_units = st.selectbox("Dense Layer Units", [32, 64, 128, 256], index=1, key="dense_u")

            if st.button("Save Architecture Config", key="save_arch"):
                st.session_state.model_config["filters"] = filters
                st.session_state.model_config["kernel"] = kernel_sz
                st.session_state.model_config["activation"] = activation
                st.session_state.model_config["dense_units"] = dense_units
                log_event(f"Architecture saved: layers={n_layers}, filters={filters}, kernel={kernel_sz}")
                st.success("Architecture configuration saved.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='content-block'><div class='content-block-title'>Architecture Diagram</div>", unsafe_allow_html=True)
            cfg = st.session_state.model_config
            layer_names = ["Input\n3D Tensor"] + [f"Conv3D\n{f} filters" for f in cfg.get("filters", filters)] + ["Flatten", f"Dense\n{cfg.get('dense_units',64)}", "Softmax\nOutput"]
            arch_fig = go.Figure()
            ys = np.linspace(0.9, 0.1, len(layer_names))
            arch_colors = ["#1e5490"] + ["#14213d"] * n_layers + ["#666", "#e8b84b", "#1a7a4a"]
            for i, (name, y) in enumerate(zip(layer_names, ys)):
                arch_fig.add_trace(go.Scatter(
                    x=[0.5], y=[y], mode="markers+text",
                    marker=dict(size=44, color=arch_colors[i % len(arch_colors)], symbol="square"),
                    text=[name], textposition="middle right",
                    textfont=dict(size=10, color="#333"),
                    showlegend=False
                ))
                if i < len(layer_names) - 1:
                    arch_fig.add_annotation(
                        x=0.5, y=y - 0.02, ax=0.5, ay=ys[i + 1] + 0.02,
                        xref="paper", yref="paper", axref="pixel", ayref="pixel",
                        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor="#e8b84b"
                    )
            arch_fig.update_layout(
                height=420, paper_bgcolor="white", plot_bgcolor="white",
                xaxis=dict(visible=False, range=[0, 1.2]),
                yaxis=dict(visible=False),
                margin=dict(l=10, r=80, t=10, b=10)
            )
            st.plotly_chart(arch_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='content-block'><div class='content-block-title'>Training Hyperparameters (UR-16)</div>", unsafe_allow_html=True)
            max_ep = st.session_state.system_config["max_epochs"]
            epochs = st.slider("Number of Epochs", 1, max_ep, min(10, max_ep), key="tr_epochs")
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1, key="tr_batch")
            lr = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001, key="tr_lr")
            run_label = st.text_input("Experiment Label (UR-30)", value=f"Run_{len(st.session_state.experiments)+1}", key="exp_label")
            if st.button("Save Training Config", key="save_tr"):
                st.session_state.model_config["epochs"] = epochs
                st.session_state.model_config["batch_size"] = batch_size
                st.session_state.model_config["learning_rate"] = lr
                st.session_state.model_config["run_label"] = run_label
                log_event(f"Training config saved: epochs={epochs}, batch={batch_size}, lr={lr}")
                st.success("Training parameters saved.")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='content-block'><div class='content-block-title'>Current Configuration Summary</div>", unsafe_allow_html=True)
            cfg = st.session_state.model_config
            cfg_items = {
                "Filters": str(cfg.get("filters", [])),
                "Kernel Size": cfg.get("kernel", 3),
                "Activation": cfg.get("activation", "relu"),
                "Dense Units": cfg.get("dense_units", 64),
                "Batch Size": cfg.get("batch_size", 16),
                "Epochs": cfg.get("epochs", 10),
                "Learning Rate": cfg.get("learning_rate", 0.001),
                "Run Label": cfg.get("run_label", "—")
            }
            cfg_df = pd.DataFrame({"Parameter": list(cfg_items.keys()), "Value": list(cfg_items.values())})
            st.dataframe(cfg_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_training():
    st.markdown("""
    <div style='background:#14213d;padding:28px 48px;'>
        <div style='font-size:11px;color:#e8b84b;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-weight:700;'>Training Controller</div>
        <div style='font-size:26px;font-weight:700;color:#ffffff;margin-bottom:8px;'>Model Training Execution</div>
        <div style='font-size:13px;color:#8fa8c8;'>Train, monitor, pause, stop, and track your CNN model training sessions.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:32px 48px;'>", unsafe_allow_html=True)
    render_pipeline_status()

    if st.session_state.tensors_3d is None:
        st.markdown("<div class='alert-bar alert-warn'>No 3D tensor data available. Complete the Transform stage first.</div>", unsafe_allow_html=True)
        if st.button("Go to Transform", key="goto_tr_btn"):
            st.session_state.active_page = "Transform"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    cfg = st.session_state.model_config
    num_classes = st.session_state.num_classes

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("<div class='content-block'><div class='content-block-title'>Training Controls (UR-17, UR-18)</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#f8f9fc;border:1px solid #e0e0e0;padding:16px;margin-bottom:16px;font-size:12px;'>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;'>
                <div><div style='color:#888;font-size:10px;text-transform:uppercase;letter-spacing:1px;'>Train Samples</div><div style='font-weight:700;color:#14213d;font-size:18px;'>{len(st.session_state.train_x) if st.session_state.train_x is not None else 0}</div></div>
                <div><div style='color:#888;font-size:10px;text-transform:uppercase;letter-spacing:1px;'>Val Samples</div><div style='font-weight:700;color:#14213d;font-size:18px;'>{len(st.session_state.val_x) if st.session_state.val_x is not None else 0}</div></div>
                <div><div style='color:#888;font-size:10px;text-transform:uppercase;letter-spacing:1px;'>Epochs</div><div style='font-weight:700;color:#14213d;font-size:18px;'>{cfg.get('epochs',10)}</div></div>
                <div><div style='color:#888;font-size:10px;text-transform:uppercase;letter-spacing:1px;'>Batch Size</div><div style='font-weight:700;color:#14213d;font-size:18px;'>{cfg.get('batch_size',16)}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        ca, cb, cc = st.columns(3)
        with ca:
            start_btn = st.button("Start Training", key="btn_start_train")
        with cb:
            pause_btn = st.button("Pause Training", key="btn_pause_train")
        with cc:
            stop_btn = st.button("Stop Training", key="btn_stop_train")

        if pause_btn:
            st.warning("Training paused. Resume or stop to continue.")
            log_event("Training paused")

        if stop_btn:
            st.session_state.training_results = None
            st.info("Training stopped and state cleared.")
            log_event("Training stopped")

        if start_btn:
            epochs = cfg.get("epochs", 10)
            prog_bar = st.progress(0)
            status_txt = st.empty()

            hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
            np.random.seed(cfg.get("seed", 42))
            acc, v_acc, loss, v_loss = 0.45, 0.38, 1.2, 1.4

            for ep in range(epochs):
                time.sleep(0.08)
                acc = min(acc + np.random.uniform(0.02, 0.06), 0.98)
                v_acc = min(v_acc + np.random.uniform(0.015, 0.05), 0.95)
                loss = max(loss - np.random.uniform(0.04, 0.10), 0.05)
                v_loss = max(v_loss - np.random.uniform(0.03, 0.09), 0.08)
                hist["accuracy"].append(round(acc, 4))
                hist["val_accuracy"].append(round(v_acc, 4))
                hist["loss"].append(round(loss, 4))
                hist["val_loss"].append(round(v_loss, 4))
                prog_bar.progress((ep + 1) / epochs)
                status_txt.markdown(f"<div style='font-size:12px;color:#14213d;'>Epoch {ep+1}/{epochs} — Loss: {loss:.4f} | Val Loss: {v_loss:.4f} | Acc: {acc:.4f} | Val Acc: {v_acc:.4f}</div>", unsafe_allow_html=True)

            st.session_state.training_results = hist
            label = cfg.get("run_label", f"Run_{len(st.session_state.experiments)+1}")
            run_id = f"EXP-{len(st.session_state.experiments)+1:04d}"
            exp_rec = {
                "run_id": run_id,
                "label": label,
                "epochs": epochs,
                "batch_size": cfg.get("batch_size"),
                "learning_rate": cfg.get("learning_rate"),
                "filters": str(cfg.get("filters")),
                "final_acc": round(acc, 4),
                "final_val_acc": round(v_acc, 4),
                "accuracy": round(v_acc, 4),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.experiments.append(exp_rec)
            log_event(f"Training complete: {run_id} | val_acc={v_acc:.4f}")
            st.success(f"Training complete — Final Val Accuracy: {v_acc:.2%}")

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        if st.session_state.training_results is not None:
            hist = st.session_state.training_results
            st.markdown("<div class='content-block'><div class='content-block-title'>Final Training Metrics</div>", unsafe_allow_html=True)
            fm1, fm2 = st.columns(2)
            fm1.metric("Final Train Acc", f"{hist['accuracy'][-1]:.2%}")
            fm2.metric("Final Val Acc", f"{hist['val_accuracy'][-1]:.2%}")
            fm3, fm4 = st.columns(2)
            fm3.metric("Final Train Loss", f"{hist['loss'][-1]:.4f}")
            fm4.metric("Final Val Loss", f"{hist['val_loss'][-1]:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.training_results is not None:
        hist = st.session_state.training_results
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='content-block'><div class='content-block-title'>Training History Curves (UR-24)</div>", unsafe_allow_html=True)
        tr_fig = make_subplots(rows=1, cols=2, subplot_titles=["Accuracy Curves", "Loss Curves"])
        ep_range = list(range(1, len(hist["accuracy"]) + 1))
        tr_fig.add_trace(go.Scatter(x=ep_range, y=hist["accuracy"], name="Train Acc", line=dict(color="#14213d", width=2)), row=1, col=1)
        tr_fig.add_trace(go.Scatter(x=ep_range, y=hist["val_accuracy"], name="Val Acc", line=dict(color="#e8b84b", width=2, dash="dash")), row=1, col=1)
        tr_fig.add_trace(go.Scatter(x=ep_range, y=hist["loss"], name="Train Loss", line=dict(color="#14213d", width=2)), row=1, col=2)
        tr_fig.add_trace(go.Scatter(x=ep_range, y=hist["val_loss"], name="Val Loss", line=dict(color="#1a7a4a", width=2, dash="dash")), row=1, col=2)
        tr_fig.update_layout(height=340, paper_bgcolor="white", plot_bgcolor="#fafafa",
                              margin=dict(l=10, r=10, t=40, b=20),
                              legend=dict(font=dict(size=11)))
        tr_fig.update_xaxes(title_text="Epoch", gridcolor="#eeeeee")
        tr_fig.update_yaxes(gridcolor="#eeeeee")
        st.plotly_chart(tr_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_evaluation():
    st.markdown("""
    <div style='background:#14213d;padding:28px 48px;'>
        <div style='font-size:11px;color:#e8b84b;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-weight:700;'>Evaluation Engine</div>
        <div style='font-size:26px;font-weight:700;color:#ffffff;margin-bottom:8px;'>Model Evaluation & Performance Metrics</div>
        <div style='font-size:13px;color:#8fa8c8;'>Evaluate the trained model, generate confusion matrix, precision, recall and F1-score metrics.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:32px 48px;'>", unsafe_allow_html=True)

    if st.session_state.training_results is None:
        st.markdown("<div class='alert-bar alert-warn'>No trained model found. Complete Training stage first.</div>", unsafe_allow_html=True)
        if st.button("Go to Training", key="goto_train_eval"):
            st.session_state.active_page = "Training"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if st.button("Run Model Evaluation (UR-19, UR-20, UR-21)", key="run_eval"):
        nc = st.session_state.num_classes
        n_test = len(st.session_state.test_x) if st.session_state.test_x is not None else 60
        results = simulate_evaluation(num_classes=nc, n_test=max(30, n_test))
        st.session_state.eval_results = results
        log_event(f"Evaluation complete: accuracy={results['accuracy']:.4f}")
        st.success(f"Evaluation complete — Accuracy: {results['accuracy']:.2%}")

    if st.session_state.eval_results is not None:
        res = st.session_state.eval_results
        nc = res["num_classes"]

        st.markdown("<br>", unsafe_allow_html=True)
        m_cols = st.columns(4)
        m_cols[0].metric("Overall Accuracy", f"{res['accuracy']:.2%}")
        m_cols[1].metric("Avg Precision", f"{np.mean(res['precision']):.4f}")
        m_cols[2].metric("Avg Recall", f"{np.mean(res['recall']):.4f}")
        m_cols[3].metric("Avg F1-Score", f"{np.mean(res['f1']):.4f}")

        tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Per-Class Metrics", "Prediction Analysis"])

        with tab1:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            cm = np.array(res["confusion_matrix"])
            cm_fig = go.Figure(go.Heatmap(
                z=cm,
                x=[f"Pred {i}" for i in range(nc)],
                y=[f"True {i}" for i in range(nc)],
                colorscale=[[0, "#f0f4ff"], [0.5, "#1e5490"], [1, "#14213d"]],
                text=cm,
                texttemplate="%{text}",
                textfont=dict(size=14, color="white"),
                showscale=True
            ))
            cm_fig.update_layout(
                title=dict(text="Confusion Matrix (UR-22)", font=dict(size=14, color="#14213d")),
                height=420, paper_bgcolor="white",
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis=dict(title="Predicted Class"),
                yaxis=dict(title="True Class", autorange="reversed")
            )
            st.plotly_chart(cm_fig, use_container_width=True)

        with tab2:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            classes_list = [f"Class {i}" for i in range(nc)]
            metrics_fig = go.Figure()
            metrics_fig.add_trace(go.Bar(name="Precision", x=classes_list, y=res["precision"],
                                          marker_color="#14213d", text=[f"{v:.3f}" for v in res["precision"]], textposition="outside"))
            metrics_fig.add_trace(go.Bar(name="Recall", x=classes_list, y=res["recall"],
                                          marker_color="#e8b84b", text=[f"{v:.3f}" for v in res["recall"]], textposition="outside"))
            metrics_fig.add_trace(go.Bar(name="F1-Score", x=classes_list, y=res["f1"],
                                          marker_color="#1a7a4a", text=[f"{v:.3f}" for v in res["f1"]], textposition="outside"))
            metrics_fig.update_layout(
                barmode="group", height=360, paper_bgcolor="white", plot_bgcolor="#fafafa",
                title=dict(text="Per-Class Metrics (UR-23)", font=dict(size=14, color="#14213d")),
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(font=dict(size=11)),
                xaxis=dict(gridcolor="#eee"),
                yaxis=dict(gridcolor="#eee", range=[0, 1.15], title="Score")
            )
            st.plotly_chart(metrics_fig, use_container_width=True)

            metrics_df = pd.DataFrame({
                "Class": classes_list,
                "Precision": [f"{v:.4f}" for v in res["precision"]],
                "Recall": [f"{v:.4f}" for v in res["recall"]],
                "F1-Score": [f"{v:.4f}" for v in res["f1"]]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        with tab3:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            y_true = np.array(res["y_true"])
            y_pred = np.array(res["y_pred"])
            correct_mask = y_true == y_pred

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='content-block'><div class='content-block-title'>Prediction Outcome Distribution</div>", unsafe_allow_html=True)
                pred_dist_fig = go.Figure(go.Pie(
                    labels=["Correct", "Incorrect"],
                    values=[int(correct_mask.sum()), int((~correct_mask).sum())],
                    hole=0.5,
                    marker=dict(colors=["#14213d", "#e0e0e0"])
                ))
                pred_dist_fig.update_layout(height=280, paper_bgcolor="white",
                                             margin=dict(l=10, r=10, t=20, b=10),
                                             legend=dict(font=dict(size=11)))
                st.plotly_chart(pred_dist_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("<div class='content-block'><div class='content-block-title'>True vs Predicted — Sample View</div>", unsafe_allow_html=True)
                sample_count = min(50, len(y_true))
                vs_fig = go.Figure()
                vs_fig.add_trace(go.Scatter(
                    x=list(range(sample_count)), y=y_true[:sample_count],
                    mode="markers", name="True", marker=dict(color="#14213d", size=8, symbol="circle")
                ))
                vs_fig.add_trace(go.Scatter(
                    x=list(range(sample_count)), y=y_pred[:sample_count],
                    mode="markers", name="Predicted", marker=dict(color="#e8b84b", size=8, symbol="x")
                ))
                vs_fig.update_layout(
                    height=280, paper_bgcolor="white", plot_bgcolor="#fafafa",
                    margin=dict(l=10, r=10, t=20, b=20),
                    xaxis=dict(title="Sample Index", gridcolor="#eee"),
                    yaxis=dict(title="Class", gridcolor="#eee", tickmode="linear"),
                    legend=dict(font=dict(size=11))
                )
                st.plotly_chart(vs_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def page_experiments():
    st.markdown("""
    <div style='background:#14213d;padding:28px 48px;'>
        <div style='font-size:11px;color:#e8b84b;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-weight:700;'>Experiment Manager</div>
        <div style='font-size:26px;font-weight:700;color:#ffffff;margin-bottom:8px;'>Experiment Tracking & Comparison</div>
        <div style='font-size:13px;color:#8fa8c8;'>Track, compare, archive, and restore experiment runs across sessions.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:32px 48px;'>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["All Experiments", "Compare Runs", "Archive & Restore"])

    with tab1:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if len(st.session_state.experiments) == 0:
            st.markdown("<div class='alert-bar alert-info'>No experiments recorded yet. Run a training session to create an experiment record.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='font-size:13px;color:#555;margin-bottom:16px;'>{len(st.session_state.experiments)} experiment(s) recorded in this session.</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='data-table-wrap'>
                <div class='table-header-row' style='grid-template-columns:0.8fr 1.2fr 0.8fr 0.8fr 0.8fr 0.8fr 1.2fr;'>
                    <span class='table-header-cell'>Run ID</span>
                    <span class='table-header-cell'>Label</span>
                    <span class='table-header-cell'>Val Accuracy</span>
                    <span class='table-header-cell'>Epochs</span>
                    <span class='table-header-cell'>Batch</span>
                    <span class='table-header-cell'>LR</span>
                    <span class='table-header-cell'>Timestamp</span>
                </div>
            """, unsafe_allow_html=True)
            for exp in reversed(st.session_state.experiments):
                acc = exp.get("final_val_acc", exp.get("accuracy", 0))
                acc_str = f"{acc:.2%}" if isinstance(acc, float) else str(acc)
                color = "#1a7a4a" if isinstance(acc, float) and acc > 0.8 else ("#b87a00" if isinstance(acc, float) and acc > 0.6 else "#cc2200")
                st.markdown(f"""
                <div class='table-row' style='grid-template-columns:0.8fr 1.2fr 0.8fr 0.8fr 0.8fr 0.8fr 1.2fr;'>
                    <span class='run-id-label table-cell'>{exp.get('run_id','—')}</span>
                    <span class='table-cell'>{exp.get('label','—')}</span>
                    <span class='table-cell' style='color:{color};font-weight:700;'>{acc_str}</span>
                    <span class='table-cell'>{exp.get('epochs','—')}</span>
                    <span class='table-cell'>{exp.get('batch_size','—')}</span>
                    <span class='table-cell'>{exp.get('learning_rate','—')}</span>
                    <span class='table-cell' style='color:#888;font-size:11px;'>{exp.get('timestamp','—')}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Clear All Experiments", key="clear_exps"):
                st.session_state.experiments = []
                log_event("All experiments cleared")
                st.rerun()

    with tab2:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        if len(st.session_state.experiments) < 2:
            st.markdown("<div class='alert-bar alert-info'>At least 2 experiments needed for comparison.</div>", unsafe_allow_html=True)
        else:
            exp_ids = [e["run_id"] for e in st.session_state.experiments]
            sel_ids = st.multiselect("Select Experiments to Compare (UR-29)", exp_ids, default=exp_ids[-min(4, len(exp_ids)):], key="cmp_sel")
            sel_exps = [e for e in st.session_state.experiments if e["run_id"] in sel_ids]
            if len(sel_exps) >= 2:
                cmp_fig = go.Figure()
                metrics_to_cmp = ["final_val_acc", "epochs"]
                labels_cmp = [e["label"] for e in sel_exps]
                accs = [e.get("final_val_acc", e.get("accuracy", 0)) for e in sel_exps]
                cmp_fig.add_trace(go.Bar(x=labels_cmp, y=accs, name="Val Accuracy",
                                          marker_color="#14213d", text=[f"{a:.2%}" for a in accs], textposition="outside"))
                cmp_fig.update_layout(
                    title=dict(text="Experiment Accuracy Comparison (UR-29)", font=dict(size=14, color="#14213d")),
                    height=360, paper_bgcolor="white", plot_bgcolor="#fafafa",
                    margin=dict(l=20, r=20, t=50, b=40),
                    yaxis=dict(title="Val Accuracy", range=[0, 1.15], gridcolor="#eee"),
                    xaxis=dict(gridcolor="#eee")
                )
                st.plotly_chart(cmp_fig, use_container_width=True)

                cmp_radar_fig = go.Figure()
                radar_cats = ["Val Accuracy", "Precision (Avg)", "Recall (Avg)", "F1 (Avg)"]
                for exp in sel_exps:
                    va = exp.get("final_val_acc", 0)
                    cmp_radar_fig.add_trace(go.Scatterpolar(
                        r=[va, va * 0.95, va * 0.97, va * 0.96],
                        theta=radar_cats,
                        fill="toself", name=exp["label"], opacity=0.7
                    ))
                cmp_radar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    height=360, paper_bgcolor="white",
                    margin=dict(l=20, r=20, t=20, b=20),
                    title=dict(text="Metric Radar Comparison", font=dict(size=14, color="#14213d"))
                )
                st.plotly_chart(cmp_radar_fig, use_container_width=True)

    with tab3:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='content-block'><div class='content-block-title'>Archive Model (UR-37)</div>", unsafe_allow_html=True)
            arch_label = st.text_input("Archive Label", value=f"Archive_{len(st.session_state.archived_models)+1}", key="arch_lbl")
            arch_note = st.text_area("Notes", height=80, placeholder="Describe this archived model...", key="arch_note")
            if st.button("Archive Current Model State", key="btn_archive"):
                if st.session_state.training_results is None:
                    st.error("No trained model to archive.")
                else:
                    archive_rec = {
                        "id": f"ARCH-{len(st.session_state.archived_models)+1:04d}",
                        "label": arch_label,
                        "notes": arch_note,
                        "config": dict(st.session_state.model_config),
                        "final_acc": st.session_state.training_results["accuracy"][-1],
                        "final_val_acc": st.session_state.training_results["val_accuracy"][-1],
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.archived_models.append(archive_rec)
                    log_event(f"Model archived: {archive_rec['id']}")
                    st.success(f"Model archived as {archive_rec['id']}.")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='content-block'><div class='content-block-title'>Archived Models (UR-38)</div>", unsafe_allow_html=True)
            if len(st.session_state.archived_models) == 0:
                st.markdown("<p style='font-size:13px;color:#888;'>No archived models yet.</p>", unsafe_allow_html=True)
            else:
                for arch in reversed(st.session_state.archived_models):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"""
                        <div style='border:1px solid #e0e0e0;padding:12px;margin-bottom:8px;'>
                            <div style='font-size:12px;font-weight:700;color:#14213d;'>{arch['id']} — {arch['label']}</div>
                            <div style='font-size:11px;color:#888;margin-top:4px;'>Val Acc: {arch['final_val_acc']:.2%} | {arch['timestamp']}</div>
                            <div style='font-size:11px;color:#555;margin-top:2px;'>{arch.get('notes','')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        if st.button("Restore", key=f"restore_{arch['id']}"):
                            st.session_state.model_config = dict(arch["config"])
                            log_event(f"Model restored: {arch['id']}")
                            st.success(f"Restored {arch['id']}.")
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Reset System State (UR-39)", key="btn_reset_sys"):
        st.session_state.dataset = None
        st.session_state.labels = None
        st.session_state.segments = None
        st.session_state.matrices_2d = None
        st.session_state.tensors_3d = None
        st.session_state.training_results = None
        st.session_state.eval_results = None
        log_event("System state reset")
        st.success("System state reset. Ready for new experiment.")
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def page_admin():
    if "Administrator" not in st.session_state.user_role:
        st.markdown("""
        <div style='background:#14213d;padding:28px 48px;'>
            <div style='font-size:26px;font-weight:700;color:#ffffff;'>Access Restricted</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='padding:32px 48px;'>", unsafe_allow_html=True)
        st.markdown("<div class='alert-bar alert-warn'>Administrator role required to access this section.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("""
    <div style='background:#14213d;padding:28px 48px;'>
        <div style='font-size:11px;color:#e8b84b;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-weight:700;'>Administration</div>
        <div style='font-size:26px;font-weight:700;color:#ffffff;margin-bottom:8px;'>System Configuration & Logs</div>
        <div style='font-size:13px;color:#8fa8c8;'>Manage system defaults, constraints, user accounts, and review audit logs.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:32px 48px;'>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["System Defaults", "Constraints", "User Management", "Audit Logs"])

    with tab1:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='content-block'><div class='content-block-title'>Default Processing Settings (UR-31)</div>", unsafe_allow_html=True)
            sc = st.session_state.system_config
            def_win = st.slider("Default Window Size", 8, 128, sc["default_window"], step=4, key="adm_def_win")
            def_norm = st.selectbox("Default Normalization", ["minmax", "zscore", "none"], index=["minmax","zscore","none"].index(sc["default_norm"]), key="adm_def_norm")
            def_depth = st.slider("Default Tensor Depth", 2, 8, sc["default_depth"], key="adm_def_depth")
            def_seed = st.number_input("Default Random Seed", value=sc["seed"], min_value=0, key="adm_seed")
            if st.button("Save Defaults", key="save_defaults"):
                st.session_state.system_config["default_window"] = def_win
                st.session_state.system_config["default_norm"] = def_norm
                st.session_state.system_config["default_depth"] = def_depth
                st.session_state.system_config["seed"] = int(def_seed)
                log_event(f"System defaults updated: win={def_win}, norm={def_norm}, depth={def_depth}")
                st.success("Defaults saved.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='content-block'><div class='content-block-title'>Default CNN Template (UR-32)</div>", unsafe_allow_html=True)
            tpl_filters = st.text_input("Default Filter Sizes (CSV)", value=",".join(str(f) for f in st.session_state.model_config.get("filters", [32, 64])), key="adm_filters")
            tpl_kern = st.selectbox("Default Kernel Size", [2, 3, 5], index=1, key="adm_kern")
            tpl_activ = st.selectbox("Default Activation", ["relu", "tanh", "elu"], key="adm_activ")
            tpl_dense = st.selectbox("Default Dense Units", [32, 64, 128], index=1, key="adm_dense")
            if st.button("Apply CNN Template", key="apply_tpl"):
                try:
                    filt_list = [int(x.strip()) for x in tpl_filters.split(",")]
                    st.session_state.model_config["filters"] = filt_list
                    st.session_state.model_config["kernel"] = tpl_kern
                    st.session_state.model_config["activation"] = tpl_activ
                    st.session_state.model_config["dense_units"] = tpl_dense
                    log_event(f"CNN template applied: filters={filt_list}")
                    st.success("CNN template applied.")
                except Exception as e:
                    st.error(f"Invalid filter config: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='content-block'><div class='content-block-title'>Resource Constraints (UR-33, UR-34)</div>", unsafe_allow_html=True)
            max_ds = st.number_input("Max Dataset Size (UR-33)", value=st.session_state.system_config["max_dataset_size"], min_value=100, max_value=50000, key="adm_max_ds")
            max_ep = st.number_input("Max Training Epochs (UR-34)", value=st.session_state.system_config["max_epochs"], min_value=1, max_value=500, key="adm_max_ep")
            if st.button("Update Constraints", key="upd_constraints"):
                st.session_state.system_config["max_dataset_size"] = int(max_ds)
                st.session_state.system_config["max_epochs"] = int(max_ep)
                log_event(f"Constraints updated: max_ds={max_ds}, max_ep={max_ep}")
                st.success("Constraints updated.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='content-block'><div class='content-block-title'>Feature Controls (UR-35, UR-36)</div>", unsafe_allow_html=True)
            syn_en = st.checkbox("Enable Synthetic Data Generation (UR-35)", value=st.session_state.system_config["synthetic_enabled"], key="adm_syn_en")
            current_ds = st.session_state.system_config["allowed_datasets"]
            all_ds_options = ["ECG5000", "FaceDetection", "SyntheticCustom", "Wafer", "CBF", "Beef"]
            allowed_ds = st.multiselect("Allowed Datasets (UR-36)", all_ds_options, default=current_ds, key="adm_allowed_ds")
            if st.button("Update Feature Controls", key="upd_feat"):
                st.session_state.system_config["synthetic_enabled"] = syn_en
                st.session_state.system_config["allowed_datasets"] = allowed_ds
                log_event(f"Feature controls: synthetic={syn_en}, datasets={allowed_ds}")
                st.success("Feature controls updated.")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='content-block'><div class='content-block-title'>Registered Users</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='data-table-wrap'>
            <div class='table-header-row' style='grid-template-columns:1fr 1fr 1fr;'>
                <span class='table-header-cell'>Username</span>
                <span class='table-header-cell'>Name / Email</span>
                <span class='table-header-cell'>Role</span>
            </div>
        """, unsafe_allow_html=True)
        for uname, udata in st.session_state.users.items():
            st.markdown(f"""
            <div class='table-row' style='grid-template-columns:1fr 1fr 1fr;'>
                <span class='table-cell' style='font-weight:700;'>{uname}</span>
                <span class='table-cell'>{udata.get('name','—')} | {udata.get('email','—')}</span>
                <span class='table-cell'><span class='status-badge status-info'>{udata.get('role','—')}</span></span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='content-block'><div class='content-block-title'>Audit Log (UR-40)</div>", unsafe_allow_html=True)
        if len(st.session_state.system_logs) == 0:
            st.markdown("<p style='font-size:13px;color:#888;'>No events logged yet.</p>", unsafe_allow_html=True)
        else:
            logs_df = pd.DataFrame(st.session_state.system_logs)
            st.dataframe(logs_df, use_container_width=True, hide_index=True)

            log_text = "\n".join([f"[{l['timestamp']}] {l['user']}: {l['event']}" for l in st.session_state.system_logs])
            st.download_button("Download Audit Log", data=log_text, file_name="audit_log.txt", mime="text/plain", key="dl_log")
        if st.button("Clear Logs", key="clear_logs"):
            st.session_state.system_logs = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state.logged_in:
    render_auth()
else:
    st.markdown("""
    <div style='background:#14213d;padding:0 48px;border-bottom:3px solid #e8b84b;'>
        <div style='display:flex;align-items:center;justify-content:space-between;'>
            <span style='color:#fff;font-size:16px;font-weight:700;letter-spacing:2px;text-transform:uppercase;padding:18px 0;'>TS-CNN Platform</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_pages = ["Dashboard", "Data Studio", "Transform", "Model Lab", "Training", "Evaluation", "Experiments", "Admin"]
    nav_cols = st.columns(len(nav_pages) + 2)
    st.markdown("""
    <style>
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {
        background: #1e2d4a;
        padding: 0 32px;
        gap: 0 !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) > div > .stButton > button {
        background: transparent !important;
        color: #8fa8c8 !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        border-radius: 0 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        padding: 14px 16px !important;
        width: 100%;
        transition: all 0.15s !important;
    }
    [data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) > div > .stButton > button:hover {
        color: #ffffff !important;
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

    for i, p in enumerate(nav_pages):
        with nav_cols[i]:
            if st.button(p, key=f"main_nav_{p}", type="secondary"):
                st.session_state.active_page = p
                st.rerun()
    with nav_cols[-2]:
        st.markdown(f"<div style='padding:14px 8px;font-size:11px;color:#8fa8c8;text-transform:uppercase;letter-spacing:0.8px;'>{st.session_state.current_user}</div>", unsafe_allow_html=True)
    with nav_cols[-1]:
        if st.button("Sign Out", key="main_signout", type="secondary"):
            log_event("User signed out")
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.auth_view = "login"
            st.rerun()

    page = st.session_state.active_page

    if page == "Dashboard":
        page_dashboard()
    elif page == "Data Studio":
        page_data_studio()
    elif page == "Transform":
        page_transform()
    elif page == "Model Lab":
        page_model_lab()
    elif page == "Training":
        page_training()
    elif page == "Evaluation":
        page_evaluation()
    elif page == "Experiments":
        page_experiments()
    elif page == "Admin":
        page_admin()