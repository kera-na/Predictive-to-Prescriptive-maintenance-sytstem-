"""Industrial Prescriptive Maintenance Command Center — C-MAPSS FD004.

Merged from two prior dashboards, combining:
  - Dashboard_6's CMMS-style theming, gauges/panels, artifact-source picker
    (local folder OR zip upload), demo-mode fallback, cost-avoidance
    estimator, and model card.
  - Dashboard_FD004_Adapted's full-fleet inference straight from the raw
    C-MAPSS test files (not a size-capped precomputed snapshot), plus a
    persisted AND displayed audit trail with CSV download.

Two real bugs from the prior versions are fixed here:
  1. A failed Gemini call used to get logged into the audit trail as if it
     were a real advisory. Prescriptions now carry an explicit ok/fail flag
     and a failed call is never treated as loggable advisory content.
  2. Switching the selected engine could show a *different* engine's stale
     AI prescription, because it lived in one shared session-state slot.
     Prescriptions are now cached per engine ID.

Run:
    pip install streamlit tensorflow tf_keras scikit-learn joblib pandas numpy plotly google-genai
    streamlit run Dashboard_Pro.py

Sidebar: point "Local artifacts folder" at the notebook's
revised_4class_artifacts/selected_deployment_models folder (or upload a zip
of it), and "FD004 data folder" at wherever test_FD004.txt / RUL_FD004.txt
live. Either input can be left as-is; if the referenced files aren't found,
the dashboard falls back to demo mode with simulated data instead of crashing.
"""

import os
# Must be set before TensorFlow is imported anywhere (including transitively).
# Exported .keras files from this project have been confirmed (by inspecting
# their own magic bytes) to actually be legacy HDF5 content saved under a
# '.keras' name by an older Colab TF/Keras runtime, using layer kwargs (e.g.
# LSTM's 'time_major') that Keras 3 removed entirely. This flag routes
# tf.keras through the tf_keras compatibility package (legacy Keras 2 engine)
# so these files load correctly. Requires: pip install tf_keras (matching
# your installed tensorflow version).
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import google.genai as genai

import tensorflow as tf

# Standard NASA C-MAPSS sensor descriptions (s1-s21), used only for display.
# The model itself keeps using the raw s1..s21 column names internally --
# this dict never touches feature ordering or model input, only labels shown
# to the user in charts, dropdowns, and the XAI panel.
SENSOR_NAME_MAP = {
    "s1": " S1: T2 (Fan inlet temperature)",
    "s2": " S2: T24 (LPC outlet temperature)",
    "s3": " S3: T30 (HPC outlet temperature)",
    "s4": " S4: T50 (LPT outlet temperature)",
    "s5": " S5: P2 (Fan inlet pressure)",
    "s6": " S6: P15 (Bypass-duct pressure)",
    "s7": " S7: P30 (HPC outlet pressure)",
    "s8": " S8: Nf (Physical fan speed)",
    "s9": " S9: Nc (Physical core speed)",
    "s10": " S10: epr (Engine pressure ratio)",
    "s11": " S11: Ps30 (HPC outlet static pressure)",
    "s12": " S12: phi (Fuel flow / Ps30 ratio)",
    "s13": " S13: NRf (Corrected fan speed)",
    "s14": " S14: NRc (Corrected core speed)",
    "s15": " S15: BPR (Bypass ratio)",
    "s16": " S16: farB (Burner fuel-air ratio)",
    "s17": " S17: htBleed (Bleed enthalpy)",
    "s18": " S18: Nf_dmd (Demanded fan speed)",
    "s19": " S19: PCNfR_dmd (Demanded corrected fan speed)",
    "s20": " S20: W31 (HPT coolant bleed)",
    "s21": " S21: W32 (LPT coolant bleed)",
}

def pretty_sensor(code):
    """Real-name label for a sensor code, falling back to the raw code itself
    for anything not in the map (e.g. if features ever includes something
    other than s1..s21)."""
    return SENSOR_NAME_MAP.get(code, code)

# ==============================================================================
# PAGE CONFIGURATION & CMMS-STYLE THEMING
# ==============================================================================
st.set_page_config(
    page_title="Predictive-to-Prescriptive Maintenance Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg": "#0b0e14", "panel": "#111827", "panel_border": "#1f2937",
    "text": "#e5e7eb", "muted": "#9ca3af",
    "green": "#10b981", "amber": "#f59e0b", "red": "#ef4444", "crimson": "#da6019",
    "blue": "#3b82f6", "purple": "#a855f7",
}

# 4-CLASS CONDITION SCHEME (sensor-derived proxy labels, not ground truth).
# Healthy -> no fault flagged. Fan / HPC -> single-component fault flagged.
# HPC+Fan -> both components flagged simultaneously, the most urgent state.
CLASS_NAMES = ["Healthy", "Fan", "HPC", "HPC+Fan"]
STATE_LABELS = {0: "Healthy", 1: "Fan", 2: "HPC", 3: "HPC+Fan"}
STATE_COLOR = {
    "Healthy": PALETTE["green"], "Fan": PALETTE["amber"],
    "HPC": PALETTE["red"], "HPC+Fan": PALETTE["crimson"],
}
WORST_CLASS_IDX = 3  # HPC+Fan drives the "urgent" gauges and the LLM prescription trigger
RAW_COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]

st.markdown(f"""
    <style>
    .stApp {{ background-color: {PALETTE['bg']}; color: {PALETTE['text']}; }}
    .block-container {{ padding-top: 1.2rem; }}
    [data-testid='stMetric'] {{ background: #141b2d; padding: 14px; border-radius: 10px; }}
    div[data-testid="stMetricValue"] {{ font-size: 26px; }}
    .section-title {{
        font-weight: 700; font-size: 15px; color: {PALETTE['text']};
        text-transform: uppercase; letter-spacing: 0.5px; margin: 18px 0 8px 0;
    }}
    .big-stat {{ font-size: 46px; font-weight: 800; color: {PALETTE['crimson']}; text-align: center; }}
    .big-stat-label {{ font-size: 12px; color: {PALETTE['muted']}; text-align: center; text-transform: uppercase; }}
    .mode-badge-live {{
        color: {PALETTE['green']}; background: rgba(16,185,129,0.12); padding: 4px 10px;
        border-radius: 6px; font-size: 12px; font-weight: 700; border: 1px solid rgba(16,185,129,0.3);
    }}
    .mode-badge-demo {{
        color: {PALETTE['purple']}; background: rgba(168,85,247,0.12); padding: 4px 10px;
        border-radius: 6px; font-size: 12px; font-weight: 700; border: 1px solid rgba(168,85,247,0.3);
    }}
    .note {{ color: {PALETTE['muted']}; font-size: .9rem; }}
    </style>
""", unsafe_allow_html=True)


def panel(title, right_html=""):
    """Plain section header (no boxed panel chrome) -- matches the flatter,
    unboxed look of the FD004_Adapted dashboard rather than a bordered card."""
    st.markdown(f"<div class='section-title'>{title}"
                f"<span style='float:right; font-weight:400; text-transform:none; "
                f"color:{PALETTE['muted']};'>{right_html}</span></div>", unsafe_allow_html=True)


# ==============================================================================
# ARTIFACT SOURCE — local folder or zip upload (Colab export -> local machine)
# ==============================================================================
LOCAL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_artifacts_cache")
_default_artifact_dir = os.environ.get(
    "PDM_ARTIFACT_DIR",
    "selected_deployment_models",   # relative to the repo root, like the other app's "deployment_model/cnn"
)
_default_data_dir = os.environ.get("PDM_DATA_DIR", ".")


def _find_dir_containing(filename, root):
    """Case-insensitive search under `root` for the folder holding `filename`.
    Linux servers (Streamlit Cloud) are case-sensitive and the repo layout may
    differ slightly from a Windows machine, so don't rely on an exact path."""
    target = filename.lower()
    for dirpath, _dirs, files in os.walk(root):
        if any(f.lower() == target for f in files):
            return dirpath
    return None


_app_root = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(_default_artifact_dir, "preprocessing_and_label_rules.joblib")):
    _found = _find_dir_containing("preprocessing_and_label_rules.joblib", _app_root)
    if _found:
        _default_artifact_dir = _found
if not os.path.exists(os.path.join(_default_data_dir, "test_FD004.txt")):
    _found = _find_dir_containing("test_FD004.txt", _app_root)
    if _found:
        _default_data_dir = _found

st.sidebar.caption("Build marker: 2026-08-20-pro (merged CMMS UI + full-fleet raw-data inference)")
st.sidebar.markdown("### 📦 Load Model Artifacts")
artifact_source = st.sidebar.radio(
    "Artifacts source", ["Local folder path", "Upload a zip"],
    horizontal=True, label_visibility="collapsed",
)

if artifact_source == "Upload a zip":
    uploaded_zip = st.sidebar.file_uploader(
        "Zip of the notebook's `selected_deployment_models` folder",
        type="zip",
        help='In a Colab cell, run: !zip -r deploy.zip "$DEPLOY_DIR"  then download deploy.zip from the '
             "Colab file browser and upload it here.",
    )
    if uploaded_zip is not None:
        zip_bytes = uploaded_zip.getvalue()
        zip_hash = hashlib.md5(zip_bytes).hexdigest()[:10]
        extract_dir = os.path.join(LOCAL_CACHE_DIR, zip_hash)
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(extract_dir)
        # The zip may contain the folder's contents directly at its root, or wrapped
        # in one subfolder (e.g. if the whole ARTIFACT_DIR was zipped) -- detect either.
        candidates = [extract_dir] + [
            os.path.join(extract_dir, d) for d in os.listdir(extract_dir)
            if os.path.isdir(os.path.join(extract_dir, d))
        ]
        resolved_dir = next(
            (c for c in candidates if os.path.exists(os.path.join(c, "preprocessing_and_label_rules.joblib"))), None
        )
        if resolved_dir:
            DEPLOY_DIR = resolved_dir
            st.sidebar.success(f"Loaded artifacts from {uploaded_zip.name}.")
        else:
            st.sidebar.error("No preprocessing_and_label_rules.joblib found anywhere in that zip -- "
                              "check you zipped the right folder.")
            DEPLOY_DIR = _default_artifact_dir
    else:
        DEPLOY_DIR = _default_artifact_dir
else:
    DEPLOY_DIR = st.sidebar.text_input(
        "Local artifacts folder", value=_default_artifact_dir,
        help="The selected_deployment_models folder containing condition_classifier.*, "
             "rul_regressor.* and preprocessing_and_label_rules.joblib.",
    )

data_dir = st.sidebar.text_input(
    "FD004 data folder", value=_default_data_dir,
    help="Folder containing test_FD004.txt and RUL_FD004.txt.",
)

AUDIT_LOG_PATH = os.path.join(DEPLOY_DIR, "audit_log.csv")
AUDIT_LOG_COLUMNS = [
    "timestamp_utc", "engine_id", "predicted_state", "predicted_rul", "true_rul",
    "advisory_source", "advisory_text", "verdict", "advisory_rating", "remarks",
]


def load_audit_log():
    if os.path.exists(AUDIT_LOG_PATH):
        try:
            return pd.read_csv(AUDIT_LOG_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=AUDIT_LOG_COLUMNS)


def append_audit_entry(entry: dict):
    row = pd.DataFrame([entry], columns=AUDIT_LOG_COLUMNS)
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(AUDIT_LOG_PATH)
    row.to_csv(AUDIT_LOG_PATH, mode="a", header=write_header, index=False)


# ==============================================================================
# MODEL LOADING — handles cloud-sync placeholders and HDF5-content-under-
# '.keras'-extension files, both confirmed real failure modes on this project.
# ==============================================================================
def _load_single_model(path_no_ext):
    """A model was saved as either .keras or .joblib, load whichever exists."""
    keras_path = path_no_ext + ".keras"
    joblib_path = path_no_ext + ".joblib"
    if os.path.exists(keras_path):
        size = os.path.getsize(keras_path)
        if size < 10_000:
            raise RuntimeError(
                f"{os.path.basename(keras_path)} exists but is only {size} bytes -- this is the signature of a "
                "cloud-sync placeholder (OneDrive Files On-Demand, Google Drive stream mode, etc.) that hasn't "
                "actually downloaded yet, not a real model file. Right-click the file in File Explorer and choose "
                "'Always keep on this device', wait for it to sync, then reload this page."
            )
        with open(keras_path, "rb") as f:
            magic = f.read(8)
        is_legacy_hdf5 = magic.startswith(b"\x89HDF")
        if is_legacy_hdf5:
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
        import tensorflow as tf

        # Fix 1: patch the real Lambda class's __init__ to supply a sane
        # output_shape when the saved config is missing one.
        _original_lambda_init = tf.keras.layers.Lambda.__init__
        def _patched_lambda_init(self, *args, **kwargs):
            if kwargs.get("output_shape") is None:
                kwargs["output_shape"] = lambda input_shape: (input_shape[0], input_shape[-1])
            _original_lambda_init(self, *args, **kwargs)
        tf.keras.layers.Lambda.__init__ = _patched_lambda_init

        # Fix 1b: models saved by a newer Keras write an 'optional' kwarg into
        # InputLayer's config, which older Keras (e.g. 2.15) rejects with
        # "Unrecognized keyword arguments: ['optional']". Drop it on load.
        _original_inputlayer_init = tf.keras.layers.InputLayer.__init__
        def _patched_inputlayer_init(self, *args, **kwargs):
            kwargs.pop("optional", None)
            _original_inputlayer_init(self, *args, **kwargs)
        tf.keras.layers.InputLayer.__init__ = _patched_inputlayer_init

        def _load(path):
            model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
            # Fix 2: custom_objects does NOT reach a raw lambda's globals in
            # Keras 3. A Lambda's saved function is unmarshaled with its
            # __globals__ pointed at Keras's own internal python_utils module
            # dict, completely disconnected from whatever custom_objects you
            # pass to load_model -- confirmed directly by inspecting the
            # loaded function object, not assumed. The only thing that
            # actually works is mutating that function's own __globals__
            # dict in place, on the model object you already have, after
            # load_model() returns.
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Lambda):
                    layer.function.__globals__["tf"] = tf
            return model

        try:
            if is_legacy_hdf5:
                tmp_dir = tempfile.mkdtemp(prefix="pdm_h5_compat_")
                tmp_h5_path = os.path.join(tmp_dir, os.path.basename(path_no_ext) + ".h5")
                shutil.copyfile(keras_path, tmp_h5_path)
                try:
                    return _load(tmp_h5_path)
                except Exception as e:
                    raise RuntimeError(
                        f"{os.path.basename(keras_path)} is HDF5-format data saved under a '.keras' name. Loading it "
                        f"via the legacy H5 loader also failed ({e}). If this mentions an unrecognized layer keyword "
                        "(e.g. LSTM's 'time_major'), install the compatibility shim with `pip install tf_keras` "
                        "(matching your tensorflow version) -- legacy mode is enabled automatically once installed."
                    ) from e
            return _load(keras_path)
        finally:
            tf.keras.layers.Lambda.__init__ = _original_lambda_init
            tf.keras.layers.InputLayer.__init__ = _original_inputlayer_init
    if os.path.exists(joblib_path):
        size = os.path.getsize(joblib_path)
        if size < 1_000:
            raise RuntimeError(
                f"{os.path.basename(joblib_path)} exists but is only {size} bytes -- likely an unsynced cloud "
                "placeholder rather than a real model file. Force it to download fully, then reload this page."
            )
        return joblib.load(joblib_path)
    return None

@st.cache_resource(show_spinner="Loading selected condition classifier and RUL regressor...")
def load_bundle(deploy_dir):
    rules_path = os.path.join(deploy_dir, "preprocessing_and_label_rules.joblib")
    if not os.path.exists(rules_path):
        return None, "Missing: preprocessing_and_label_rules.joblib"
    try:
        classifier = _load_single_model(os.path.join(deploy_dir, "condition_classifier"))
        rul_model = _load_single_model(os.path.join(deploy_dir, "rul_regressor"))
    except Exception as exc:
        return None, f"Failed to load models: {exc}"

    missing = []
    if classifier is None:
        missing.append("condition_classifier.keras or condition_classifier.joblib")
    if rul_model is None:
        missing.append("rul_regressor.keras or rul_regressor.joblib")
    if missing:
        return None, f"Missing: {', '.join(missing)}"

    manifest = None
    manifest_path = os.path.join(deploy_dir, "deployment_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    def _read_csv_if_present(name):
        path = os.path.join(deploy_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    class_cmp = _read_csv_if_present("classification_all_strategies_comparison.csv")
    rul_cmp = _read_csv_if_present("rul_all_models_comparison.csv")
    window_sensitivity = _read_csv_if_present("window_size_sensitivity_rul.csv")

    try:
        rules = joblib.load(rules_path)
    except Exception as exc:
        return None, f"Failed to load preprocessing_and_label_rules.joblib: {exc}"

    return {
        "classifier": classifier, "rul_model": rul_model, "rules": rules,
        "manifest": manifest, "class_cmp": class_cmp, "rul_cmp": rul_cmp,
        "window_sensitivity": window_sensitivity,
    }, None


# ==============================================================================
# RAW DATA PIPELINE — reproduces the notebook's test-time preprocessing over
# the FULL held-out fleet, not a size-capped precomputed snapshot.
# ==============================================================================
@st.cache_data(show_spinner="Reading FD004 raw telemetry...")
def load_raw_test(data_dir):
    test_path = os.path.join(data_dir, "test_FD004.txt")
    rul_path = os.path.join(data_dir, "RUL_FD004.txt")
    if not (os.path.exists(test_path) and os.path.exists(rul_path)):
        return None, None, f"Missing test_FD004.txt or RUL_FD004.txt in {data_dir}"
    test = pd.read_csv(test_path, sep=r"\s+", header=None).iloc[:, :26]
    test.columns = RAW_COLS
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    return test, true_rul, None


def preprocess(test, rules):
    """Reproduces the notebook's test-time clipping, regime scaling and scores."""
    frame = test.copy()
    features = rules["features"]
    for sensor, (low, high) in rules["outlier_bounds"].items():
        frame[sensor] = frame[sensor].clip(low, high)
    op_cols = ["setting1", "setting2", "setting3"]
    frame["regime"] = rules["kmeans"].predict(rules["op_scaler"].transform(frame[op_cols]))
    for regime, scaler in rules["regime_scalers"].items():
        mask = frame["regime"].eq(regime)
        if mask.any():
            frame.loc[mask, features] = scaler.transform(frame.loc[mask, features])
    for score_name, sensors in [("fan_score", rules["fan_sensors"]), ("hpc_score", rules["hpc_sensors"])]:
        squared = frame[sensors].pow(2)
        rolling = squared.groupby(frame["unit"]).transform(lambda x: x.rolling(10, min_periods=3).mean())
        frame[score_name] = np.sqrt(rolling.mean(axis=1)).fillna(0.0)
    return frame


def make_window(engine_hist, features, seq_len, upto_cycle=None):
    """Builds one model-ready [1, seq_len, n_features] window for an engine,
    either its full/latest history (upto_cycle=None) or truncated at a given
    cycle count -- the latter powers the Engine Deep-Dive timeline slider."""
    engine_hist = engine_hist.sort_values("cycle")
    if upto_cycle is not None:
        engine_hist = engine_hist.iloc[:upto_cycle]
    values = engine_hist[features].to_numpy(dtype="float32")
    if len(values) == 0:
        values = np.zeros((1, len(features)), dtype="float32")
    if len(values) < seq_len:
        values = np.vstack([np.repeat(values[:1], seq_len - len(values), axis=0), values])
    window = values[-seq_len:][None, ...]
    return window, window.reshape(1, -1)


def build_sequences(frame, features, seq_len):
    sequences, ids, history = [], [], {}
    for unit, engine in frame.groupby("unit", sort=True):
        engine = engine.sort_values("cycle")
        history[unit] = engine
        window, _ = make_window(engine, features, seq_len)
        sequences.append(window[0])
        ids.append(unit)
    x = np.asarray(sequences, dtype="float32")
    return x, np.asarray(ids), history


def predict_condition_probs(model, x_seq, x_flat, n_classes=len(CLASS_NAMES)):
    """Architecture-agnostic classifier call: sklearn, flat-input Keras (ANN),
    sequence-input Keras (CNN/LSTM/BiLSTM/GRU), or the multi-task hybrid."""
    seq_error = None
    try:
        out = model(x_seq, training=False)
        if isinstance(out, (list, tuple)):
            arr = np.array(out[1])
            if arr.ndim == 2 and arr.shape[1] == n_classes:
                return arr
        arr = np.array(out)
        if arr.ndim == 2 and arr.shape[1] == n_classes:
            return arr
    except Exception as e:
        seq_error = e
    try:
        arr = np.array(model(x_flat, training=False))
        if arr.ndim == 2 and arr.shape[1] == n_classes:
            return arr
    except Exception:
        pass
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_flat)
    try:
        preds = np.asarray(model.predict(x_flat)).astype(int).ravel()
        onehot = np.zeros((len(preds), n_classes))
        onehot[np.arange(len(preds)), preds] = 1.0
        return onehot
    except Exception as flat_error:
        # Every attempt failed. Raise the FIRST (sequence-input) error, since
        # that was the one that should have worked and its real cause was
        # previously being silently discarded by the fallback logic above.
        raise RuntimeError(
            f"Classifier call failed on sequence input: {seq_error}\n"
            f"(also failed on flat input: {flat_error})"
        ) from seq_error


def predict_rul_values(model, x_seq, x_flat):
    """Architecture-agnostic RUL call, same adapter pattern as the classifier."""
    seq_error = None
    try:
        out = model(x_seq, training=False)
        if isinstance(out, (list, tuple)):
            return np.array(out[0]).flatten()
        arr = np.array(out)
        if arr.ndim <= 2:
            return arr.flatten()
    except Exception as e:
        seq_error = e
    try:
        return np.array(model(x_flat, training=False)).flatten()
    except Exception:
        pass
    try:
        return np.asarray(model.predict(x_flat)).flatten()
    except Exception as flat_error:
        raise RuntimeError(
            f"RUL regressor call failed on sequence input: {seq_error}\n"
            f"(also failed on flat input: {flat_error})"
        ) from seq_error


@st.cache_data(show_spinner="Computing XAI sensor attribution across the fleet...")
def compute_feature_impacts(_classifier, _rul_model, x, x_flat, features):
    """Model-agnostic feature-perturbation attribution: zero each sensor in
    turn, measure the shift in predicted RUL. Leading underscores tell
    st.cache_data not to try to hash the model objects themselves."""
    baseline = predict_rul_values(_rul_model, x, x_flat)
    impacts = np.zeros((x.shape[0], len(features)), dtype="float32")
    for i in range(len(features)):
        x_pert = x.copy()
        x_pert[:, :, i] = 0.0
        pert_pred = predict_rul_values(_rul_model, x_pert, x_pert.reshape(len(x_pert), -1))
        impacts[:, i] = np.abs(baseline - pert_pred)
    return impacts


def single_window_impacts(classifier, rul_model, window, window_flat, features):
    """Same perturbation attribution as compute_feature_impacts, but for one
    already-built window -- used by the Engine Deep-Dive timeline slider,
    where recomputing the whole fleet on every slider move would be wasteful."""
    baseline = predict_rul_values(rul_model, window, window_flat)
    impacts = np.zeros(len(features), dtype="float32")
    for i in range(len(features)):
        pert = window.copy()
        pert[:, :, i] = 0.0
        pert_pred = predict_rul_values(rul_model, pert, pert.reshape(1, -1))
        impacts[i] = np.abs(baseline - pert_pred)[0]
    return impacts


@st.cache_data(show_spinner="Running fleet-wide inference...")
def infer_fleet(_classifier, _rul_model, _test, _true_rul, _rules):
    prepared = preprocess(_test, _rules)
    features = _rules["features"]
    seq_len = _rules["sequence_length"]
    x, ids, history = build_sequences(prepared, features, seq_len)
    x_flat = x.reshape(len(x), -1)

    probabilities = predict_condition_probs(_classifier, x, x_flat)
    predicted_rul = np.clip(predict_rul_values(_rul_model, x, x_flat), 0, 125)
    states = probabilities.argmax(axis=1)

    feature_impacts = compute_feature_impacts(_classifier, _rul_model, x, x_flat, features)
    top_driver_idx = feature_impacts.argmax(axis=1)
    drivers = [pretty_sensor(features[i]) for i in top_driver_idx]

    fleet = pd.DataFrame({
        "Engine ID": [f"FD004-{unit:03d}" for unit in ids], "Unit": ids,
        "Predicted State": [CLASS_NAMES[s] for s in states],
        "Predicted RUL": predicted_rul.round(1), "True RUL": _true_rul.round(1), "Top Driver": drivers,
    })
    for i, name in enumerate(CLASS_NAMES):
        fleet[f"P({name})"] = probabilities[:, i].round(3)
    return fleet, probabilities, history, feature_impacts


# ==============================================================================
# DEMO / SIMULATED FALLBACK — used whenever artifacts or raw data are missing
# ==============================================================================
@st.cache_data
def build_demo_fleet():
    np.random.seed(7)
    n = 42
    true_rul = np.random.exponential(scale=35, size=n).clip(0, 125)
    pred_rul = (true_rul + np.random.normal(0, 4, n)).clip(0, 125)
    fault_prob = np.clip((60 - true_rul) / 60, 0.02, 0.9)
    is_faulted = np.random.rand(n) < fault_prob
    fault_type = np.random.choice([1, 2, 2, 3], size=n)  # Fan, HPC (x2 weight), HPC+Fan
    state = np.where(is_faulted, fault_type, 0)

    raw_probs = np.random.dirichlet(alpha=[3, 1, 1, 0.5], size=n)
    for i in range(n):
        boost = np.zeros(4)
        boost[state[i]] = 1.5
        raw_probs[i] = (raw_probs[i] + boost)
        raw_probs[i] /= raw_probs[i].sum()

    drivers = np.random.choice(
        ["s2 (EGT)", "s4 (Pressure Ratio)", "s11 (Core Speed)", "s7 (Fan Speed)", "s15 (Bleed Enthalpy)"], size=n)
    df = pd.DataFrame({
        "Engine ID": [f"SIM-{i:03d}" for i in range(1, n + 1)],
        "Predicted State": [STATE_LABELS[s] for s in state],
        "Predicted RUL": np.round(pred_rul, 1), "True RUL": np.round(true_rul, 1), "Top Driver": drivers,
    })
    for i, name in enumerate(CLASS_NAMES):
        df[f"P({name})"] = np.round(raw_probs[:, i], 3)
    return df


# ==============================================================================
# LOAD EVERYTHING — falls back to demo mode on any missing piece, never crashes
# ==============================================================================
bundle, bundle_error = load_bundle(DEPLOY_DIR)
raw_test, true_rul, data_error = (None, None, None)
LIVE_MODE = False
load_warning = None

if bundle is not None:
    raw_test, true_rul, data_error = load_raw_test(data_dir)
    if raw_test is not None:
        try:
            fleet_df, probabilities, histories, feature_impacts = infer_fleet(
                bundle["classifier"], bundle["rul_model"], raw_test, true_rul, bundle["rules"]
            )
            LIVE_MODE = True
        except Exception as exc:
            load_warning = f"Artifacts and data loaded, but inference failed ({exc}). Falling back to demo mode."
    else:
        load_warning = f"Model artifacts loaded, but raw FD004 data wasn't found ({data_error}). Falling back to demo mode."
else:
    load_warning = f"Could not load deployment artifacts ({bundle_error}). Falling back to demo mode."

if not LIVE_MODE:
    fleet_df = build_demo_fleet()

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.markdown("<h2 style='text-align:center; color:#60a5fa;'>🛰️ Fleet Command</h2>", unsafe_allow_html=True)
mode_badge = '<span class="mode-badge-live">🟢 LIVE MODEL</span>' if LIVE_MODE else '<span class="mode-badge-demo">◆ DEMO / SIMULATED</span>'
st.sidebar.markdown(f"<div style='text-align:center;'>{mode_badge}</div>", unsafe_allow_html=True)
if load_warning:
    st.sidebar.caption(load_warning)
st.sidebar.divider()

st.sidebar.markdown("### 🔑 Gemini API Integration")
if "gemini_client" not in st.session_state:
    st.session_state["gemini_client"] = None
if st.session_state["gemini_client"] is None:
    entered_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="AIza...",
                                         help="Connects once, then this field hides itself. Kept only in this session's memory.")
    if entered_key:
        try:
            st.session_state["gemini_client"] = genai.Client(api_key=entered_key)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Could not connect: {e}")
else:
    st.sidebar.success("🟢 Gemini connected — running in the background")
    if st.sidebar.button("Disconnect / change key", width="stretch"):
        st.session_state["gemini_client"] = None
        st.rerun()
gemini_client = st.session_state["gemini_client"]

st.sidebar.divider()
st.sidebar.markdown("### 🧠 Deployed Models")
if LIVE_MODE and bundle.get("manifest"):
    m = bundle["manifest"]
    clf_info, rul_info = m.get("condition_classifier", {}), m.get("rul_regressor", {})
    st.sidebar.info(
        f"**Classifier:** {clf_info.get('model', 'unknown')} ({clf_info.get('strategy', 'unknown strategy')})  \n"
        f"**RUL regressor:** {rul_info.get('model', 'unknown')}  \nSelected independently on validation metrics, "
        "4-class condition (Healthy/Fan/HPC/HPC+Fan) plus a separate RUL estimate."
    )
elif LIVE_MODE:
    st.sidebar.info("**Live mode** — models loaded, but `deployment_manifest.json` was not found, so model names are unavailable.")
else:
    st.sidebar.info("**Demo mode** — connect real artifacts and FD004 data to show the actual deployed classifier and RUL regressor.")

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("<h1 style='color:#60a5fa; margin-bottom:0;'>Industrial Prescriptive Maintenance Command Center</h1>", unsafe_allow_html=True)
st.caption("C-MAPSS FD004 Turbofan Fleet — Prognostics · Explainable AI · LLM Advisory · Human-in-the-Loop Review")
st.markdown("<p style='color:#9ca3af; font-size:0.85rem;'>Healthy, Fan, HPC and HPC+Fan are inferred sensor-condition "
            "proxies; they are not ground-truth NASA failure-mode labels.</p>", unsafe_allow_html=True)

tab_fleet, tab_engine, tab_gov = st.tabs(["🏭 Fleet Overview", "🔧 Engine Deep-Dive", "📋 Model Governance & Impact"])

# ==============================================================================
# TAB 1 — FLEET OVERVIEW
# ==============================================================================
with tab_fleet:
    n_total = len(fleet_df)
    class_counts = {name: int((fleet_df["Predicted State"] == name).sum()) for name in CLASS_NAMES}
    n_healthy, n_fan, n_hpc, n_compound = (class_counts[c] for c in CLASS_NAMES)
    fleet_health_pct = 100 * n_healthy / n_total
    avg_rul = fleet_df["Predicted RUL"].mean()
    compound_ratio = 100 * n_compound / n_total
    any_fault_ratio = 100 * (n_total - n_healthy) / n_total

    n_flagged_total = n_total - n_healthy

    # --- KPI summary cards, matching the plain metric-card look of image 2 ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engines monitored", n_total)
    c2.metric("Healthy", f"{fleet_health_pct:.1f}%")
    c3.metric("Flagged conditions", n_flagged_total)
    c4.metric("Average predicted RUL", f"{avg_rul:.1f} cycles")

    # --- Fleet condition distribution + Predicted-vs-True RUL scatter -------
    left, right = st.columns([1.2, 1])
    with left:
        state_counts = fleet_df["Predicted State"].value_counts().reindex(CLASS_NAMES).fillna(0)
        fig = go.Figure(go.Bar(x=state_counts.index, y=state_counts.values,
                                marker_color=[STATE_COLOR[s] for s in state_counts.index],
                                text=state_counts.values, textposition="outside"))
        fig.update_layout(title="Fleet condition distribution", height=340,
                           margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        scatter = go.Figure()
        for name in CLASS_NAMES:
            sub = fleet_df[fleet_df["Predicted State"] == name]
            scatter.add_trace(go.Scatter(
                x=sub["Predicted RUL"], y=sub["True RUL"], mode="markers", name=name,
                marker=dict(color=STATE_COLOR[name], size=8),
                text=sub["Engine ID"], hovertemplate="%{text}<br>Predicted: %{x}<br>True: %{y}<extra></extra>",
            ))
        max_axis = max(fleet_df["Predicted RUL"].max(), fleet_df["True RUL"].max()) * 1.05
        scatter.add_shape(type="line", x0=0, y0=0, x1=max_axis, y1=max_axis, line=dict(color="#94a3b8", dash="dash"))
        scatter.update_layout(title="Predicted versus true terminal RUL", height=340,
                               margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb",
                               xaxis_title="Predicted RUL", yaxis_title="True RUL",
                               legend=dict(title="Predicted State"))
        st.plotly_chart(scatter, width="stretch")

    # --- Live gauge row, below the main charts -------------------------------
    panel("Fleet Health — Live Gauges", f"{n_total} engines monitored")
    g1, g2, g3, g4 = st.columns(4)

    def gauge(value, title, color, suffix="%", max_val=100):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=value,
            number={"suffix": suffix, "font": {"size": 28, "color": "#e5e7eb"}},
            title={"text": title, "font": {"size": 13, "color": "#9ca3af"}},
            gauge={
                "axis": {"range": [0, max_val], "tickcolor": "#4b5563"},
                "bar": {"color": color}, "bgcolor": "#1f2937",
                "steps": [{"range": [0, max_val * 0.5], "color": "#1f2937"},
                          {"range": [max_val * 0.5, max_val * 0.8], "color": "#27303f"}],
            }
        ))
        fig.update_layout(height=190, margin=dict(l=15, r=15, t=40, b=5), paper_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
        return fig

    with g1:
        st.plotly_chart(gauge(fleet_health_pct, "Fleet Availability (Healthy %)", PALETTE["green"]), width="stretch")
    with g2:
        st.plotly_chart(gauge(100 - any_fault_ratio, "No Fault Flagged", PALETTE["blue"]), width="stretch")
    with g3:
        st.plotly_chart(gauge(avg_rul, "Avg. Fleet RUL", PALETTE["amber"], suffix=" cyc", max_val=125), width="stretch")
    with g4:
        st.plotly_chart(gauge(compound_ratio, "HPC+Fan (Compound) Ratio", PALETTE["crimson"]), width="stretch")

    col_left, col_right = st.columns([2.2, 1])
    with col_left:
        panel("Top Failure Drivers (Any Non-Healthy Class)")
        driver_counts = fleet_df.loc[fleet_df["Predicted State"] != "Healthy", "Top Driver"].value_counts().sort_values()
        fig2 = go.Figure(go.Bar(x=driver_counts.values, y=driver_counts.index, orientation="h",
                                 marker_color=PALETTE["purple"], text=driver_counts.values, textposition="outside"))
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
        st.plotly_chart(fig2, width="stretch")

    with col_right:
        panel("Fleet Composition")
        donut = go.Figure(go.Pie(labels=CLASS_NAMES, values=[n_healthy, n_fan, n_hpc, n_compound],
                                  hole=0.55, marker_colors=[STATE_COLOR[c] for c in CLASS_NAMES]))
        donut.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                             font_color="#e5e7eb", showlegend=True, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(donut, width="stretch")
        st.markdown(f"<div class='big-stat'>{n_compound}</div><div class='big-stat-label'>Engines in Compound (HPC+Fan) State</div>",
                    unsafe_allow_html=True)

    panel("Fault Conditions Being Monitored (Fan, HPC, or HPC+Fan)", f"{n_total - n_healthy} flagged")
    watch_df = fleet_df[fleet_df["Predicted State"] != "Healthy"].sort_values("Predicted RUL")
    prob_cols = [c for c in fleet_df.columns if c.startswith("P(")]

    def style_state(val):
        color = STATE_COLOR.get(val, "#9ca3af")
        return f"background-color: {color}22; color: {color}; font-weight: 700;"

    if len(watch_df) > 0:
        styled = watch_df.style.apply(lambda col: col.map(style_state), subset=["Predicted State"])
        if prob_cols:
            styled = styled.format({c: "{:.1%}" for c in prob_cols})
        st.dataframe(styled, width="stretch", height=320)
    else:
        st.success("No Fan, HPC, or HPC+Fan engines in the current fleet snapshot.")

    if not LIVE_MODE:
        st.caption("◆ Demo mode — this tab is populated with simulated fleet data. Connect real model artifacts and "
                    "FD004 data (see sidebar) to see the actual held-out test fleet.")
    else:
        st.caption(f"Live model — fleet metrics cover all {n_total} held-out test engines from the raw FD004 files.")

# ==============================================================================
# TAB 2 — ENGINE DEEP-DIVE
# ==============================================================================
with tab_engine:
    if LIVE_MODE:
        rules = bundle["rules"]
        features = rules["features"]
        seq_len = rules["sequence_length"]
        engine_ids_all = fleet_df["Engine ID"].tolist()
        engine_choice = st.selectbox("Select Asset for Deep-Dive (full real held-out test fleet)", engine_ids_all)
        unit = int(fleet_df.loc[fleet_df["Engine ID"] == engine_choice, "Unit"].iloc[0])
        engine_hist = histories[unit]
        n_cycles = len(engine_hist)

        cur_cycle = st.slider("Timeline Cursor (cycles observed so far)", 1, n_cycles, n_cycles)
        window, window_flat = make_window(engine_hist, features, seq_len, upto_cycle=cur_cycle)

        rul_pred_raw = predict_rul_values(bundle["rul_model"], window, window_flat)
        state_probs_raw = predict_condition_probs(bundle["classifier"], window, window_flat, len(CLASS_NAMES))
        predicted_rul = max(0.0, float(np.array(rul_pred_raw).flatten()[0]))
        state_probs = np.array(state_probs_raw).flatten()
        predicted_state_id = int(np.argmax(state_probs))
        state_text = STATE_LABELS[predicted_state_id]
        failure_prob = float(state_probs[WORST_CLASS_IDX])
        true_rul_val = float(fleet_df.loc[fleet_df["Engine ID"] == engine_choice, "True RUL"].iloc[0])
        margin = None
        if bundle.get("manifest"):
            margin = bundle["manifest"].get("rul_regressor", {}).get("val_rmse")

        impacts = single_window_impacts(bundle["classifier"], bundle["rul_model"], window, window_flat, features)
        top_idx = np.argsort(impacts)[::-1][:5]
        top_sensors = [pretty_sensor(features[i]) for i in top_idx]
        top_impacts = [float(impacts[i]) for i in top_idx]

        telemetry_df = engine_hist.sort_values("cycle").iloc[:cur_cycle][["cycle"] + features].rename(columns={"cycle": "Cycle"})
    else:
        st.info("◆ Demo mode — showing one simulated asset. Connect real model artifacts and FD004 data to browse "
                "the real held-out test fleet.")
        engine_choice = "SIM-ENGINE-001"
        cur_cycle = st.slider("Simulated Operational Cycle", 50, 145, 95)
        np.random.seed(101 + cur_cycle)
        true_rul_val = max(0, 145 - cur_cycle)
        state_id = 3 if true_rul_val <= 10 else (2 if true_rul_val <= 25 else (1 if true_rul_val <= 45 else 0))
        state_text = STATE_LABELS[state_id]
        predicted_rul = max(0, min(125, true_rul_val) + np.random.normal(0, 1.8))
        state_probs = np.random.dirichlet(alpha=[2, 1, 1, 0.5])
        boost = np.zeros(4); boost[state_id] = 2.0
        state_probs = (state_probs + boost); state_probs /= state_probs.sum()
        failure_prob = float(state_probs[WORST_CLASS_IDX])
        margin = None
        top_sensors = ["Sensor 2 (EGT)", "Sensor 11 (Core Speed)", "Sensor 4 (Pressure Ratio)"]
        cf = cur_cycle / 145.0
        top_impacts = [0.38 * cf, 0.29 * cf, 0.18 * cf]
        cycles = list(range(50, cur_cycle + 1))
        telemetry_df = pd.DataFrame({
            "Cycle": cycles,
            "Sensor_2_EGT": [549.68 + c * 0.28 + np.random.normal(0, 0.6) for c in cycles],
            "Sensor_11_CoreSpeed": [41.76 + c * 0.02 + np.random.normal(0, 0.05) for c in cycles],
        })

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted condition", state_text)
    k2.metric("RUL estimate", f"{predicted_rul:.1f} cycles",
              delta=(f"± {margin:.1f} cycles (val. RMSE)" if margin else None), delta_color="off",
              help="Validation-set RMSE of the selected RUL regressor, shown as a rough uncertainty band." if margin else None)
    k3.metric("Known terminal RUL", f"{true_rul_val:.1f} cycles" if LIVE_MODE else "N/A (demo mode)")
    k4.metric("Primary telemetry driver", top_sensors[0])

    st.divider()
    c1, c2 = st.columns([3, 2])
    with c1:
        panel("Sensor Telemetry Stream")
        sensor_cols = [c for c in telemetry_df.columns if c != "Cycle"]
        sensor_display_map = {pretty_sensor(c): c for c in sensor_cols}
        sensor_choice_label = st.selectbox("Telemetry Signal", list(sensor_display_map.keys()))
        sensor_choice = sensor_display_map[sensor_choice_label]
        st.line_chart(telemetry_df.set_index("Cycle")[sensor_choice])
        if LIVE_MODE:
            st.caption("Per-regime standardized sensor readings, as fed to the model (not raw physical units).")
    with c2:
        panel("Class Probabilities — All 4 States")
        prob_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": state_probs}).sort_values("Probability")
        fig_probs = go.Figure(go.Bar(x=prob_df["Probability"], y=prob_df["Class"], orientation="h",
                                      marker_color=[STATE_COLOR[c] for c in prob_df["Class"]],
                                      text=[f"{p*100:.1f}%" for p in prob_df["Probability"]], textposition="outside"))
        fig_probs.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb", xaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_probs, width="stretch")

        panel("Explainable AI — Top Drivers")
        xai_df = pd.DataFrame({"Sensor": top_sensors, "Attribution": top_impacts}).sort_values("Attribution")
        fig_xai = go.Figure(go.Bar(x=xai_df["Attribution"], y=xai_df["Sensor"], orientation="h", marker_color=PALETTE["purple"]))
        fig_xai.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
        st.plotly_chart(fig_xai, width="stretch")
        st.caption("Higher weight = zeroing that sensor shifts predicted RUL more." if LIVE_MODE else "Simulated attribution for demo purposes.")

    st.divider()
    panel("LLM Maintenance Advisory & Human-in-the-Loop Review")

    FALLBACK_ADVISORY = {
        "HPC+Fan": "**Ground the asset immediately.** Both HPC and Fan sensors are flagged together. Stage replacement "
                   "core and fan modules before next dispatch. No further flight cycles authorized pending review.",
        "HPC": "**Schedule a compressor inspection within the next 10 operational cycles.** Pre-stage HPC blade/seal "
               "pack. Restrict to short-haul routes until inspected.",
        "Fan": "**Schedule a fan-module inspection within the next 15 operational cycles.** Check fan blade condition "
               "and bleed valves. Continue normal routing pending inspection.",
        "Healthy": "**Maintain baseline operation.** Routine scheduled check at the next maintenance interval.",
    }

    def generate_gemini_prescription(state_name, predicted_rul_value, failure_prob_value,
                                 top_sensors, top_impacts, live_mode, client):
        """Returns (text, ok). ok=False means the call failed."""
        import time

        system_instruction = (
            "You are an expert aerospace propulsion and systems maintenance engineer operating under Industry 5.0 "
            "standards. Review ML diagnostic predictions for a gas turbine engine (C-MAPSS FD004) and generate a "
            "precise, actionable prescriptive maintenance plan compliant with ISO 13374. The predicted state is one "
            "of four sensor-derived proxy classes: Healthy, Fan, HPC, or HPC+Fan. Tailor the plan to the specific "
            "component(s) implicated. Use markdown with three headers: 📋 ACTION PLAN, 🗓️ LOGISTICS & SCHEDULE, "
            "and 📦 PARTS REQUIRED."
            )

        def _fmt_attribution(v):
            try:
                return f"{float(v):.3f}"
            except (TypeError, ValueError):
                return str(v)

        driver_lines = "\n".join(
            f"- {s}: attribution = {_fmt_attribution(v)}"
            for s, v in zip(top_sensors, top_impacts)
            )
        source_note = "real held-out test engine" if live_mode else "simulated demo telemetry"
        user_message = (
            f"Review diagnostics for the active engine node ({source_note}):\n"
            f"- Predicted state: {state_name}\n"
            f"- Predicted RUL: {predicted_rul_value:.1f} cycles\n"
            f"- P(HPC+Fan, worst class): {failure_prob_value*100:.2f}%\n"
            f"Top contributing sensors (XAI attribution):\n{driver_lines}\n"
            f"Write a custom engineering prescriptive action plan, scheduling guidance, and replacement parts."
        )

        RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "504", "DEADLINE_EXCEEDED")
        backoff_seconds = [2, 5]  # delays before retries

        for attempt in range(len(backoff_seconds) + 1):
            try:
                response = client.models.generate_content(
                model="models/gemini-3.1-flash-lite",
                contents=[system_instruction, user_message]
                )
                return response.text, True
            except Exception as e:
                err_str = str(e)
                if any(marker in err_str for marker in RETRYABLE_MARKERS) and attempt < len(backoff_seconds):
                    time.sleep(backoff_seconds[attempt])
                    continue
            return f"Gemini API error: {err_str}", False

    # Cached per engine ID, not in one shared slot -- switching engines above
    # must not show a stale prescription generated for a different engine.
    prescription_key = f"prescription_{engine_choice}"
    advisory_text, advisory_source = None, None

    if gemini_client:
        if st.button("Generate Gemini Prescription", width="stretch"):
            with st.spinner("Analyzing telemetry with Gemini... (will auto-retry briefly if Gemini's servers are busy)"):
                text, ok = generate_gemini_prescription(
                    state_text, predicted_rul, failure_prob, top_sensors, top_impacts, LIVE_MODE, gemini_client
                )
                st.session_state[prescription_key] = {"text": text, "ok": ok, "source": "gemini"}
        cached = st.session_state.get(prescription_key)
        if cached:
            if cached["ok"]:
                st.info(cached["text"])
                advisory_text, advisory_source = cached["text"], "gemini"
            else:
                st.error(f"⚠️ {cached['text']} — not logged as a valid advisory. Falling back to rule-based guidance below.")
                st.markdown(FALLBACK_ADVISORY[state_text])
                advisory_text, advisory_source = FALLBACK_ADVISORY[state_text], "fallback (gemini error)"
    else:
        st.warning("Connect a Gemini API key in the sidebar to generate a custom prescription. Fallback rule-based guidance:")
        st.markdown(FALLBACK_ADVISORY[state_text])
        advisory_text, advisory_source = FALLBACK_ADVISORY[state_text], "fallback (no key)"

    st.write("---")
    st.markdown("**Human Operator Audit Trail** — AI proposes; a certified maintenance expert always authorizes.")
    with st.form(f"hitl_form_{engine_choice}"):
        verdict = st.radio("Authorized Action:", ["Deploy prescribed work order", "Modify schedule / parts", "Reject recommendation"])
        advisory_rating = st.select_slider(
            "How useful/accurate was the advisory above?",
            options=["1 - Poor", "2 - Weak", "3 - Adequate", "4 - Good", "5 - Excellent"], value="3 - Adequate",
        )
        remarks = st.text_area("Audit remarks (required):")
        if st.form_submit_button("Submit & Log"):
            if not remarks.strip():
                st.error("Audit remarks are required before logging a decision.")
            else:
                entry = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "engine_id": engine_choice, "predicted_state": state_text,
                    "predicted_rul": round(float(predicted_rul), 1),
                    "true_rul": round(float(true_rul_val), 1) if LIVE_MODE else None,
                    "advisory_source": advisory_source or "none generated",
                    "advisory_text": (advisory_text or "").replace("\n", " ")[:2000],
                    "verdict": verdict, "advisory_rating": advisory_rating, "remarks": remarks.strip(),
                }
                try:
                    append_audit_entry(entry)
                    st.toast("Logged to fleet audit trail.", icon="✅")
                    st.success(f"Verdict logged for {engine_choice}: *{verdict}* (rating {advisory_rating}, source: {entry['advisory_source']})")
                except Exception as e:
                    st.error(f"Could not write to audit log: {e}")

    audit_log = load_audit_log()
    with st.expander(f"📜 Audit trail ({len(audit_log)} logged decisions)", expanded=False):
        if audit_log.empty:
            st.caption(f"No decisions logged yet. Submitted verdicts persist to `{os.path.basename(AUDIT_LOG_PATH)}` and will appear here.")
        else:
            st.dataframe(audit_log.sort_values("timestamp_utc", ascending=False), width="stretch", height=280)
            st.download_button("Download full audit log (CSV)", data=audit_log.to_csv(index=False).encode("utf-8"),
                                file_name="fd004_audit_log.csv", mime="text/csv")

# ==============================================================================
# TAB 3 — MODEL GOVERNANCE & IMPACT
# ==============================================================================
with tab_gov:
    panel("Selected Deployment Models vs All Candidates Compared")
    if LIVE_MODE and bundle.get("manifest"):
        m = bundle["manifest"]
        clf_info, rul_info = m.get("condition_classifier", {}), m.get("rul_regressor", {})
        st.dataframe(pd.DataFrame([
            ["Condition classifier", clf_info.get("model", "unknown"), "Validation macro-F1", clf_info.get("val_macro_f1"),
             "Validation balanced accuracy", clf_info.get("val_balanced_accuracy")],
            ["RUL regressor", rul_info.get("model", "unknown"), "Validation RMSE", rul_info.get("val_rmse"),
             "Validation MAE", rul_info.get("val_mae")],
        ], columns=["Task", "Model", "Validation metric", "Value", "Secondary metric", "Value 2"]), width="stretch")
    elif LIVE_MODE:
        st.caption("Model loaded, but `deployment_manifest.json` was not found — selected-model metrics unavailable.")
    else:
        st.dataframe(pd.DataFrame({
            "Model": ["Random Forest", "SVM", "ANN", "1D-CNN", "CNN-BiLSTM Hybrid (deployed)"],
            "Task": ["4-class classification"] * 4 + ["4-class + RUL regression"],
            "Note": ["No RUL output, no sequence modeling"] * 4 + ["Adopted for prescriptive decisions"],
        }), width="stretch")
    if LIVE_MODE and bundle.get("class_cmp") is not None:
        st.markdown("**All classification strategies compared** (`classification_all_strategies_comparison.csv`)")
        st.dataframe(bundle["class_cmp"], width="stretch")
    if LIVE_MODE and bundle.get("rul_cmp") is not None:
        st.markdown("**All RUL regressors compared** (`rul_all_models_comparison.csv`)")
        st.dataframe(bundle["rul_cmp"], width="stretch")

    if LIVE_MODE and bundle.get("window_sensitivity") is not None:
        panel("RUL Regressor Robustness to Sequence Window Size",
              "test RMSE distribution across repeated seeds per window size")
        st.caption(
            "Generated by the notebook's window-size robustness sweep cell, not the deployed "
            "model's own training run — SEQ_LEN stays fixed at its deployed value for every "
            "prediction shown elsewhere in this dashboard."
        )
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import ptitprince as pt

            ws_df = bundle["window_sensitivity"]
            fig_rc, ax_rc = plt.subplots(figsize=(7.5, 4.6))
            pt.RainCloud(
                x="window_size", y="test_rmse", data=ws_df, ax=ax_rc,
                palette="Set2", bw=.3, width_viol=.55, orient="v",
                box_showfliers=False, point_size=3, alpha=.75,
            )
            ax_rc.set_xlabel("Window Size (cycles)")
            ax_rc.set_ylabel("Test RMSE (cycles)")
            ax_rc.set_title(f"Winning architecture: {bundle['manifest'].get('rul_regressor', {}).get('model', 'unknown')}"
                             if bundle.get("manifest") else "")
            st.pyplot(fig_rc, clear_figure=True)
        except ImportError:
            st.info("Install `ptitprince` and `matplotlib` to render this chart (`pip install ptitprince`).")

    panel("Illustrative Maintenance Cost-Avoidance Estimator", "assumptions are editable")
    st.caption("These figures are **not** derived from real financial data — adjust the assumptions to your own fleet's costs.")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        cost_unplanned = st.number_input("Cost of an unplanned in-flight failure ($)", value=250000, step=10000)
    with cc2:
        cost_scheduled = st.number_input("Cost of a scheduled preventive repair ($)", value=35000, step=5000)
    with cc3:
        catch_rate = st.slider("Assumed % of Fan/HPC/HPC+Fan cases caught early", 0, 100, 85)
    n_flagged = int((fleet_df["Predicted State"] != "Healthy").sum())
    n_caught = int(n_flagged * catch_rate / 100)
    savings = n_caught * (cost_unplanned - cost_scheduled)
    m1, m2, m3 = st.columns(3)
    m1.metric("Flagged Engines (Fan+HPC+HPC/Fan)", n_flagged)
    m2.metric("Assumed Early Interventions", n_caught)
    m3.metric("Illustrative Cost Avoidance", f"${savings:,.0f}")

    panel("Model Card")
    if LIVE_MODE:
        rules = bundle["rules"]
        manifest = bundle.get("manifest") or {}
        clf_info, rul_info = manifest.get("condition_classifier", {}), manifest.get("rul_regressor", {})
        st.json({
            "architecture": "Two independently selected models: a condition classifier and a RUL regressor "
                             "(not a single multi-task hybrid).",
            "condition_classifier": {"model": clf_info.get("model"), "strategy": clf_info.get("strategy"),
                                      "val_macro_f1": clf_info.get("val_macro_f1")},
            "rul_regressor": {"model": rul_info.get("model"), "val_rmse": rul_info.get("val_rmse")},
            "classes": rules.get("classes", CLASS_NAMES),
            "sequence_length": rules.get("sequence_length"),
            "input_features": [pretty_sensor(f) for f in rules.get("features", [])],
            "label_thresholds": {"fan_threshold": float(rules.get("fan_threshold", 0)),
                                  "hpc_threshold": float(rules.get("hpc_threshold", 0))},
            "label_note": rules.get("label_note", "Healthy/Fan/HPC/HPC+Fan are sensor-derived proxy labels, "
                                                    "not C-MAPSS ground-truth fault modes."),
            "human_review": "Required before maintenance action.",
        })
    else:
        st.markdown("Connect real model artifacts to display the live model card (architecture, thresholds, training configuration).")
