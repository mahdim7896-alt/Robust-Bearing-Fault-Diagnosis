"""
================================================================================
Industrial IIoT Bearing Fault Diagnosis System - Receiver & Inference Engine
================================================================================

Description:
    This script acts as the MQTT subscriber and diagnostic engine in a decoupled 
    Predictive Maintenance (PdM) pipeline. It listens to real-time vibration 
    data streamed from the DAQ simulator, runs Machine Learning classification, 
    and validates results using Traditional DSP (Local SNR / Fault Frequencies).

Key Features & Pipeline Steps:
    1. MQTT Networking:
       - Connects to the local Mosquitto Broker using modern Paho-MQTT API v2.
       - Subscribes to the vibration topic (`factory/bearing_1/vibration`) 
         to receive continuous time-domain signal windows (2048 samples).
         
    2. Machine Learning Inference:
       - Extracts statistical and frequency-domain features from each window.
       - Standardizes features using a pre-trained StandardScaler.
       - Predicts the health state (Normal, Outer Race Fault, Inner Race Fault) 
         using a trained ML classifier.

    3. Classical DSP Validation (Load-Invariant Local SNR):
       - Implements high-resolution Zero-Padding (n_fft = 32768) to refine 
         frequency bins and avoid zero-size array errors.
       - Computes characteristic fault frequencies (BPFO & BPFI) based on MFPT 
         bearing geometry and shaft rotational speed ($25 \text{ Hz}$).
       - Calculates local signal-to-noise ratio (SNR) by comparing peak amplitude 
         against the local median noise floor.
       - Evaluates multi-fault conditions to output robust classical decisions.

    4. Robust System Design:
       - Handles graceful shutdowns via KeyboardInterrupt (Ctrl+C).
       - Implements safe connection-refused error catching.

Author: Mahdi
Architecture: Hybrid IIoT Predictive Maintenance Pipeline
================================================================================
"""

# ==========================================
# IMPORTS & DEPENDENCIES
# ==========================================
import json
import joblib
import pandas as pd 
from pathlib import Path
import numpy as np
import paho.mqtt.client as mqtt
from mfpt import extract_features_from_window  # Local Custom Modules
from scipy.signal import butter, sosfilt, hilbert

# ===============================================================
# PATH CONFIGURATION
# ===============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pdm_bundle.joblib"


# ===============================================================
# LOAD ARTIFACTS
# ===============================================================
print("[INIT] Loading PdM bundle from disk...")
bundle = joblib.load(MODEL_PATH)

scaler = bundle["scaler"]
model = bundle["model"]
feature_cols = bundle["feature_cols"]
class_names = bundle["class_names"]
print("[SUCCESS] All model artifacts loaded successfully.")

# ===============================
# NETWORK & BROKER CONFIGURATION
# ===============================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "factory/bearing_1/vibration"
KEEP_ALIVE = 60


# ====================
# --- MQTT CALLBACKS
# ====================
def on_connect(client, _userdata, _flags, reason_code, _properties=None):
    """Callback triggered when the client connects to the MQTT broker."""
    if reason_code == 0:
        print(f"[CONNECTED] Successfully connected to broker at {MQTT_BROKER}:{MQTT_PORT}")
        # Subscribe to the vibration stream topic
        client.subscribe(MQTT_TOPIC)
        print(f"[SUBSCRIBED] Listening for vibration windows on topic: '{MQTT_TOPIC}'")
    else:
        print(f"[ERROR] Failed to connect to broker. Reason code: {reason_code}")

def calculate_bearing_frequencies(n=8, d=0.235, D=1.245, alpha_deg=0.0, fr=25.0):
    """
    Automatic calculation of bearing characteristic fault frequencies based on
    bearing geometry and shaft rotational speed.

    """
    alpha_rad = np.radians(alpha_deg)
    gamma = (d / D) * np.cos(alpha_rad)
    
    bpfo = (n / 2.0) * fr * (1.0 - gamma)
    bpfi = (n / 2.0) * fr * (1.0 + gamma)
    bsf  = (D / (2.0 * d)) * fr * (1.0 - gamma**2)
    ftf  = 0.5 * fr * (1.0 - gamma)
    
    return {
        "BPFO": bpfo, 
        "BPFI": bpfi ,
        "BSF":  bsf, 
        "FTF":  ftf 
    }

def detect_fault_at_frequency(signal_window, target_freq, Fs=48828, threshold_ratio=2.5):
    """
    Detects bearing fault peak using FFT spectrum and local SNR threshold.
    """    
    n_fft = 32768
    N = len(signal_window)


    # 1. FFT spectrum and fine frequency axis
    amps = np.abs(np.fft.rfft(signal_window, n=n_fft)) * (2.0 / len(signal_window))
    freqs = np.fft.rfftfreq(n_fft, d=1/Fs)

    
    # 2. Narrow search window for peak (+/- 5.0 Hz)
    peak_mask = (freqs >= target_freq - 5.0) & (freqs <= target_freq + 5.0)
    if not np.any(peak_mask):
        return False, 0.0
    
    # 3. Local noise window excluding the peak (+/- 25.0 Hz)
    band_mask = (freqs >= target_freq - 25.0) & (freqs <= target_freq + 25.0)
    
    # 4. Local noise floor excluding the peak
    noise_mask = band_mask & (~peak_mask)
    if np.any(noise_mask):
        noise_floor = np.median(amps[noise_mask])
    else:
        noise_floor = np.median(amps)
    
    peak_amp = np.max(amps[peak_mask])

    
    snr_ratio = peak_amp / (noise_floor + 1e-9)
    is_fault = bool(snr_ratio >= threshold_ratio)
    
    return is_fault, snr_ratio


def run_classical_dsp_pipeline(vibration_signal,fs=48828):

    sos = butter(4, [2000, 5000], btype='bandpass', fs=fs, output='sos')
    filtered_signal = sosfilt(sos, vibration_signal)
    
    # Compute envelope via Hilbert transform and remove DC offset
    envelope = np.abs(hilbert(filtered_signal))
    processed_signal = envelope - np.mean(envelope)

    bearing_freqs = calculate_bearing_frequencies()

    # Test for Outer Race fault
    is_outer_fault, outer_snr = detect_fault_at_frequency(
        processed_signal, 
        target_freq=bearing_freqs["BPFO"]
    )

    # Test for Ineer Race fault
    is_inner_fault, inner_snr = detect_fault_at_frequency(
        processed_signal, 
        target_freq=bearing_freqs["BPFI"]
    )


    # --- Evaluate Fault Conditions and Log Decision ---
    
    if is_outer_fault:
        dsp_decision = "OUTER_RACE"
    elif is_inner_fault:
        dsp_decision = "INNER_RACE"
    elif is_outer_fault and is_inner_fault:
        dsp_decision = "BOTH_FAULTS"
    else:
        dsp_decision = "NORMAL"

    print(f"  [DSP DETAILS] BPFO Fault: {is_outer_fault} (SNR: {outer_snr:.2f}) | BPFI Fault: {is_inner_fault} (SNR: {inner_snr:.2f})")
    print(f"  └── [TRADITIONAL DSP] Decision: >>> {dsp_decision} <<<")
    
    return dsp_decision




def on_message(_client, _userdata, message):
    """Callback triggered whenever a new vibration payload arrives."""
    # Step 1: Decode the raw payload from bytes to string
    raw_payload_str = message.payload.decode("utf-8")
    
    # Step 2: Convert JSON string into a Python dictionary
    data = json.loads(raw_payload_str)
    
    # Step 3: Extract metadata and raw signal window
    file_name = data.get("file_name", "UNKNOWN")
    win_idx = data.get("window_index", -1)
    raw_samples = data.get("raw_samples", [])
    
    print(f"[RECEIVED] Win #{win_idx:03d} from '{file_name}' | Samples count: {len(raw_samples)}")

    signal = np.array(raw_samples, dtype=np.float64)
    if len(signal) == 0:
        print("[WARN] Received an empty vibration window. Skipping...")
        return

    #  ML Inference & Classical DSP Pipeline Execution ---
    features_dict = extract_features_from_window(signal)
    feat_df = pd.DataFrame([features_dict])[feature_cols]
    X_scaled = scaler.transform(feat_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    pred_idx = model.predict(X_scaled_df)[0]
    pred_label = class_names[pred_idx]
    print(f"  └── [INFERENCE] Result: >>> {pred_label.upper()} <<< (Class ID: {pred_idx})\n")
    run_classical_dsp_pipeline(signal)
    print("\n" * 3)

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    # Create MQTT client using Callback API v2
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    # Bind callback functions to network events
    client.on_connect = on_connect
    client.on_message = on_message

    print("[SYSTEM] Connecting to MQTT Broker at localhost:1883...")

    try:
        # Connect to the local broker
        client.connect("localhost", 1883, keepalive=60)
        
        # Start the continuous listening loop (blocking)
        print("[SYSTEM] Inference engine is running. Waiting for telemetry data...\n")
        client.loop_forever()

    except KeyboardInterrupt:
        # Handle graceful shutdown on user interruption (Ctrl+C)
        print("\n[SYSTEM] Stopping receiver gracefully...")
        client.disconnect()

    except ConnectionRefusedError:
        # Handle connection failure if the MQTT broker is down or unreachable
        print("\n[ERROR] Could not connect to Mosquitto Broker! Is the service running?")





