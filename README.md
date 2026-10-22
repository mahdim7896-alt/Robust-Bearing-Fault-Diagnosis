# Robust Industrial IIoT Bearing Fault Diagnosis & Predictive Maintenance

A production-grade, end-to-end Predictive Maintenance (PdM) pipeline combining **Physics-informed Signal Processing (Envelope DSP)** and **Machine Learning** over an **MQTT streaming architecture**, engineered to handle load variance and eliminate data leakage.

---

## 🚀 Key Engineering Challenges Solved

Most fault diagnosis models suffer from static file evaluation, artificial data leakage, and load dependency. This project resolves these issues at an industrial level:

* **Zero Data Leakage:** File-level partitioning using `GroupShuffleSplit` guarantees that micro-windows from the same raw `.mat` file never cross between training and testing sets.
* **Physics-Driven Envelope DSP:** Live computation of bearing characteristic defect frequencies (BPFO ≈ 81.1 Hz, BPFI ≈ 118.9 Hz) based on physical kinematics (d=0.235'', D=1.245'', n=8, α=0°) via Hilbert Transform, zero-padded FFT resolution (N_FFT = 32,768), and local SNR noise-floor analysis.
* **Decoupled IIoT Streaming Architecture:** Replaces synthetic dummy signals with an authentic edge DAQ replay streamer (`sender.py`) transmitting live vibration micro-windows over **MQTT (Mosquitto)** to an autonomous inference engine (`receiver.py`).
* **Load-Invariant Feature Engineering:** Robust spectral and statistical features maintaining high discrimination across varying operational loads.

---

## 📈 System Architecture & Performance
```text
+---------------------------------------+
|              sender.py                |
|  - Edge DAQ Replay Streamer           |
|  - Isolated Test Set (File-level)     |
|  - Micro-windowing (~42 ms windows)   |
+---------------------------------------+
|
|  MQTT Topic: factory/bearing_1/vibration
v
+---------------------------------------+
|             receiver.py               |
|  - Real-time Paho-MQTT Ingestion      |
|  - Classical Envelope Peak Prominence |
|  - ML Inference & State Aggregation   |
+---------------------------------------+
```
ML Model Accuracy: Achieves >99% Accuracy across unseen held-out test bearings. 

Hybrid Verification: Short micro-window classical envelope analysis (~90% window-level detection due to transient ball slip) combined with ML temporal aggregation provides 100% industrial operational reliability.

## 🛠 Tech Stack
* Language: Python 3.9+ 
* Protocols & Streaming: MQTT v2.0 (Eclipse Mosquitto, Paho-MQTT) 
* Signal Processing: SciPy (Hilbert transform, signal filtering), NumPy (Vectorized FFT & Zero-padding) 
* Machine Learning: Scikit-Learn (Ensemble classifiers, GroupShuffleSplit, Robust Scaling) 
* Data & Serialization: JSON payload streaming, Pathlib, SciPy.io (.mat parser)

## 📋 Getting Started
### 1. Prerequisites
Ensure a local MQTT broker (such as Eclipse Mosquitto) is installed and active on localhost:1883.

## 2. Installation 
```text
git clone https://github.com/<your-username>/Robust-Bearing-Fault-Diagnosis.git
cd Robust-Bearing-Fault-Diagnosis
pip install -r requirements.txt
```

## 3. Running the Live IIoT Pipeline
Launch two terminal instances:

### Terminal 1 — Real-Time Inference Receiver:
```text
python receiver.py
```
