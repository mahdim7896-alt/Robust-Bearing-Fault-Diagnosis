import os
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

# ---------------------------------------------------------------------------
# Import your own pipeline functions without duplicating any code
# (Replace 'main_pipeline' with your actual script filename without .py)
# ---------------------------------------------------------------------------
from mfpt import (
    load_single_mat,
    preprocess_signal,
    WINDOW_SIZE,
    STEP_SIZE,
)

# ---------------- Configuration ----------------
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "factory/bearing_1/vibration"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RANDOM_STATE = 42
TEST_RATIO = 0.25
REPLAY_SECONDS = 20.0
STREAM_INTERVAL_SEC = 1.0  # Transmission rate delay per window


def get_test_files_by_group(data_dir: str, test_ratio: float = 0.2, seed: int = 42):
    """
    Scans raw .mat files, labels them by filename, and performs
    a GroupShuffleSplit at the raw file level to isolate the Test Set.
    """
    file_records = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith(".mat"):
                full_path = os.path.join(root, f)
                name_lower = f.lower()

                if "baseline" in name_lower or "normal" in name_lower:
                    lbl = 0
                elif "outer" in name_lower:
                    lbl = 1
                elif "inner" in name_lower:
                    lbl = 2
                else:
                    continue

                file_records.append({
                    "file_name": f,
                    "file_path": full_path,
                    "label": lbl
                })

    if not file_records:
        raise FileNotFoundError(f"No valid .mat files located in {data_dir}")

    file_names = np.array([r["file_name"] for r in file_records])
    gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    _, test_idx = next(gss.split(file_names, groups=file_names))

    return [file_records[i]["file_path"] for i in test_idx]

def baseline_first_sort(file_path):
    """
    Prioritize healthy baseline data over fault cases in the replay queue.
    Ensures the receiver first validates normal operating conditions before
    processing progressive degradation and bearing fault patterns.
    """
    name = Path(file_path).name.lower()
    if "normal" in name or "baseline" in name:
        return 0  # اولویت اول (سالم)
    return 1      # اولویت دوم (خرابی‌ها)


def slice_into_windows(signal: np.ndarray, window_size: int, step_size: int):
    """Slices 1D raw signal into discrete windows."""
    total_len = len(signal)
    for start in range(0, total_len - window_size + 1, step_size):
        yield signal[start : start + window_size]


def main():
    """
    Orchestrate the continuous streaming of MFPT vibration test windows
    to the MQTT broker, simulating a real-time industrial sensor DAQ.
    """
    client = mqtt.Client()
    try:
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        print(f"[MQTT] Connected to Broker at {BROKER_HOST}:{BROKER_PORT}")
    except Exception as e:
        print(f"[MQTT ERROR] Failed to connect: {e}")
        return

    print("[INIT] Scanning files and isolating raw Test Set...")
    test_files = get_test_files_by_group(DATA_DIR, test_ratio=TEST_RATIO, seed=RANDOM_STATE)
    sorted_test_files = sorted(test_files, key=baseline_first_sort)
    # for file_path in sorted_test_files:
    #     print(f"\n[SENDER] Streaming test file: {Path(file_path).name}")

    print(f"[INIT] {len(test_files)} test files ready for transmission.")
    print(*[os.path.basename(f) for f in sorted_test_files], sep="\n")
    while True:
        for file_path in sorted_test_files:
            file_name = os.path.basename(file_path)

            raw_signal, fs, shaft_speed, load = load_single_mat(file_path)
            clean_signal = preprocess_signal(raw_signal)

            # Ensure continuous ~20 seconds duration
            target_samples = int(REPLAY_SECONDS * fs)
            if len(clean_signal) < target_samples:
                repeats = int(np.ceil(target_samples / len(clean_signal)))
                stream_buffer = np.tile(clean_signal, repeats)[:target_samples]
            else:
                stream_buffer = clean_signal[:target_samples]

            print(f"\n[STREAMING] File: {file_name} | Fs: {fs}Hz | Speed: {shaft_speed}Hz | Load: {load} lbs")

            win_idx = 0
            for window in slice_into_windows(stream_buffer, WINDOW_SIZE, STEP_SIZE):
                payload = {
                    "file_name": file_name,
                    "window_index": win_idx,
                    "fs": fs,
                    "shaft_speed_hz": shaft_speed,
                    "load_lbs": float(load) if not np.isnan(load) else 0.0,
                    "raw_samples": window.tolist()
                }

                client.publish(TOPIC, json.dumps(payload))
                # print(f"[SENT] {file_name} -> Win #{win_idx:03d} ({len(window)} samples)")

                win_idx += 1
                time.sleep(STREAM_INTERVAL_SEC)


if __name__ == "__main__":
    main()
