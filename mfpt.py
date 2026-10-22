"""
    Evaluate model performance and generate diagnostic visualizations.

    Calculates predictions, generates a confusion matrix, and visualizes 
    feature importance. Saves results as high-resolution PNG files 
    in the specified output directory. Designed for non-interactive 
    environments (headless).
"""

# ==============================================================================
# SECTION 1: SYSTEM DEPENDENCIES & CORE SCIENTIFIC LIBRARIES
# ==============================================================================
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kurtosis, skew
from scipy.signal import resample as res
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# ==========================================
# SECTION 2: Configuration & Constants
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"                    # Set your local dataset directory path
DATA_DIR.mkdir(parents=True, exist_ok=True)     # Set your local dataset directory path
TARGET_FS = 48828                               # Standard sampling frequency for alignment (Hz)
WINDOW_SIZE = 4096                 # Samples per window (~0.042 seconds)
STEP_SIZE = 4096 
INNER_VLOAD_MAPPING = {
    '1': 0.0,
    '2': 50.0,
    '3': 100.0,
    '4': 150.0,
    '5': 200.0,
    '6': 250.0,
    '7': 300.0
}

OUTER_VLOAD_MAPPING = {
    '1': 25.0,
    '2': 50.0,
    '3': 100.0,
    '4': 150.0,
    '5': 200.0,
    '6': 250.0,
    '7': 300.0
}

# =========================================
# SECTION 3 : DATA LOADING & PREPROCESSING
# =========================================
def load_single_mat(file_path: str):

    """
    Load a single MFPT .mat file, extract acceleration signal and metadata,
    and resample if necessary to maintain consistent sampling rate.
    """
    mat = sio.loadmat(file_path)
    if 'bearing' not in mat:
        raise KeyError(f"'bearing' struct not found in {file_path}")
        
    b_data = mat['bearing'][0, 0]
    gs = b_data['gs'].flatten()
    sr = int(b_data['sr'][0, 0])
    rate = float(b_data['rate'][0, 0])


    # Extract load: check struct first; if missing, parse from filename
    load = np.nan
    try:
        val = float(b_data['load'][0, 0])
        if not np.isnan(val) and val > 0:
            load = val
    except (ValueError, IndexError, KeyError):
        pass

        # 2. Parse from file name based on condition type
    file_name = os.path.basename(file_path).lower().replace('.mat', '')
    if 'vload_' in file_name:
        load_idx = file_name.split('vload_')[-1]
        if 'inner' in file_name:
            load = INNER_VLOAD_MAPPING.get(load_idx, np.nan)
        elif 'outer' in file_name:
            load = OUTER_VLOAD_MAPPING.get(load_idx, np.nan)
            
    
    #Standardize sampling frequency: Downsample baseline signals from ~97.6kHz to 48.8kHz
    if sr > 60000:
        num_target_samples = int(len(gs) * (TARGET_FS / sr))
        gs = res(gs, num_target_samples)
        sr = TARGET_FS
    return gs, sr, rate, load


# =============================================
# SECTION 4: Preprocessing & Feature Extraction
# =============================================
def preprocess_signal(signal: np.ndarray) -> np.ndarray:

    """
    Remove DC offset (mean shift) caused by static load or sensor bias.
    Ensures vibration dynamics oscillate strictly around zero.
    """
    return signal - np.mean(signal)

def extract_features_from_window(window: np.ndarray) -> dict:
    """
    Compute minimal and highly discriminative time-domain features for a window.
    """
    eps = 1e-12  # Numerical stability constant
    
    # 1. Root Mean Square (RMS) - Total vibration energy
    rms = np.sqrt(np.mean(window ** 2))
    
    # 2. Peak Value - Maximum impact amplitude
    peak = np.max(np.abs(window))
    
    # 3. Crest Factor (CF) - Peak to RMS ratio (early fault indicator)
    crest_factor = peak / (rms + eps)
    
    # 4. Kurtosis - Measure of impulsiveness / tail heaviness (Gaussian = 3.0)
    # fisher=False ensures standard Pearson kurtosis where Normal distribution = 3.0
    kurt = kurtosis(window, fisher=False)
    
    # 5. Skewness - Asymmetry of vibration distribution
    skew_val = skew(window)
    
    # 6. Shape Factor (SF) - Ratio of RMS to Mean Absolute Value
    mean_abs = np.mean(np.abs(window))
    shape_factor = rms / (mean_abs + eps)
    
    return {
        'rms': rms,
        'peak': peak,
        'crest_factor': crest_factor,
        'kurtosis': kurt,
        'skewness': skew_val,
        'shape_factor': shape_factor
    }


# ======================================================================
# SECTION 5: DATA PIPELINE: Dataset Registry & Feature Extraction Engine 
# ======================================================================
def build_dataset_registry(data_dir: str, window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE) -> pd.DataFrame:

    records = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            if file.endswith('.mat'):
                path = os.path.join(root, file)
                name_lower = file.lower()
                if 'baseline' in name_lower or 'normal' in name_lower:
                    label = 0
                    label_name = 'Normal'
                elif 'outer' in name_lower:
                    label = 1
                    label_name = 'Outer_Race'
                elif 'inner' in name_lower:
                    label = 2
                    label_name = 'Inner_Race'
                else:
                    continue  # Skip helper/script files
                
                
                # Load signal and metadata
                signal, fs, shaft_speed, load = load_single_mat(path)
                
                # Step 1: Remove DC component (Shift due to load)
                clean_signal = preprocess_signal(signal)
                
                # Step 2: Segment into windows (Rolling windows with overlap)
                for start in range(0, len(clean_signal) - window_size + 1, step_size):
                    window = clean_signal[start : start + window_size]
                

                    
                    # Step 3: Extract time-domain features
                    feats = extract_features_from_window(window)
                    
                    # Step 4: Append operational parameters & metadata
                    feats.update({
                        'load_lbs': load,
                        "fs": fs, 
                        "duration_sec": len(signal)/fs ,
                        'shaft_speed_hz': shaft_speed,
                        'label': label,
                        'label_name': label_name,
                        'file_name': file
                    })
                    
                    records.append(feats)
                    
    dataset_df = pd.DataFrame(records)
    return dataset_df


# =============================================================================
#  MODEL EVALUATION & DIAGNOSTIC VISUALIZATION ENGINE ---
# =============================================================================

def evaluate_and_plot(model, X_test, y_test, feature_names,class_names):

    """  Evaluates the trained model and saves visualization plots. """
    y_pred = rf_model.predict(X_test)
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Bearing Fault Diagnosis')
    plt.show()


    # 2. Feature Importance Plot
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(y=sorted_importances, x=sorted_features,   hue=sorted_features, palette="viridis", legend=False)
    plt.title('Feature Importance Analysis')
    plt.ylabel('Importance Score')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("[INFO] Scanning and loading MFPT dataset...")
    df = build_dataset_registry(DATA_DIR)

    print("=" * 80)
    print(f"[SUCCESS] Successfully loaded {len(df)} files.")
    print("=" * 80)
    
    # Display dataset summary table
    summary_cols = ['file_name', 'label_name', 'fs', 'load_lbs', 'duration_sec']
    print(df[summary_cols].to_string(index=False))


# ===============================================================
# MODEL TRAINING PIPELINE: Split, Preprocessing & Random Forest
# ===============================================================
    FEATURE_COLS = [
        'rms', 
        'peak', 
        'crest_factor', 
        'kurtosis', 
        'skewness', 
        'shape_factor'
    ]
    TARGET_COL = 'label'
    GROUP_COL = 'file_name'
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    groups = df[GROUP_COL]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_files = set(df.iloc[train_idx][GROUP_COL].unique())
    test_files = set(df.iloc[test_idx][GROUP_COL].unique())
    if train_files.intersection(test_files):
        raise ValueError("eror")
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), 
    columns=FEATURE_COLS, 
    index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), 
    columns=FEATURE_COLS, 
    index=X_test.index
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)



    class_names = ['Normal', 'Outer_Race', 'Inner_Race']

    evaluate_and_plot(
        model=rf_model,
        X_test=X_test_scaled,
        y_test=y_test,
        feature_names=FEATURE_COLS,
        class_names=class_names
    )

    # ===============================================================
    # EXPORT PIPELINE ARTIFACTS (For Real-Time IIoT Inference)
    # ===============================================================

    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_PATH = MODELS_DIR / "pdm_bundle.joblib"

    # Bundle all essential artifacts for the streaming and receiver environment
    pdm_artifacts = {
        "scaler": scaler,
        "model": rf_model,
        "feature_cols": FEATURE_COLS,
        "class_names": class_names
    }

    joblib.dump(pdm_artifacts, MODEL_PATH)
    print(f"\n[INFO] Complete PdM bundle successfully serialized to: {MODEL_PATH}")