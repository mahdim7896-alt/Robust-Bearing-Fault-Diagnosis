# Robust Bearing Fault Diagnosis (MFPT Dataset)

A professional-grade machine learning pipeline for vibration-based fault diagnosis, specifically engineered to handle load-variance and prevent data leakage.

## 🚀 The Challenge
Most fault diagnosis models suffer from **Data Leakage** and **Load Dependency**. In this project, we successfully addressed:
*   **Data Leakage Prevention:** Utilizing `GroupShuffleSplit` to ensure that data from the same file never leaks between training and testing sets.
*   **Load-Invariant Features:** Engineering a feature extraction process that maintains accuracy even when the machine's load changes.
*   **Sampling Rate Variance:** Handling mixed sampling rates across the MFPT dataset effectively.

## 📈 Performance
The model achieves an impressive **99.2% Accuracy** on the testing set, verified through rigorous validation methods to ensure the results are robust and not overfitted.

![Confusion Matrix](Figure_1.webp)

## 🛠 Tech Stack
*   **Language:** Python
*   **Libraries:** NumPy, Pandas, Scikit-learn, OpenCV (for signal visualization)
*   **Technique:** Signal Processing, Group-based Cross-Validation, StandardScaler

## 📋 Getting Started
1. Clone the repo: `git clone https://github.com/your-username/Robust-Bearing-Fault-Diagnosis.git`
2. Prepare your MFPT dataset in the `/data` directory.
3. Run the training pipeline: `python main.py`

## 🔮 Roadmap
- [x] MFPT Dataset Integration
- [x] Load-Invariant Model Architecture
- [ ] Synthetic Real-time Signal Generator (In Progress)

---
*Developed by Mahdi - AI/ML Developer.*
