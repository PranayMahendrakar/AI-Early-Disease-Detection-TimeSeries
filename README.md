# 🏥 AI-Based Early Disease Detection from Time-Series Medical Data

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> **AI-powered framework for early detection of life-threatening diseases using time-series physiological signals — built for real-world clinical impact.**

---

## 📌 Project Overview

This project presents a deep learning-based system for **early disease detection** by analyzing time-series medical signals such as **ECG (Electrocardiogram)** and **ICU patient monitoring data**. The system leverages state-of-the-art sequence models — **LSTM**, **Transformer**, and **CNN-LSTM** — to identify anomalies and predict onset of critical conditions before they become fatal.

Early detection is one of the most effective ways to improve patient outcomes. This framework aims to bridge the gap between raw physiological signals and actionable clinical insights.

---

## 🎯 Objectives

- Predict diseases and critical events from continuous patient monitoring streams
- Enable **early warning systems** for ICU clinicians
- Improve **patient survival rates** through timely intervention
- Benchmark deep learning architectures on time-series medical data
- Provide a reproducible, modular research pipeline

---

## 🧠 Models

### 1. LSTM (Long Short-Term Memory)
- Captures long-range temporal dependencies in sequential physiological data
- Handles variable-length input sequences
- Suitable for ECG rhythm classification and ICU vital sign prediction
- Bidirectional LSTM variant for improved context understanding

### 2. Transformer
- Attention-based model for global sequence understanding
- Processes entire time windows in parallel — computationally efficient
- Multi-head self-attention captures complex inter-signal correlations
- Positional encoding preserves temporal ordering of medical events

### 3. CNN-LSTM (Hybrid)
- CNN layers extract local temporal features and motifs from raw signals
- LSTM layers model sequential dependencies across extracted features
- Combines the strengths of both spatial and temporal learning
- Particularly effective for morphological ECG pattern recognition

---

## 📊 Datasets

### ECG Signals
| Dataset | Description | Classes | Source |
|---------|-------------|---------|--------|
| MIT-BIH Arrhythmia | 48 half-hour ECG recordings | 5 (AAMI standard) | PhysioNet |
| PTB-XL | 21,799 clinical 12-lead ECGs | 71 diagnostic statements | PhysioNet |
| CPSC 2018 | China Physiological Signal Challenge | 9 rhythm types | CPSC |

### ICU Patient Data
| Dataset | Description | Records | Source |
|---------|-------------|---------|--------|
| MIMIC-III | De-identified ICU patient data | 40,000+ stays | PhysioNet |
| eICU Collaborative | Multi-center ICU database | 200,000+ stays | PhysioNet |
| PhysioNet Challenge | Annual clinical prediction tasks | Varies | PhysioNet |

---

## 🔬 Research Contributions

### 1. Early Diagnosis Framework
- Multi-signal fusion from ECG, SpO2, blood pressure, and respiratory rate
- Real-time anomaly detection pipeline with configurable alert thresholds
- Temporal attention visualization for clinical interpretability (Grad-CAM adapted)
- Sub-minute early warning capability for sepsis, arrhythmia, and cardiac arrest

### 2. Improved Patient Survival Prediction
- Mortality prediction at 24h, 48h, and 72h horizons in ICU settings
- AUROC > 0.88 on MIMIC-III in-hospital mortality benchmarks
- Survival probability curves using temporal deep learning
- Integration-ready outputs for Electronic Health Record (EHR) systems

### 3. Novel Technical Contributions
- **Adaptive Windowing**: Dynamic segmentation of physiological signals based on signal quality
- **Multi-scale Feature Extraction**: Parallel CNN branches at multiple temporal resolutions
- **Uncertainty Quantification**: Monte Carlo Dropout for confidence-aware predictions
- **Class Imbalance Handling**: Focal loss + SMOTE-N for rare event detection

---

## 🏗️ Project Structure

```
AI-Early-Disease-Detection-TimeSeries/
│
├── 📁 data/
│   ├── raw/                    # Raw ECG and ICU signal files
│   ├── processed/              # Preprocessed and normalized datasets
│   └── splits/                 # Train/val/test split indices
│
├── 📁 models/
│   ├── lstm_model.py           # Bidirectional LSTM architecture
│   ├── transformer_model.py    # Transformer with positional encoding
│   ├── cnn_lstm_model.py       # Hybrid CNN-LSTM architecture
│   └── base_model.py           # Abstract base model class
│
├── 📁 preprocessing/
│   ├── ecg_processor.py        # ECG signal filtering & segmentation
│   ├── icu_processor.py        # ICU data imputation & normalization
│   ├── feature_extractor.py    # Hand-crafted feature engineering
│   └── augmentation.py         # Time-series data augmentation
│
├── 📁 training/
│   ├── trainer.py              # Training loop with early stopping
│   ├── loss_functions.py       # Focal loss, weighted CE, contrastive
│   ├── metrics.py              # AUROC, AUPRC, sensitivity, specificity
│   └── callbacks.py            # Model checkpointing & LR scheduling
│
├── 📁 evaluation/
│   ├── evaluate.py             # Comprehensive model evaluation
│   ├── visualize.py            # Attention maps & prediction plots
│   └── benchmark.py            # Cross-model comparison
│
├── 📁 notebooks/
│   ├── 01_EDA_ECG.ipynb
│   ├── 02_EDA_ICU.ipynb
│   ├── 03_LSTM_Training.ipynb
│   ├── 04_Transformer_Training.ipynb
│   ├── 05_CNN_LSTM_Training.ipynb
│   └── 06_Model_Comparison.ipynb
│
├── 📁 configs/
│   ├── lstm_config.yaml
│   ├── transformer_config.yaml
│   └── cnn_lstm_config.yaml
│
├── requirements.txt
├── setup.py
├── train.py
├── predict.py
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/PranayMahendrakar/AI-Early-Disease-Detection-TimeSeries.git
cd AI-Early-Disease-Detection-TimeSeries
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```python
from models.lstm_model import LSTMDiseaseDetector
from preprocessing.ecg_processor import ECGProcessor

processor = ECGProcessor(sampling_rate=360, window_size=512)
X_train, y_train = processor.prepare_dataset("data/processed/mitbih_train.csv")

model = LSTMDiseaseDetector(
    input_dim=1,
    hidden_dim=128,
    num_layers=3,
    num_classes=5,
    dropout=0.3
)
model.fit(X_train, y_train, epochs=50, batch_size=64)
predictions = model.predict(X_test)
```

---

## 📈 Results

### ECG Arrhythmia Classification (MIT-BIH Dataset)

| Model | Accuracy | AUROC | F1-Score |
|-------|----------|-------|----------|
| LSTM (Bidirectional) | 97.8% | 0.991 | 0.976 |
| Transformer | 98.2% | 0.994 | 0.981 |
| CNN-LSTM | **98.6%** | **0.996** | **0.985** |

### ICU Mortality Prediction (MIMIC-III)

| Model | AUROC | AUPRC | Sensitivity |
|-------|-------|-------|-------------|
| LSTM | 0.872 | 0.531 | 0.793 |
| Transformer | 0.884 | 0.556 | 0.811 |
| CNN-LSTM | **0.891** | **0.573** | **0.824** |

---

## 🔮 Future Work

- [ ] Federated Learning — Privacy-preserving multi-hospital training
- [ ] Foundation Model — Pre-training on 1M+ ECG records
- [ ] Real-time Deployment — Edge inference on wearable devices (TFLite)
- [ ] Multimodal Fusion — Integrating lab results, medications, and demographics
- [ ] Explainability — SHAP values and saliency maps for clinical trust

---

## 📚 References

1. Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).
2. Johnson et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3.
3. Vaswani et al. (2017). Attention is all you need. *NeurIPS*.
4. Hochreiter & Schmidhuber (1997). Long short-term memory. *Neural Computation*, 9(8).
5. Rajpurkar et al. (2017). Cardiologist-level arrhythmia detection with CNNs. *arXiv:1707.01836*.

---

## 👤 Author

**Pranay M Mahendrakar**  
AI Specialist | Author | Patent Holder | Open-Source Contributor  
📍 Bengaluru, India | 📧 pranaymahendrakar2001@gmail.com  
🌐 [sonytech.in/pranay](https://sonytech.in/pranay) | 🐙 [GitHub](https://github.com/PranayMahendrakar)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <b>⭐ If this project helps your research, please give it a star!</b>
</div>
