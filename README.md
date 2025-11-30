# Suicide Risk Detection on Social Media

This project aims to assess suicide risk on social media by prioritizing the ordinal nature of risk levels (Indicator < Ideation < Behavior < Attempt). Addressing the limitations of existing models that ignore real-time gaps between posts, we propose a **Time-Aware Ordinal Model** using the RSD-15K dataset.

## 📌 Project Overview

This project implements four distinct approaches to evaluate the effectiveness of ordinal regression and time-aware modeling in suicide risk assessment.

### The 4 Approaches
1.  **Baseline ML:** XGBoost with TF-IDF and LIWC features.
2.  **Baseline DL:** DeBERTa Transformer trained with standard Cross-Entropy Loss.
3.  **Ordinal Model (SISMO-based):** DeBERTa + BiLSTM + Ordinal Loss (based on Sawhney et al.).
4.  **Time-Aware Ordinal Model:** Enhancing Approach 3 with explicit timestamp features from the RSD-15K dataset.

## 📂 Repository Structure

```text
Suicide-Risk-Detection/
├── data/
│   ├── raw/               # Raw CSV files (e.g., rsd_15k.csv)
│   └── processed/         # Pickled dataframes (train.pkl, test.pkl)
├── notebooks/
│   ├── 00_preprocessing.ipynb     # Data cleaning & Splitting
│   ├── 01_baseline_xgboost.ipynb  # Approach 1
│   ├── 02_baseline_deberta.ipynb  # Approach 2
│   ├── 03_ordinal_sismo.ipynb     # Approach 3
│   └── 04_time_aware_model.ipynb  # Approach 4
├── src/
│   ├── __init__.py        # Initialization for src package
│   ├── utils.py           # Evaluation metrics (Graded Precision/Recall)
│   └── loss.py            # Custom Ordinal Loss implementation
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🚀 Getting Started
1. **Prerequisites**
   - **Python 3.11** (Recommended for best performance on Apple Silicon)
   - **PyTorch 2.2+** (Required for MPS acceleration on Mac)
   - Transformers
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Data Preparation**
   - Place the ```rsd_15k.csv``` file into the ```data/raw/``` directory.
   - Run ```notebooks/00_preprocessing.ipynb``` to generate train/test splits

## 📈 Evaluation Metrics
Since standard accuracy is insufficient for ordinal risk levels, we utilize Graded Metrics:
- **Graded Precision (GP):** Penalizes overestimation of risk.
- **Graded Recall (GR):** Penalizes underestimation of risk.
- **Graded F1-Score:** The harmonic mean of GP and GR.

These metrics are implemented in ```src/utils.py``` to ensure consistency across all approaches.

## 📜 References
- Sawhney et al., "Towards Ordinal Suicide Ideation Detection on Social Media", WSDM '21.
- Zheng et al., "RSD-15K: A Large-Scale User-Level Annotated Dataset...", arXiv 2025.