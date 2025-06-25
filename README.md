# 🧬 Breast Cancer Detection using Logistic Regression

This project is a Machine Learning model built using Logistic Regression to predict whether a breast tumor is malignant or benign based on input features extracted from digitized images of fine needle aspirate (FNA) of breast mass.

## 📂 Dataset

The dataset used is the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, which contains 569 samples of tumors with 30 numeric features each, computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

- ✅ Total Samples: 569  
- ✅ Features: 30  
- ✅ Labels:  
  - `M` → Malignant (encoded as `1`)  
  - `B` → Benign (encoded as `0`)  

## 📌 Project Structure

```text
├── breast_cancer_detection.py  # Main model and prediction code
├── Dataset/
│   ├── wdbc.data               # Raw data file
│   └── wdbc.names              # Description of features
└── README.md                   # Project documentation
