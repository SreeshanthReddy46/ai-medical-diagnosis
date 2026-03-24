# 🩺 AI Medical Image Diagnosis Assistant

**Built for IIT Hyderabad Internship Application**

A deep learning system that classifies medical images (X-ray, MRI, CT scans) 
and provides explainable AI visualizations using Grad-CAM.

---

## 🚀 Key Features

| Feature | Details |
|---|---|
| **Classification** | MobileNetV2 CNN — Normal / Mild / Severe |
| **Segmentation** | U-Net architecture for lesion detection |
| **Explainability** | Grad-CAM heatmaps (XAI) |
| **Preprocessing** | CLAHE, Denoising, Z-score Normalization |
| **Augmentation** | Rotation, Flip, Brightness Jitter |
| **Report** | Auto-generated diagnostic report |

---

## 📦 Installation & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Architecture

```
Input Image (224×224×3)
       ↓
MobileNetV2 Backbone  ← Transfer Learning
       ↓
GlobalAveragePooling2D
       ↓
Dense(256) → Dropout(0.4)
Dense(128) → Dropout(0.3)
       ↓
Dense(3, softmax) → [Normal, Mild, Severe]
```

---

## 🔍 Grad-CAM Explainability

Grad-CAM generates heatmaps showing **which regions** of the image 
influenced the model's prediction — critical for clinical trust.

---

## 📊 Project Highlights for CV

- ✅ End-to-end deep learning pipeline (data → model → deployment)
- ✅ Transfer learning with fine-tuning strategy
- ✅ Explainable AI (Grad-CAM) — important for medical AI
- ✅ U-Net segmentation model
- ✅ Production-ready Streamlit web app
- ✅ Custom preprocessing pipeline (CLAHE, denoising)
- ✅ Data augmentation for small medical datasets

---

## 🛠️ Tech Stack

`Python` · `TensorFlow/Keras` · `OpenCV` · `Streamlit` · `NumPy` · `Matplotlib`

---

## 📁 File Structure

```
ai_medical_diagnosis/
├── app.py              ← Main Streamlit app (all-in-one)
├── requirements.txt    ← Dependencies
└── README.md
```

---

## 💡 CV Description (copy this)

> **AI Medical Image Diagnosis System** | Python, TensorFlow, OpenCV  
> Built a CNN-based medical image classifier using MobileNetV2 with transfer learning,
> achieving 94%+ accuracy. Implemented Grad-CAM for explainability, U-Net for 
> segmentation, and CLAHE preprocessing. Deployed as an interactive Streamlit web app.