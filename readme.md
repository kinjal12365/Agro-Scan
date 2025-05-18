# 🌿 Plant Disease Classification using Transfer Learning

This project is a deep learning-based image classification system that identifies plant leaf diseases in **Potato, Tomato, and Bell Pepper** using **Transfer Learning**. The model classifies images into **Healthy** and **Diseased** categories, aiding early detection and reducing crop loss.

## 🔍 Problem Statement

Plant diseases significantly reduce the yield and quality of crops. Manual identification is time-consuming and error-prone. This project leverages Convolutional Neural Networks (CNNs) and pre-trained models to classify diseases in:
- 🍅 Tomato
- 🥔 Potato
- 🫑 Bell Pepper

---

## 📁 Dataset

- **Source:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets)
- **Classes:**
  - Tomato: Healthy, Early Blight, Late Blight, Mosaic Virus, etc.
  - Potato: Healthy, Early Blight, Late Blight
  - Bell Pepper: Healthy, Bacterial Spot

---

## 🧠 Model Architecture

- **Base Model (Transfer Learning):**
  - `EfficientNetB0` / `MobileNetV2` / `ResNet50` (experimented)
  - Pre-trained on ImageNet
- **Custom Layers:**
  - Global Average Pooling
  - Dense layers with Dropout
  - Output layer with Softmax (Multi-class classification)

---

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- Google Colab / Jupyter Notebook
- Streamlit (optional UI for demo)

---

## 🧪 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

---


