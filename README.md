# 🧠 Brain MRI Tumor Detection Model using Deep Learning

This project implements a **Convolutional Neural Network (CNN)** with a **pre-trained MobileNetV2** model to detect brain tumors in **MRI scans**. The model performs binary classification to determine whether an MRI image contains a tumor or not. Developed on **macOS (MacBook Air)** and tested with Python 3.12.4.

---

## 📁 Dataset Used
The dataset used is from **Kaggle**:  
[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  
*Credit to the original dataset providers for making this project possible.*

---

## 🛠️ Requirements

### **Version of Python in MacOs for this proyect**

```Bash
% Python3 --version
% Python 3.12.4 
```
---

## 🐍 Python & Modules

### **Install the modules of python**

```Bash
% pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
```
---
## 🚀 Instalation & Setup

### 1. Clone the Repository

```Bash
% git clone https://github.com/BMaster03/Brain_MRI_Tumor_Detection_Model.git
% cd Brain_MRI_Tumor_Detection_Model/
```

### 2. Create a Virtual Environment (Recommended)

```Bash
% python -m venv venv
% source venv/bin/activate  # macOS/Linux
% .\venv\Scripts\activate   # Windows
```
### 3. Install Dependencies 

```Bash
% pip install -r requirements.txt
```
---

## 🧮 Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet, fine-tuned for MRI images).

- **Custom Layers**:  
  - Global Average Pooling  
  - Dropout (for regularization)  
  - Dense Layer (Binary Classification: sigmoid activation)

- **Optimizer**: Adam (adaptive learning rate).

- **Loss Function**: Binary Crossentropy.
---


