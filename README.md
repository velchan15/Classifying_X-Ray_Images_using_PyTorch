# Chest X-Ray Pneumonia Classification using ResNet-18

A Deep Learning project designed to detect Pneumonia from chest X-ray images using **Transfer Learning** with the **ResNet-18** architecture in PyTorch.

---

## 📌 Project Overview

Pneumonia is one of the leading respiratory illnesses worldwide. Timely and accurate diagnosis is critical for effective treatment. This project leverages a pre-trained Convolutional Neural Network (CNN) to assist medical experts by classifying chest X-ray images into two distinct categories:

- **NORMAL**: Healthy lungs
- **PNEUMONIA**: Lungs affected by pneumonia

By fine-tuning a pre-trained **ResNet-18** model, we achieve high classification performance while significantly reducing training time and computational resource requirements.

---

## 📁 Repository Structure

```text
.
├── data/                  # Contains raw/extracted X-ray image datasets
│   └── chestxrays/        # Train and test folders (NORMAL & PNEUMONIA)
├── notebooks/             # Contains Jupyter Notebooks for training & evaluation
│   └── notebook.ipynb
└── requirements.txt       # Project dependencies

```

---

## 📊 Dataset Description

The dataset consists of preprocessed chest X-ray images divided into training and testing sets:

* **Training Set:** 300 images (150 NORMAL, 150 PNEUMONIA)
* **Test Set:** 100 images (50 NORMAL, 50 PNEUMONIA)

---

## 🛠️ Model Architecture & Preprocessing

1. **Image Preprocessing & Transformation:**
* Images are converted to tensors and normalized using standard ImageNet parameters:
* `mean = [0.485, 0.456, 0.406]`
* `std = [0.229, 0.224, 0.225]`



2. **Transfer Learning (ResNet-18):**
* Utilizes pre-trained weights (`ResNet18_Weights.DEFAULT`).
* Freezes early convolutional layers to retain general visual feature representations.
* Replaces the final Fully Connected (FC) layer to perform classification on the binary target.


3. **Training & Optimization:**
* Optimizer: `Adam` (learning rate = `0.001`)
* Loss Function: `BCEWithLogitsLoss` / `CrossEntropyLoss`


4. **Evaluation:**
* Model accuracy and performance are evaluated using **Accuracy** and **F1-Score**.

---

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/velchan15/Classifying_X-Ray_Images_using_PyTorch
cd Classifying_X-Ray_Images_using_PyTorch

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```
*(Or install core packages manually: `pip install torch torchvision torchmetrics numpy`)*

3. **Execute the notebook:**
Open and run `notebooks/notebook.ipynb` sequentially to execute dataset extraction, model training, and performance evaluation.
