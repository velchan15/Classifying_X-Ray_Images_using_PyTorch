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
