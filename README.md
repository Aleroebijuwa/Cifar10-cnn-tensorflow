# Cifar10-cnn-tensorflow
CNN model for CIFAR-10 image classification using TensorFlow
# CIFAR-10 CNN Image Classifier with TensorFlow

A deep learning image classification project built using TensorFlow and Keras.  
This project trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset to classify images into 10 different object categories.

---

# Project Overview

This project demonstrates an end-to-end machine learning workflow including:

- Data preprocessing
- CNN model development
- Model training and evaluation
- Data augmentation
- Hyperparameter tuning
- Performance visualization
- Model export for deployment

The model is trained using the CIFAR-10 dataset, which contains 60,000 color images across 10 classes.

---

# CIFAR-10 Classes

The model classifies images into the following categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Project Structure

cifar10-cnn-tensorflow/
│
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── hyperparameter_tuning.py
│   ├── cifar10_augmentation.py
│   └── final_export.py
│
├── models/
│   └── cifar10_model.keras
│
├── saved_model/
│   └── cifar10_cnn/
│
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── metrics.txt
│
├── requirements.txt
└── README.md
