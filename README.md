# Brain Tumor Detection using Deep Learning

This project is a deep learning-based solution to detect brain tumors from MRI images. It utilizes a Convolutional Neural Network (CNN) to classify brain MRI images into four categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**. The solution is built with TensorFlow/Keras for the deep learning model, and Streamlit for creating an interactive web application that allows users to upload images and get real-time predictions.

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Model](#model)
- [Usage](#usage)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Project Structure](#project-structure)

## Dataset

The dataset used in this project is publicly available on Kaggle and contains 3264 MRI images categorized into four types:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

You can download the dataset from [this link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

## Features

- **Deep Learning Model:** Built using TensorFlow/Keras, the model is a CNN designed to classify MRI images into tumor categories.
- **Web Application:** An intuitive interface built with Streamlit allows users to upload brain MRI images and get predictions.
- **Image Preprocessing:** The app handles preprocessing like resizing, rescaling, and image color format conversion before prediction.
- **Real-time Predictions:** The model provides quick and accurate tumor type predictions with confidence scores.

## Installation

### Prerequisites
- Python 3.x
- TensorFlow
- Streamlit
- NumPy
- OpenCV
- PIL (Python Imaging Library)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/braintumor-detection.git
   cd braintumor-detection
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the appropriate directory for training.

## Model

The model used in this project is a **Convolutional Neural Network (CNN)**, built using TensorFlow/Keras. It has been trained on the Brain Tumor MRI dataset and can classify images into the following categories:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**

The model file (`BrainTumorModel.keras`) is pre-trained and included in the repository. You can use it directly to make predictions.

## Usage

You can use the pre-trained model to make predictions using the Streamlit app.

1. To start the web app, run the following command:
   ```bash
   streamlit run app.py
   ```

2. Once the app is running, upload an MRI image (JPG, PNG, or JPEG). The model will classify the image into one of the four categories and provide a confidence score.

   Example prediction:
   ```
   This image most likely belongs to 'Glioma' with a 94.50% confidence.
   ```

## Results

After training, the model achieved an accuracy of over **90%** on the test dataset. Evaluation metrics include **precision**, **recall**, and **F1-score** for each class.

| Class           | Precision | Recall | F1-Score |
|-----------------|-----------|--------|----------|
| Glioma          | 0.91      | 0.92   | 0.91     |
| Meningioma      | 0.90      | 0.89   | 0.89     |
| Pituitary Tumor | 0.92      | 0.93   | 0.92     |
| No Tumor        | 0.93      | 0.94   | 0.93     |

## Future Enhancements

- **Improve Accuracy:** Fine-tune hyperparameters or experiment with more advanced architectures such as **ResNet** or **DenseNet**.
- **Data Augmentation:** Apply more augmentation techniques to improve model robustness.
- **Model Deployment:** Deploy the model to a cloud platform (e.g., AWS, Google Cloud) for real-time usage in production.
- **Mobile Support:** Build a mobile app to increase accessibility.

## Project Structure

```
.
├── app.py                  # Streamlit web app script for image upload and prediction
├── Brain Tumor Detection.ipynb  # Jupyter Notebook for experimentation and model development
├── BrainTumorModel.keras   # Pre-trained Keras model file for predictions
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```
