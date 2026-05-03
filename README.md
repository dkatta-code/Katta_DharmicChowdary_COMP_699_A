# 2D to 3D Time Series Classification System

## Developed by Dharmic Katta

---

## Project Overview

This project presents a complete system designed to improve time series classification using deep learning techniques. In many real-world applications such as healthcare, finance, and sensor systems, time series data is commonly used but difficult to analyze using traditional methods. This system addresses that problem by converting one-dimensional time series data into two-dimensional and three-dimensional representations, allowing better pattern recognition and improved classification accuracy. 

---

## Problem Statement

Traditional time series analysis methods work directly on one-dimensional data, which limits the ability to capture complex patterns and relationships. In addition, most existing workflows require multiple tools for preprocessing, training, and evaluation, making the process time-consuming and difficult to manage. This project aims to provide a unified system that simplifies the entire workflow while improving model performance. 

---

## System Objectives

* Convert raw 1D time series data into structured 2D and 3D representations
* Improve classification accuracy using Convolutional Neural Networks
* Provide an integrated workflow for data processing, model training, and evaluation
* Enable interactive visualization of data and results through a simple dashboard
* Ensure the system runs on standard hardware without requiring GPU or paid tools 

---

## Key Features

* Supports both uploaded datasets and publicly available time series data
* Provides preprocessing options such as segmentation and normalization
* Converts 1D data into 2D matrices and then into 3D tensors
* Implements a custom CNN model for classification
* Displays evaluation metrics including accuracy, precision, recall, and F1-score
* Generates visual outputs such as graphs and 3D representations
* Includes role-based access for different types of users 

---

## System Workflow

### 1. Dataset Handling

The system allows users to either upload a dataset or select an existing one. The dataset is validated and prepared for further processing.

### 2. Data Preprocessing

The data is divided into smaller segments and normalized to ensure consistency and improve model performance.

### 3. Data Transformation

The system converts 1D time series data into 2D matrix form to capture spatial patterns. Multiple 2D matrices are then stacked to form 3D tensors to represent temporal changes. 

### 4. Model Configuration

Users can define CNN architecture parameters such as layers, filters, and activation functions. The system ensures configurations remain within valid limits.

### 5. Model Training

The system trains the CNN model using prepared data. Training progress is monitored and controlled to ensure efficient execution.

### 6. Evaluation

The trained model is evaluated using test data. Metrics such as accuracy, confusion matrix, precision, recall, and F1-score are generated.

### 7. Visualization

Results are displayed using graphs and visual tools. Users can also inspect 2D and 3D data representations interactively. 

---

## Technologies Used

* Python
* NumPy and Pandas for data processing
* TensorFlow with Keras API for deep learning model
* Streamlit for building the user interface
* Matplotlib and Plotly for visualization 

---

## System Requirements

* Standard laptop or desktop system
* CPU-based execution (no GPU required)
* Python environment with required libraries installed
* No dependency on paid tools or cloud services 

---

## How to Run the Project

1. Install Python on your system
2. Install required libraries using pip
3. Download or clone the project files
4. Navigate to the project directory
5. Run the application using the command:
   python system.py
6. Open the Streamlit interface in your browser and start using the system

---

## User Roles

* Research User: Handles dataset selection, preprocessing, and synthetic data generation
* Analyst User: Performs transformation, model training, and evaluation
* Administrator User: Manages system settings, constraints, and configurations 

---

## Conclusion

This system provides a structured and efficient way to handle time series classification using modern deep learning techniques. By transforming data into higher-dimensional formats and integrating all steps into a single workflow, it improves both accuracy and usability. The system is simple, flexible, and suitable for academic as well as practical use.
