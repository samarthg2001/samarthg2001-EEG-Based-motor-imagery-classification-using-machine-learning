# EEG-Based Motor Imagery Classification Using Machine Learning

This repository contains the implementation and analysis of various machine learning techniques for EEG-based motor imagery classification. The project is designed to explore how advanced machine learning algorithms can enhance the accuracy and robustness of brain-computer interface (BCI) systems. The primary aim is to classify motor imagery tasks, such as imagining left-hand versus right-hand movements, based on electroencephalography (EEG) signals.

## Abstract

Motor imagery classification is an essential component of brain-computer interfaces (BCIs), particularly for assistive technologies aimed at individuals with motor disabilities. This project leverages machine learning algorithms—Support Vector Machines (SVM), K-Nearest Neighbors (KNN), XGBoost, Logistic Regression, and Naive Bayes—for classifying motor imagery EEG signals. 

A robust preprocessing pipeline, including Power Spectral Density (PSD) analysis, bandpass filtering, and feature extraction using Common Spatial Patterns (CSP), ensures data readiness. Features such as log-variance of filtered EEG signals are computed and fed into classifiers for model training and evaluation. Visualization techniques like scatter plots, decision boundaries, and confusion matrices provide insights into model performance.

---

## Features

- **Signal Preprocessing**: Bandpass filtering and PSD computation focus on specific frequency bands relevant to motor imagery tasks.
- **Feature Extraction**: CSP and log-variance calculations maximize the discriminative power of features.
- **Machine Learning Models**:
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - XGBoost
  - Logistic Regression
  - Naive Bayes
- **Visualization**: Scatter plots, confusion matrices, and decision boundary visualizations.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:
   ```bash
   python app2.py
   ```

4. Open a web browser and navigate to the local server (e.g., `http://127.0.0.1:5000`).

5. Use the web interface to:
   - Upload an EEG dataset.
   - Select the desired machine learning algorithm for classification.
   - View the results, including visualizations and model performance metrics.

---

## Results

- **Support Vector Machines**: Achieved the highest accuracy (97.5%), demonstrating excellent classification boundaries.
- **XGBoost**: Delivered robust performance with 95% accuracy, excelling in handling noisy EEG data.
- **KNN**: Achieved 92.5% accuracy, suitable for smaller datasets due to simplicity.
- **Logistic Regression and Naive Bayes**: Both models showed strong performance with accuracy near 97.5%.

---

## Applications

- **Assistive Technologies**: Control of wheelchairs, prosthetic limbs, and other devices using non-invasive BCIs.
- **Neurofeedback**: Brain training and mental state monitoring.
- **Healthcare**: Rehabilitation and diagnostic support for neurological conditions.
- **Human-Machine Interaction**: Advanced interaction models for robotics and gaming.

---

## Future Enhancements

- **Deep Learning**: Integration of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) for feature extraction and temporal analysis.
- **Real-Time Processing**: Development of pipelines for live classification.
- **Multi-Modal Integration**: Combining EEG with other physiological signals for enhanced performance.
- **User Adaptation**: Adaptive algorithms and transfer learning for personalized BCIs.
- **Cloud Integration**: Cloud-based storage and processing for scalable solutions.

---

## Prerequisites

### Software
- Python 3.x
- Libraries: NumPy, SciPy, Matplotlib, Seaborn, scikit-learn
- Flask for backend development (optional for web-based extensions)

### Hardware
- Minimum: Intel i5, 8GB RAM, 256GB SSD
- Recommended: 16GB RAM and GPU for advanced processing

---

## Acknowledgments

The study is inspired by advancements in BCIs and leverages state-of-the-art machine learning and signal processing techniques to address challenges in motor imagery classification.


---

Let me know if you need further adjustments!
