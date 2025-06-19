# Predicting Heart Disease Using Machine Learning

This project aims to build a machine learning model capable of predicting whether a patient has heart disease based on various clinical parameters. The dataset used in this project comes from the Cleveland Heart Disease dataset available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease) and [Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci).

The goal is to achieve at least **95% accuracy** in predicting heart disease during the proof of concept.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Tools and Libraries](#tools-and-libraries)
- [Models Used](#models-used)
- [Results](#results)
- [How to Use](#how-to-use)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project explores various machine learning techniques to predict heart disease. It follows a systematic approach: from understanding the data, feature analysis, modeling, evaluation, to optimization.

---

## Dataset

The dataset contains 303 patient records with 14 clinical features:

- **age**: Age of the patient (years)
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (4 values)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (values 0,1,2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (values 1,2,3)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
- **target**: Presence of heart disease (1 = yes, 0 = no)

---

## Approach

1. **Problem Definition**: Predict heart disease using clinical parameters.
2. **Data Exploration**: Analyze dataset to understand structure and feature distributions.
3. **Feature Analysis**: Assess feature importance and relationships.
4. **Modeling**: Implement and evaluate multiple ML algorithms.
5. **Experimentation**: Tune models for optimal performance.
6. **Evaluation**: Use accuracy, precision, recall, and F1-score as metrics.

---

## Tools and Libraries

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning models and evaluation
- **Jupyter Notebook**: Interactive development environment

---

## Models Used

- **Random Forest Classifier**
- **Logistic Regression**
- **K-Nearest Neighbors (KNN) Classifier**

---

## Results

Models are evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

The goal is to identify the model that best predicts heart disease, with a target of at least 95% accuracy.

---

## How to Use

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Amogh-7pai/Heart-disease-project.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd Heart-disease-project
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset:**
    - If the data is not included, download it from [Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci) or [UCI](https://archive.ics.uci.edu/ml/datasets/heart+Disease) and place it in the project folder.

5. **Open the Jupyter Notebook to explore the project:**
    ```bash
    jupyter notebook Project1.ipynb
    ```

---

## Future Work

- Experiment with more machine learning models (e.g., SVM, XGBoost).
- Perform hyperparameter tuning for better accuracy.
- Deploy the model as a web application for real-time predictions.
- Integrate with healthcare platforms for wider accessibility.

---

## License

This project is open-source and available under the [MIT License](LICENSE).  
Feel free to use, modify, and distribute it as needed.

---

## Contact

For questions or feedback, please reach out to the project maintainer via [GitHub Issues](https://github.com/Amogh-7pai/Heart-disease-project/issues).
