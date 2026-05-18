# diabetes-classification-ml
CSC-44112 Part 2 - Diabetes prediction using ML
Diabetes Classification – CSC-44112 Part 2
Keele University | Advanced Applications of AI and ML | 2024/2025
Overview
Binary classification of diabetes diagnosis using the Pima Indians Diabetes Dataset. Four ML models are trained, tuned, and evaluated as part of a full end-to-end data science pipeline.
Dataset

Source: Pima Indians Diabetes Database – Kaggle
768 patient records | 8 clinical features | Binary target (Diabetic / Non-Diabetic)

Models & Results
ModelAccuracyAUC-ROCLogistic Regression80.5%0.848SVM79.9%0.840Gradient Boosting77.9%0.826Random Forest75.3%0.819

Best model: Logistic Regression — Top features: Glucose, BMI, Age

How to Run
bashpip install numpy pandas matplotlib seaborn scikit-learn
python diabetes_classification.py
Code Structure
The script is divided into clearly labelled sections matching the report:

Section 3 – Exploratory Data Analysis
Section 4 – Data Preprocessing (zero-imputation, scaling)
Section 5 – Model Training & Hyperparameter Tuning (GridSearchCV, 10-Fold CV)
Section 6 – Evaluation (Accuracy, F1, Precision, Recall, AUC-ROC)
Section 7 – Visualisations (Figures 1–10)
