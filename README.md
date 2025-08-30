💳 Credit Card Default Prediction App

App link: https://creditcardapp1.streamlit.app/

Overview

This project implements a Credit Card Default Prediction App using AdaBoost Classifier. The app predicts whether a customer will default on their credit card payment next month based on 8 key features.

The app is built using Python, Scikit-learn, and Streamlit, providing an interactive interface to input customer details and get predictions instantly.

Features Implemented

The following 8 features are used for prediction:

Feature	Description
PAY_0	Payment status for the current month
PAY_AMT2	Payment made two months ago
LIMIT_BAL	Credit limit of the customer
PAY_2	Payment status two months ago
PAY_AMT3	Payment made three months ago
BILL_AMT1	Bill amount for the current month
PAY_AMT1	Payment made one month ago
MARRIAGE	Marital status (1=Married, 2=Single, 3=Others)
What I Have Done

Data Preparation

Loaded the dataset and renamed the target column default.payment.next.month to default.

Selected only the 8 important features for prediction.

Applied StandardScaler to scale the features for better model performance.

Model Training

Trained an AdaBoost Classifier using a Decision Tree (max depth=3) as the base estimator.

Used GridSearchCV with StratifiedKFold (5 splits) to find the best parameters (n_estimators and learning_rate).

Evaluated the model using:

Accuracy

Confusion Matrix

Classification Report

ROC-AUC score

Visualization

Plotted Confusion Matrix using Seaborn.

Plotted ROC Curve for model performance visualization.

Prediction

Created a sample input (dictionary / array) for testing predictions.

Scaled the input using the same scaler as training.

Predicted Default / No Default and Probability of Default using the trained model.

Streamlit App

Built a Streamlit interface to input customer details.

Converted Marital Status to numeric values for prediction.

Displayed prediction results and probability interactively.

How to Run

Install required packages:

pip install -r requirements.txt


(Packages: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, joblib)

Run the app:

streamlit run app.py


Enter the customer details in the fields and click Predict.

Sample Input
PAY_0: 0
PAY_AMT2: 5000
LIMIT_BAL: 50000
PAY_2: 0
PAY_AMT3: 5000
BILL_AMT1: 10000
PAY_AMT1: 5000
MARRIAGE: Married


Sample Output:

Prediction → No Default
Probability of Default: 0.15
