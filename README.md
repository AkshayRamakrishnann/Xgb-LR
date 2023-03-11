# Hybrid Model using XGBoost Classifier and Linear Regression for Breast Cancer Dataset with Intel oneAPI

This repository contains an example code for a hybrid model using XGBoost classifier and linear regression for breast cancer dataset with Intel oneAPI. The code is written in Python and uses scikit-learn library for machine learning.

## Requirements

Python 3.6 or later
scikit-learn library
pandas library
xgboost library
matplotlib library
seaborn library
Intel oneAPI Base Toolkit

## Dataset
The dataset used in this example is the breast cancer dataset from scikit-learn library. The dataset contains information about breast cancer patients, including various features such as mean radius, mean texture, mean perimeter, etc. The goal is to predict whether a patient has a malignant or benign tumor.

## Models
Two models are used in this example: XGBoost classifier and linear regression. XGBoost is a popular gradient boosting library that is known for its efficiency and accuracy, while linear regression is a simple yet powerful regression algorithm that is widely used in machine learning.

## Hybrid Model
The hybrid model combines the predictions of XGBoost classifier and linear regression to improve the accuracy of the model. The basic idea is to use XGBoost classifier to predict the probability of each class (malignant or benign) and then use linear regression to fit a model to these probabilities. The hybrid model then uses the fitted model to make the final prediction.

## Usage
Clone the repository using git clone https://github.com/example/hybrid-model.git command.
Install the required libraries using pip install -r requirements.txt command.
Install Intel oneAPI Base Toolkit.
Run the code using python hybrid_model.py command.
The code will split the dataset into train and test sets, create an XGBoost classifier and a linear regression model, fit them on the training data, and use them to make predictions on the test data. It will then combine the XGBoost and linear regression predictions to create a hybrid model and calculate the accuracy of the model.

## Results
The results of the hybrid model are visualized using a confusion matrix and a ROC curve. The confusion matrix shows the number of true positives, false positives, true negatives, and false negatives, while the ROC curve shows the trade-off between true positive rate and false positive rate for different classification thresholds.


![download (7)](https://user-images.githubusercontent.com/111365771/224487887-13fa904e-95fd-4046-99a6-a9d375c17d95.png)


## License
This project is licensed under the MIT License - see the LICENSE file for details.



