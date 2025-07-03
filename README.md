# Elevate-labs-Task-7

SVM Breast Cancer Classification
Overview
This project implements a Support Vector Machine (SVM) classifier to predict breast cancer diagnosis (Malignant or Benign) using the Breast Cancer Wisconsin dataset. The script performs binary classification, trains SVM models with linear and RBF kernels, visualizes decision boundaries, tunes hyperparameters, and evaluates performance using cross-validation.
Prerequisites

Python 3.6+
Dependencies (install via pip):pip install pandas numpy scikit-learn matplotlib seaborn


Dataset: breast-cancer.csv (must be in the same directory as the script)
Contains features like radius_mean, texture_mean, and the target diagnosis (M for Malignant, B for Benign).



Installation

Clone or download this repository.
Ensure breast-cancer.csv is in the project directory.
Install dependencies:pip install -r requirements.txt

Or manually install the required libraries listed above.

Usage

Place breast-cancer.csv in the same directory as svm_breast_cancer.py.
Run the script in a Python environment (e.g., Jupyter Notebook, terminal):python svm_breast_cancer.py


Outputs:
Console: Accuracy scores, 5-fold cross-validation results, and best RBF SVM parameters.
Plots:
Confusion matrix for the best RBF SVM model.
Decision boundary visualizations for linear and best RBF SVM models.





File Structure

svm_breast_cancer.py: Main Python script implementing SVM classification.
breast-cancer.csv: Dataset file (not included in this repository; user must provide).
README.md: This documentation file.

Script Details
The script performs the following tasks:

Data Preparation:

Loads breast-cancer.csv using pandas.
Selects radius_mean and texture_mean for 2D visualization.
Encodes diagnosis as binary (Malignant: 1, Benign: 0).
Splits data into 80% training and 20% testing sets.
Normalizes features using StandardScaler.


SVM Training:

Trains two SVM models using scikit-learn's SVC:
Linear kernel (kernel='linear').
RBF kernel (kernel='rbf').




Decision Boundary Visualization:

Plots decision boundaries for both models using a 2D mesh grid.
Displays training points (Benign: red, Malignant: green) on original scale (cm for radius, texture units).
Uses light colors (#FFAAAA, #AAFFAA) for boundaries and bold colors (#FF0000, #00FF00) for points.


Hyperparameter Tuning:

Uses GridSearchCV to tune RBF kernel parameters:
C: [0.1, 1, 10, 100]
gamma: ['scale', 'auto', 0.01, 0.1, 1]


Selects the best model based on 5-fold cross-validation.


Performance Evaluation:

Computes test accuracy for both models.
Performs 5-fold cross-validation on training data.
Displays a confusion matrix for the best RBF SVM model.



Results
Example output (based on the provided execution):

Linear SVM Accuracy: 0.904
Linear SVM 5-Fold CV Accuracy: 0.884 (±0.028)
RBF SVM Accuracy: 0.921
RBF SVM 5-Fold CV Accuracy: 0.899 (±0.036)
Best RBF SVM Parameters: {'C': 10, 'gamma': 0.1}
Best RBF SVM Accuracy: 0.912
Confusion Matrix: [[66, 5], [5, 38]] (True Positives: 38, True Negatives: 66, False Positives: 5, False Negatives: 5)

Notes

Ensure breast-cancer.csv is correctly formatted and present in the working directory.
For Jupyter Notebook, add %matplotlib inline to display plots inline.
To save plots, modify the script to use plt.savefig('plot_name.png').
The script uses a small step size (h=0.02) for detailed decision boundary plots, which may be computationally intensive. Adjust h (e.g., to 0.1) for faster rendering.
