import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sklearn
import xgboost as xgb
import seaborn as sns

strokeData = pd.read_csv('necessaryfiles/healthcare-dataset-stroke-data.csv')

####################### Data Exploration ######################

print(strokeData.head())
shape = strokeData.shape
print(f"Shape: {shape}")

strokeData = strokeData.dropna()
numDuplicates = strokeData.duplicated().sum()
print(numDuplicates) 

strokeData['gender'] = strokeData['gender'].replace({'Male': 0, 'Female': 1, 'Other': 2})
strokeData['ever_married'] = strokeData['ever_married'].replace({'No': 0, 'Yes': 1})
strokeData['work_type'] = strokeData['work_type'].replace({'Private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3, 'Never_worked': 4})
strokeData['Residence_type'] = strokeData['Residence_type'].replace({'Urban': 1, 'Rural': 0})
strokeData['smoking_status'] = strokeData['smoking_status'].replace({'never smoked' : 1, 'formerly smoked': 0, 'smokes': 2, 'Unknown': 3})

strokeData = strokeData.set_index('id')

summary = strokeData.describe()
print(summary)

medianValues = strokeData.median()
# print(medianValues)

modeValues = strokeData.mode()
# print(modeValues)

countStroke = strokeData['stroke'].value_counts()
print(f"Number of people who had a stroke: {countStroke[1]}\nNumber of people who did NOT have a stroke: {countStroke[0]}")

strokeData.nunique()

axes = strokeData.hist(bins=15, figsize=(15, 10), edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=16)

for ax in axes.flatten():
    ax.grid(False)

plt.tight_layout()
plt.show()

strokeData.groupby('stroke').mean()

correlationMatrix = strokeData.corr()
print(correlationMatrix['stroke'].sort_values(ascending=False))

####################### Data Preparation #######################

X = strokeData.drop(columns=['stroke'])
y = strokeData['stroke']

print(X.shape)
print(y.shape)

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=3)

def computeMeanStd(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof = 1)
    return mean, std

def standardize(X, mean, std):
    XStd = (X - mean) / std
    return XStd   

meanTrain, stdTrain = computeMeanStd(XTrain)

XTrainScaled = standardize(XTrain, meanTrain, stdTrain)
XTestScaled = standardize(XTest, meanTrain, stdTrain)

####################### Feature Selection ##########################

from sklearn.linear_model import Lasso

alpha = 0.01 # Higher alpha means more SPARSITY
lasso = Lasso(alpha=alpha)

lasso.fit(XTrainScaled, yTrain)

print("Coefficients of the Lasso Regression Model:", lasso.coef_)

covarianceMatrix = np.cov(XTrainScaled, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)

loadingFactors = eigenvectors * np.sqrt(eigenvalues)
print("Loading Factors of the PCA Model:", loadingFactors)

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(loadingFactors, annot=True, cmap='coolwarm')
plt.xlabel("Principal Component")
plt.ylabel("Original Feature")
plt.show()

import statsmodels.api as sm

# Add intercept to XTrainScaled for stats models
XTrainScaledWithIntercept = sm.add_constant(XTrainScaled)

# Fit the logistic regression model using statsmodels
logit_model = sm.Logit(yTrain, XTrainScaledWithIntercept)
result = logit_model.fit(disp=False)

p_values = result.pvalues
coefficients = result.params

for i in range(len(p_values)):
    if p_values[i] < 0.05:
        print(f"Feature {i} is statistically significant with a p-value of {p_values[i]}")

from scipy.stats import chi2_contingency

categoricalFeatures = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

XTrainScaledDf = pd.DataFrame(XTrainScaled, columns = strokeData.drop(columns=['stroke']).columns)
yTrainDf = pd.DataFrame(yTrain, columns=['stroke'])

for feature in categoricalFeatures:
    # Create a contingency table
    contingencyTable = pd.crosstab(XTrainScaledDf[feature], yTrainDf['stroke'])
    
    # Perform the Chi-Squared test
    chi2, p, dof, expected = chi2_contingency(contingencyTable)
    
    # Print the results
    print(f"Chi-Squared Test for {feature}: p-value {p}")
    if p < 0.05:
        print(f"{feature} is statistically significant")
    else:
        print(f"{feature} is not statistically significant")

XTrainScaledReduced = XTrainScaledDf[['age', 'hypertension', 'heart_disease', 'avg_glucose_level']]
XTestScaledDf = pd.DataFrame(XTestScaled, columns=strokeData.drop(columns=['stroke']).columns)
XTestScaledReduced = XTestScaledDf[['age', 'hypertension', 'heart_disease', 'avg_glucose_level']]

####################### Model Implementation ##########################

####################### XGBoost Classifier ##########################
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

xgb = XGBClassifier(
    eval_metric='logloss', 
    scale_pos_weight=100, # Increased for imbalance between stroke and no stroke
    max_depth=6, # Deeper trees to capture complex patterns
    learning_rate=0.01, # Slower learning rate for better convergence
    n_estimators=400, # More trees for better performance
    subsample=0.9 
)

xgb.fit(XTrainScaled, yTrain)

yPredProba = xgb.predict_proba(XTestScaled)[:, 1]

yPred = (yPredProba >= 0.1).astype(int) # We want to flag for a doc even if there is a 10% chance of a stroke

xgbAccuracy = sklearn.metrics.accuracy_score(yTest, yPred)
xgbPrecision = sklearn.metrics.precision_score(yTest, yPred)
xgbRecall = sklearn.metrics.recall_score(yTest, yPred)
xgbF1 = sklearn.metrics.f1_score(yTest, yPred)

print(f"XGB Accuracy: {xgbAccuracy:.4f}")
print(f"XGB Precision: {xgbPrecision:.4f}")
print(f"XGB Recall: {xgbRecall:.4f}")
print(f"XGB F1 Score: {xgbF1:.4f}")

print("XGBoost Classification Report:")
print(classification_report(yTest, yPred))

xgbCm = confusion_matrix(yTest, yPred)
disp = ConfusionMatrixDisplay(confusion_matrix=xgbCm, display_labels=xgb.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('XGBoost Confusion Matrix (Threshold=0.15)')
plt.show()

####################### XGB k-Fold Cross-Validation ##########################
from sklearn.model_selection import KFold
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

xgbAccuracys = []
xgbPrecisions = []
xgbRecalls = []
xgbF1s = []

threshold = 0.15

# k-Fold Cross-Validation
for trainIndex, testIndex in kf.split(XTrainScaled):

    XTrainFold, XTestFold = XTrainScaled[trainIndex], XTrainScaled[testIndex]
    yTrainFold, yTestFold = yTrain[trainIndex], yTrain[testIndex]
    
    xgb.fit(XTrainFold, yTrainFold)
    
    yPredProbaFold = xgb.predict_proba(XTestFold)[:, 1]
    yPredFold = (yPredProbaFold >= threshold).astype(int)
    
    xgbAccuracys.append(sklearn.metrics.accuracy_score(yTestFold, yPredFold))
    xgbPrecisions.append(sklearn.metrics.precision_score(yTestFold, yPredFold))
    xgbRecalls.append(sklearn.metrics.recall_score(yTestFold, yPredFold))
    xgbF1s.append(sklearn.metrics.f1_score(yTestFold, yPredFold))

print(f"XGB {k}-Fold CV AVERAGE Accuracy: {np.mean(xgbAccuracys):.4f}")
print(f"XGB {k}-Fold CV AVERAGE Precision: {np.mean(xgbPrecisions):.4f}")
print(f"XGB {k}-Fold CV AVERAGE Recall: {np.mean(xgbRecalls):.4f}")
print(f"XGB {k}-Fold CV AVERAGE F1 Score: {np.mean(xgbF1s):.4f}")


####################### XGBoost Classifier with Reduced Features ##########################
xgb = XGBClassifier(
    eval_metric='logloss', 
    scale_pos_weight=100, # Increased for imbalance between stroke and no stroke
    max_depth=6, # Deeper trees to capture complex patterns
    learning_rate=0.01, # Slower learning rate for better convergence
    n_estimators=400, # More trees for better performance
    subsample=0.9 
)

xgb.fit(XTrainScaledReduced, yTrain)

yPredProbaReduced = xgb.predict_proba(XTestScaledReduced)[:, 1]

yPredReduced = (yPredProbaReduced >= 0.1).astype(int) # We want to flag for a doc even if there is a 10% chance of a stroke

xgbAccuracy = sklearn.metrics.accuracy_score(yTest, yPredReduced)
xgbPrecision = sklearn.metrics.precision_score(yTest, yPredReduced)
xgbRecall = sklearn.metrics.recall_score(yTest, yPredReduced)
xgbF1 = sklearn.metrics.f1_score(yTest, yPredReduced)

print(f"XGB Accuracy: {xgbAccuracy:.4f}")
print(f"XGB Precision: {xgbPrecision:.4f}")
print(f"XGB Recall: {xgbRecall:.4f}")
print(f"XGB F1 Score: {xgbF1:.4f}")


print("XGBoost Classification Report:")
print(classification_report(yTest, yPredReduced))

xgbCm = confusion_matrix(yTest, yPredReduced)
disp = ConfusionMatrixDisplay(confusion_matrix=xgbCm, display_labels=xgb.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('XGBoost Confusion Matrix (Threshold=0.15)')
plt.show()


