# Adapted from: https://www.kaggle.com/pavansanagapati/anomaly-detection-credit-card-fraud-analysis

import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams

# import plotly.plotly as py
# import plotly.graph_objs as go
# import plotly
# import plotly.figure_factory as ff

rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
filepath = "../data/creditcard.csv"



def plot_amount(df):

    # fraud = df[df['Class'] == 1]
    # normal = df[df['Class'] == 0]

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle('Amount per transaction by class')
    bins = 50

    ax1.hist(fraud.Amount, bins = bins)
    ax1.set_title('Fraud')

    ax2.hist(normal.Amount, bins = bins)
    ax2.set_title('Normal')

    plt.xlabel('Amount ($)')
    plt.ylabel('Number of Transactions')
    plt.xlim((0, 20000))
    plt.yscale('log')
    return plt

def plot_times(df):

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle('Time of transaction vs Amount by class')
    ax1.scatter(fraud.Time, fraud.Amount)
    ax1.set_title('Fraud')
    ax2.scatter(normal.Time, normal.Amount)
    ax2.set_title('Normal')
    plt.xlabel('Time (in Seconds)')
    plt.ylabel('Amount')
    return plt

def run_models(clf_dict, X, y, outlier_fraction): # Start here, do train_test_split

     
    # n_outliers = len(Fraud)
    for i, (clf_name, clf) in enumerate(clf_dict.items()):
        #Fit the data and tag outliers
        if clf_name == "Local Outlier Factor":
            y_pred = clf.fit_predict(X)
            scores_prediction = clf.negative_outlier_factor_
        elif clf_name == "Support Vector Machine":
            clf.fit(X)
            y_pred = clf.predict(X)
        else:    
            clf.fit(X)
            scores_prediction = clf.decision_function(X)
            y_pred = clf.predict(X)
        #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        n_errors = (y_pred != y).sum()
        # Run Classification Metrics
        print("{}: {}".format(clf_name,n_errors))
        print("Accuracy Score :")
        print(accuracy_score(y,y_pred))
        print("Classification Report :")
        print(classification_report(y,y_pred))

if __name__ == "__main__":

    data = pd.read_csv(filepath)
    
    fraud = data[data['Class'] == 1]
    normal = data[data['Class'] == 0]

    outlier_fraction = len(fraud)/len(normal)

    X = data.drop('Class', axis=1)
    y = data['Class']

    classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                    contamination=outlier_fraction,random_state=42, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                            leaf_size=30, metric='minkowski',
                                            p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                        max_iter=-1, random_state=42)
   }

    run_models(classifiers, X, y, outlier_fraction)

    

