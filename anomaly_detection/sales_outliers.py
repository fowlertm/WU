## Adapted from: https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1

import os
import sys
from time import time

import pandas as pd
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.ensemble import IsolationForest

path = "../data/superstore_sales.xls"

def get_data(filename, is_excel=True):
    df = pd.read_excel(filename)
    return df

def get_stats(df, col, numbers_too=False):

    desc = df[col].describe()
    skew = df[col].skew()
    kurtosis = df[col].kurtosis()
    message = "Description: "+"\n"+str(desc)+"\n"+"Skew: "+str(skew)+"\n"+"Kurtosis: "+str(kurtosis)

    if numbers_too:
        return skew, kurtosis, message
    else:
        return message

def visualizations(df, col):

    plt.scatter(range(df.shape[0]), np.sort(df[col].values))
    plt.xlabel('index')
    plt.ylabel(col)
    plt.title(f"{col} distribution")

    return plt

def isolation_forest_detector(df, col):

    isolation_forest = IsolationForest(n_estimators=100)
    isolation_forest.fit(df[col].values.reshape(-1, 1))
    xx = np.linspace(df[col].min(), df[col].max(), len(df)).reshape(-1,1)
    anomaly_score = isolation_forest.decision_function(xx)
    outlier = isolation_forest.predict(xx)

    plt.figure(figsize=(10,4))
    plt.plot(xx, anomaly_score, label='anomaly score')
    plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                    where=outlier==-1, color='r', 
                    alpha=.4, label='outlier region')
    plt.legend()
    plt.ylabel('anomaly score')
    plt.xlabel(col)

    return plt



if __name__ == "__main__":

    user_col = input("Which column would you like to use?")
    sales_df = get_data(path)
    message = get_stats(sales_df, user_col)
    print(message)
    plot = visualizations(sales_df, user_col)
    plot.show()
    iso_plot = isolation_forest_detector(sales_df, user_col)
    iso_plot.show()
