import pandas as pd
import os
import datetime
from datetime import datetime
import numpy as np
from pandas import Timestamp
import re
import string
import unicodedata
from scipy import stats
import scipy.cluster
import seaborn as sns
import statistics

from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, FeatureAgglomeration
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster

# Data Preprocessing
# exec(open("Data Preprocessing.py", "r").read())

pd.set_option('display.max_columns',0)

originalCustomer = SP_Customer.copy()

SP_Customer.drop(SP_Customer.columns.difference(['GroupTicket_Percentage','TotalTransAmount','BlackActivityDays','isEducation_count','TuesdayAttendee_tendancy','AttendsWithChild_tendancy','OpensEmail_value_LOW','OpensEmail_value_MED','OpensEmail_value_HIGH','NightTransCount']), 1, inplace=True)

min_max_scaler = MinMaxScaler()
X = SP_Customer.to_numpy()
X = min_max_scaler.fit_transform(X)

def checkClusterSize(labels, clusterNumber):
    total = 0
    clusterSizes = []
    for i in range(0,clusterNumber):
        clusterCount = np.count_nonzero(labels == i)
        clusterSizes.append(clusterCount)
        print("Cluster ", i, ": ", clusterCount)
        total += clusterCount
    return clusterSizes


def stats_to_df(d,cols,medians):
    tmp_df = pd.DataFrame(columns=cols)
    try:
        tmp_df.loc[0] = np.round(d.minmax[0],2)
        tmp_df.loc[1] = np.round(medians,2)
        tmp_df.loc[2] = np.round(d.mean,2)
        tmp_df.loc[3] = np.round(d.minmax[1],2)
        tmp_df.loc[4] = np.round(d.variance,2)
        tmp_df.loc[5] = np.round(d.skewness,2)
        tmp_df.loc[6] = np.round(d.kurtosis,2)
        tmp_df.index = ['Min', 'Median', 'Mean', 'Max', 'Variance', 'Skewness', 'Kurtosis']
    except Exception:
        print(d)
    return tmp_df.T


def printStats(labels,X,cols,cluster_n):
    for i in range(0,cluster_n):
        d = stats.describe(min_max_scaler.inverse_transform(X[i==labels]),axis=0)
        print('\nCluster {}:'.format(i+1), 'Number of instances: {}'.format(d.nobs), 'Percentage: {:.2f}%'.format(d.nobs/X.shape[0]*100))
        print(stats_to_df(d,cols,getMedians(labels,X,i)))


def printSScoreChart(X, affinity, linkage):
    silhouette_score_values=list()
    NumberOfClusters = range(6,10)
    for i in NumberOfClusters:
        agg = AgglomerativeClustering(n_clusters=i, affinity=affinity, linkage=linkage)
        agg.fit(X)
        silhouette_score_values.append(silhouette_score(X,agg.labels_ ,metric=affinity, sample_size=None, random_state=None))
    plt.plot(NumberOfClusters, silhouette_score_values)
    plt.title("Silhouette score values vs Numbers of Clusters")
    plt.show()


def printAllData(X, cols):
    print('All Data:')
    print('Number of Instances: {}'.format(X.shape[0]))
    d = stats.describe(min_max_scaler.inverse_transform(X), axis=0)
    print(stats_to_df(d,cols,getAllMedians(X)))


def getMedians(labels,X,cols):
    medians = []
    df = min_max_scaler.inverse_transform(X[cols==labels])
    for c in range(0,10):
        medians.append(statistics.median(df[:,c]))
    return medians


def getAllMedians(X):
    medians = []
    df = min_max_scaler.inverse_transform(X)
    for c in range(0,10):
        medians.append(statistics.median(df[:,c]))
    return medians


def printBoxPlots(labels,df):
    print('Printing box plots for each feature ...')
    columns = ['age','GroupTicket_Percentage','TotalTransAmount','BlackActivityDays','isEducation_count','TuesdayAttendee_tendancy','AttendsWithChild_tendancy','OpensEmail_value_LOW','OpensEmail_value_MED','OpensEmail_value_HIGH','NightTransCount']
    df['lable'] = labels
    for m, n in enumerate(columns):
        df.boxplot(column=n, by='lable', figsize = (10, 5), showfliers=False)
        plt.show()


def printCorrPlotForAllData(data):
    print('Printing correlation plot for whole dataset ...')
    f, ax = plt.subplots(figsize=(10, 8))
    corr = data.corr()
    sns.set(font_scale=0.8)
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
    ax.set_title('Correlation Plot for All Data')
    plt.show()


def printCorrPlots(X,labels):
    print('Printing correlation plots for each cluster ...')
    for i in range(0,8):
        f, ax = plt.subplots(figsize=(10, 8))
        df = min_max_scaler.inverse_transform(X[i==labels])
        dataset = pd.DataFrame({'BlackActivityDays': df[:, 0], 'TuesdayAttendee_tendancy': df[:, 1], 'AttendsWithChild_tendancy': df[:, 2], 'GroupTicket_Percentage': df[:, 3], 'OpensEmail_value_LOW': df[:, 4], 'OpensEmail_value_MED': df[:, 5], 'OpensEmail_value_HIGH': df[:, 6], 'TotalTransAmount': df[:, 7], 'NightTransCount': df[:, 8], 'isEducation_count': df[:, 9]})
        corr = dataset.corr()
        sns.set(font_scale=0.8)
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
        ax.set_title('Correlation Plot for Cluster {}'.format(i+1))
        plt.show()



printAllData(X,SP_Customer.columns)
printSScoreChart(X, 'euclidean', 'ward')
agg = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='ward')
agg.fit(X)
s_score = silhouette_score(X, agg.labels_)
print('\nS Score: ', s_score,'\n')
checkClusterSize(agg.labels_, 8)
printStats(agg.labels_,X,SP_Customer.columns,8)
printBoxPlots(agg.labels_, originalCustomer.copy())
printCorrPlotForAllData(SP_Customer)
printCorrPlots(X,agg.labels_)
