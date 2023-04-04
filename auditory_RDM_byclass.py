from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import h5py
import scipy.stats
from scipy.stats import pearsonr
from neurora.stuff import limtozero

#LOAD DATASET
actList = ['JG', 'MM', 'JY']
#actList = ['MM', 'JG']
# load data
df_voice = pd.read_excel('/Users/yeye/Downloads/analysis.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
#df_voice = pd.read_excel('/Users/yeye/Downloads/analysis.xlsx', sheet_name=0, header=0,skiprows=[2, 4])
#基频转换
df_voice = df_voice.apply(lambda x:12*np.log2(x/100) if x.name in ['pitchMax','pitchMin', 'pitchrange','f1meanfrequency', 'f2meanfrequency', 'f3meanfrequency'] else x)
#df_voice = df_voice.apply(lambda x:12*np.log2(x/100) if x.name in ['pitchMax','pitchMin', 'pitchrange'] else x)

#实验数据分组
#df_voice = df_voice.drop(df_voice[(df_voice['speaker']=='f1') | (df_voice['speaker']=='f2')].index)
#df_voice = df_voice.drop(df_voice[(df_voice['speaker']=='m1') | (df_voice['speaker']=='m2') | (df_voice['speaker']=='f1')].index)
#df_voice = df_voice.drop(df_voice[(df_voice['speechact']=='JY')].index)

search_list = ['duration','f2meanfrequency', 'f3meanfrequency','meanintensity','pitchMax', 'pitchrange']
#search_list = ['meanintensity','pitchMax','duration','f3meanfrequency']
#search_list = ['pitchMax', 'f1energy','HNR','f3meanfrequency', 'gravity']
'''
search_list = ['meanintensity', 'pitchMax', 'pitchMin', 'pitchrange', 'duration',
        'HNR', 'gravity', 'f0energy', 'f1energy', 'f2energy', 
        'f3energy','f1meanfrequency', 'f2meanfrequency', 'f3meanfrequency']
'''

#按照意图类别分组   'JG'->0, 'MM'->1, 'JY'->2
X = df_voice[df_voice['speechact']=='JG'].loc[:,search_list].values
#y = df_voice.iloc[:, 2].values
y = np.array([0 for _ in range(48)])
X = np.append(X,df_voice[df_voice['speechact']=='MM'].loc[:,search_list].values, axis=0)
y = np.append(y, np.array([1 for _ in range(48)]), axis=0)
X = np.append(X, df_voice[df_voice['speechact']=='JY'].loc[:,search_list].values, axis=0)
y = np.append(y, np.array([2 for _ in range(48)]), axis=0)
#归一化
X = preprocessing.scale(X)
print(X.shape,y.shape)

print(X[0], y[0],X[49],y[49],X[-1],y[-1])

def audRDM(auditory_data, label, method="correlation", abs=False):
    # get the number of conditions & the number of subjects
    assert auditory_data.shape[0] == label.shape[0]

    # shape of emotion_data: [N_trials, feature_dim]

    # save the number of trials of each condition
    n_trials = auditory_data.shape[0]
    print("<<<<<<<< Computing RDM <<<<<<<<")
    print(n_trials)
    # initialize the RDM
    rdm = np.zeros([n_trials, n_trials], dtype=np.float64)

    # assignment
    # save the data for each subject under each condition, average the trials
    data = auditory_data

    # calculate the values in RDM
    for i in range(n_trials):
        for j in range(n_trials):
            if method is 'correlation':
                # calculate the Pearson Coefficient
                r = pearsonr(data[i], data[j])[0]
                # calculate the dissimilarity
                if abs == True:
                    rdm[i, j] = limtozero(1 - np.abs(r))
                else:
                    rdm[i, j] = limtozero(1 - r)
            elif method is 'euclidean':
                rdm[i, j] = np.linalg.norm(data[i]-data[j])
            elif method is 'mahalanobis':
                X = np.transpose(np.vstack((data[i], data[j])), (1, 0))
                X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
                rdm[i, j] = np.linalg.norm(X[:, 0]-X[:, 1])
    if method is 'euclidean' or method is 'mahalanobis':
        max = np.max(rdm)
        min = np.min(rdm)
        rdm = (rdm-min)/(max-min)

    print("\nRDM computing finished!")

    return rdm

# 'correlation'->Pearson相关系数；'euclidean'->欧氏距离；'mahalanobis'->马氏距离
auditory_rdm = audRDM(X, y, method='euclidean', abs=False)
f = h5py.File("rdms/auditory_6dim_euclidean.h5", "w")
f.create_dataset("auditory_6dim", data=auditory_rdm)
f.close()

from neurora.rsa_plot import plot_rdm
plot_rdm(auditory_rdm)
