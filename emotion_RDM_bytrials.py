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


subs = ["sub01", "sub02", "sub03", "sub05", "sub06", 'sub07','sub10','sub11',
'sub12','sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19','sub20','sub21',
'sub22','sub23','sub24','sub25','sub26','sub27','sub28','sub29','sub30','sub31']
nsubs = len(subs)

#LOAD DATASET
actList = ['JG', 'MM', 'JY']
#actList = ['MM', 'JG']
# load data
#search_list = ['duration','f2meanfrequency', 'f3meanfrequency','meanintensity','pitchMax', 'pitchrange']
#search_list = ['meanintensity','pitchMax','duration','f3meanfrequency']

'''
search_list = ['meanintensity', 'pitchMax', 'pitchMin', 'pitchrange', 'duration',
        'HNR', 'gravity', 'f0energy', 'f1energy', 'f2energy', 
        'f3energy','f1meanfrequency', 'f2meanfrequency', 'f3meanfrequency']
'''
df_bhv = pd.read_excel('./behave_28subs_new.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
df_index = pd.read_excel('./index_order.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
#df_index = df_index.apply(lambda x:12*np.log2(x/100) if x.name in ['pitchMax','pitchMin', 'pitchrange','f1meanfrequency', 'f2meanfrequency', 'f3meanfrequency'] else x)

#df_voice = pd.read_excel('./analysis.xlsx', sheet_name=0, header=0)
index_order = df_index['trial_ID'].values
#print(index_order)
for sub in subs:
   subid = int(sub[3:])
   trial_ID = df_bhv[df_bhv['subject']==subid].loc[:,'trial_ID'].values
   index_order = [i for i in index_order if i in trial_ID]
print(len(index_order))
#search_list = ['meanintensity','duration','pitchMax', 'f3meanfrequency']
search_list = ['attitude','valence','arousal','nature']
#print(df_index[df_index['trial_ID'].isin(index_order)].shape)
X = df_index[df_index['trial_ID'].isin(index_order)].loc[:,search_list].values
print(X.shape)


def emoRDM(emotion_data, method="correlation", abs=False):
    # get the number of conditions & the number of subjects
    assert emotion_data.shape[0] == 73
    # shape of emotion_data: [N_cons, feature_dim]
    # save the number of trials of each condition
    n_cons = emotion_data.shape[0]
    print("<<<<<<<< Computing RDM <<<<<<<<")
    print(n_cons)
    # initialize the RDM
    rdm = np.zeros([n_cons, n_cons], dtype=np.float64)

    # assignment
    # save the data for each subject under each condition, average the trials
    data = emotion_data

    # calculate the values in RDM
    for i in range(n_cons):
        for j in range(n_cons):
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
emotion_rdm = emoRDM(X, method='euclidean', abs=True)
f = h5py.File("rdms/emotion_euclidean_bytrials.h5", "w")
f.create_dataset("auditory", data=emotion_rdm)
f.close()

from neurora.rsa_plot import plot_rdm
plot_rdm(emotion_rdm)
