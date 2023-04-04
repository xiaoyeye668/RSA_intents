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
#df_voice = pd.read_excel('./behave_28subs.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
df_voice = pd.read_excel('./analysis.xlsx', sheet_name=0, header=0)

search_list = ['speaker','item','speechact','attitude','valence','arousal','nature',
'duration','f2meanfrequency', 'f3meanfrequency','meanintensity','pitchMax', 'pitchrange']
#按照意图类别分组   'JG'->0, 'MM'->1, 'JY'->2
#X = np.average(df_voice[df_voice['speechact']=='JG'].loc[:,search_list].values, axis=0)
X = df_voice.loc[:,search_list].values
print(X.shape)
#data = np.zeros((X.shape[0], X.shape[1]+1))
#print(data.shape)
for i in range(len(X)):
    trial_ID = X[i][0]+'_'+X[i][1]+'_'+X[i][2]
    #print(trial_ID)
    if i == 0:
        data = np.append(X[i], trial_ID)
        data = data[np.newaxis,:]
    else:
        new_row = np.append(X[i], trial_ID)[np.newaxis,:]
        #print(new_row, new_row.shape, data.shape)
        data = np.vstack((data, new_row))
print(data.shape)
print('<<<<<<<<<<<<<<< resave <<<<<<<<<<<<<<<<<<<<')
new_list = search_list + ['trial_ID']
print(new_list)
dataframe = pd.DataFrame(data,columns=new_list)
dataframe.to_excel("index_order.xlsx",na_rep=False)
'''
#y = df_voice.iloc[:, 2].values
#y = np.array([0 for _ in range(48)])
#print(df_voice[df_voice['speechact']=='JY'].loc[:,search_list].index)
emo_MM = np.average(df_voice[df_voice['speechact']=='MM'].loc[:,search_list].values, axis=0)
X = np.append(X, emo_MM[np.newaxis,:], axis=0)
print(X.shape)
#y = np.append(y, np.array([1 for _ in range(48)]), axis=0)
emo_JY = np.average(df_voice[df_voice['speechact']=='JY'].loc[:,search_list].values, axis=0)
X = np.append(X, emo_JY[np.newaxis,:], axis=0)
print(X.shape)
#y = np.append(y, np.array([2 for _ in range(48)]), axis=0)
print(X[0])
'''