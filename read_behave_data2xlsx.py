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
df_voice = pd.read_excel('./behave_28subs.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头

subs = ["sub01", "sub02", "sub03", "sub05", "sub06", 'sub07','sub10','sub11',
'sub12','sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19','sub20','sub21',
'sub22','sub23','sub24','sub25','sub26','sub27','sub28','sub29','sub30','sub31']
nsubs = len(subs)
bhv_data = np.zeros([3, nsubs, 41], dtype=np.float64)   # [n_cons, n_subs, n_trials]
subindex = 0
for sub in subs:
    subid = int(sub[3:])
    df_data = df_voice[df_voice['subject']==subid]
    #print(subid, df_data.shape)
    trialindex = np.zeros([3], dtype=np.int32)  #分别记录类别index
    for index, row in df_data.iterrows():  
        if row['speechact'] == 'JG' and trialindex[0] < 41:
            bhv_data[0,subindex,trialindex[0]] = row['behavior intention']
            trialindex[0] += 1
        elif row['speechact'] == 'MM' and trialindex[1] < 41:
            bhv_data[1,subindex,trialindex[1]] = row['behavior intention']
            trialindex[1] += 1
        elif row['speechact'] == 'JY' and trialindex[2] < 41:
            bhv_data[2,subindex,trialindex[2]] = row['behavior intention']
            trialindex[2] += 1
        else:
            continue
    subindex += 1

#LOAD DATASET
actList = ['JG', 'MM', 'JY']
#actList = ['MM', 'JG']
# load data
df_voice = pd.read_excel('./behave_28subs.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
#df_voice = pd.read_excel('/Users/yeye/Downloads/analysis.xlsx', sheet_name=0, header=0,skiprows=[2, 4])

search_list = ['subject','speaker','group','speechact','behavior intention','attitude','valence','meanintensity',
'pitchMax','duration','f3MeanFrequency','logpitchmax','logf3MeanFrequency']
#按照意图类别分组   'JG'->0, 'MM'->1, 'JY'->2
#X = np.average(df_voice[df_voice['speechact']=='JG'].loc[:,search_list].values, axis=0)
X = df_voice.loc[:,search_list].values
print(X.shape)
#data = np.zeros((X.shape[0], X.shape[1]+1))
#print(data.shape)
for i in range(len(X)):
    trial_ID = X[i][1]+'_'+X[i][2]+'_'+X[i][3]
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
dataframe.to_excel("behave_28subs_new.xlsx",na_rep=False)
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
