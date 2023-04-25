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
from collections import defaultdict

def bhvRDM(bhv_data, sub_opt=1, method="correlation", abs=False):

    """
    Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for behavioral data
    Parameters
    ----------
    bhv_data : array
        The behavioral data.
        The shape of bhv_data must be [n_cons, n_subs, n_trials].
        n_cons, n_subs & n_trials represent the number of conidtions, the number of subjects & the number of trials,
        respectively.
    sub_opt: int 0 or 1. Default is 1.
        Return the results for each subject or after averaging.
        If sub_opt=1, return the results of each subject.
        If sub_opt=0, return the average result.
    method : string 'correlation' or 'euclidean' or 'mahalanobis'. Default is 'correlation'.
        The method to calculate the dissimilarities.
        If method='correlation', the dissimilarity is calculated by Pearson Correlation.
        If method='euclidean', the dissimilarity is calculated by Euclidean Distance, the results will be normalized.
        If method='mahalanobis', the dissimilarity is calculated by Mahalanobis Distance, the results will be normalized.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not. Only works when method='correlation'.
    Returns
    -------
    RDM(s) : array
        The behavioral RDM.
        If sub_opt=1, return n_subs RDMs. The shape is [n_subs, n_cons, n_cons].
        If sub_opt=0, return only one RDM. The shape is [n_cons, n_cons].
    Notes
    -----
    This function can also be used to calculate the RDM for computational simulation data.
        For example, users can extract the activations for a certain layer i which includes Nn nodes in a deep
        convolutional neural network (DCNN) corresponding to Ni images. Thus, the input could be a [Ni, 1, Nn] matrix, M.
        Using "bhvRDM(M, sub_opt=0)", users can obtain the DCNN RDM for layer i.
    """

    if len(np.shape(bhv_data)) != 3:

        print("\nThe shape of input for bhvEEG() function must be [n_cons, n_subs, n_trials].\n")

        return "Invalid input!"

    # get the number of conditions & the number of subjects
    cons = len(bhv_data)

    # get the number of conditions
    n_subs = []

    for i in range(cons):
        n_subs.append(np.shape(bhv_data[i])[0])

    subs = n_subs[0]

    # shape of bhv_data: [N_cons, N_subs, N_trials]

    # save the number of trials of each condition
    n_trials = []

    for i in range(cons):
        n_trials.append(np.shape(bhv_data[i])[1])

    # save the number of trials of each condition
    if len(set(n_trials)) != 1:
            return None

    # sub_opt=1

    if sub_opt == 1:

        print("\nComputing RDMs")

        # initialize the RDMs
        rdms = np.zeros([subs, cons, cons], dtype=np.float64)

        # calculate the values in RDMs
        for sub in range(subs):
            rdm = np.zeros([cons, cons], dtype=np.float)
            for i in range(cons):
                for j in range(cons):
                    # calculate the difference
                    if abs == True:
                        rdm[i, j] = np.abs(np.average(bhv_data[i, sub])-np.average(bhv_data[j, sub]))
                    else:
                        rdm[i, j] = np.average(bhv_data[i, sub]) - np.average(bhv_data[j, sub])

            # flatten the RDM
            vrdm = np.reshape(rdm, [cons * cons])
            # array -> set -> list
            svrdm = set(vrdm)
            lvrdm = list(svrdm)
            lvrdm.sort()

            # get max & min
            maxvalue = lvrdm[-1]
            minvalue = lvrdm[1]

            # rescale
            if maxvalue != minvalue:

                for i in range(cons):
                    for j in range(cons):

                        # not on the diagnal
                        if i != j:
                            rdm[i, j] = (rdm[i, j] - minvalue) / (maxvalue - minvalue)
            rdms[sub] = rdm

        print("\nRDMs computing finished!")

        return rdms

    # & sub_opt=0

    print("\nComputing RDM")

    # initialize the RDM
    rdm = np.zeros([cons, cons], dtype=np.float64)

    # judge whether numbers of trials of different conditions are same
    if len(set(n_subs)) != 1:
        return None

    # assignment
    # save the data for each subject under each condition, average the trials
    #data = np.average(bhv_data, axis=2)
    data = bhv_data
    #print(bhv_data.shape, data.shape)
    # calculate the values in RDM
    for i in range(cons):
        for j in range(cons):
            if method is 'correlation':
                # calculate the Pearson Coefficient
                r = pearsonr(data[i].ravel(), data[j].ravel())[0]
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
            elif method is 'absolute':
                x, y = np.average(data[i]), np.average(data[j])
                print(x, y)
                rdm[i, j] = np.abs(x - y)

            print(data[i], data[j],rdm[i, j])
    if method is 'euclidean' or method is 'mahalanobis':
        max = np.max(rdm)
        min = np.min(rdm)
        rdm = (rdm-min)/(max-min)

    print("\nRDM computing finished!")

    return rdm

subs = ["sub01", "sub02", "sub03", "sub05", "sub06", 'sub07','sub10','sub11',
'sub12','sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19','sub20','sub21',
'sub22','sub23','sub24','sub25','sub26','sub27','sub28','sub29','sub30','sub31']
nsubs = len(subs)

#LOAD DATASET
actList = ['JG', 'MM', 'JY']
df_bhv = pd.read_excel('./behave_28subs_new.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
df_index = pd.read_excel('./index_order.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
#df_index = df_index.apply(lambda x:12*np.log2(x/100) if x.name in ['pitchMax','pitchMin', 'pitchrange','f1meanfrequency', 'f2meanfrequency', 'f3meanfrequency'] else x)

#df_voice = pd.read_excel('./analysis.xlsx', sheet_name=0, header=0)
index_order_tmp = df_index['trial_ID'].values
index_order = []
#print(index_order)

for sub in subs:
   subid = int(sub[3:])
   trial_ID = df_bhv[df_bhv['subject']==subid].loc[:,'trial_ID'].values
   index_order_tmp = [i for i in index_order_tmp if i in trial_ID]
print(len(index_order))
#计算筛选后各类别trial数
dictmp = defaultdict()
for trialid in index_order_tmp:
    condition = trialid.split('_')[-1]
    if condition not in dictmp:
        dictmp[condition] = 1
    else:
        dictmp[condition] += 1
    #index_order.append(trialid)
    if dictmp[condition] <= 20:
        index_order.append(trialid)
    
print(dictmp, len(index_order))
search_list = ['behavior intention']
#print(df_index[df_index['trial_ID'].isin(index_order)].shape)
#X = df_bhv[df_bhv['trial_ID'].isin(index_order)].loc[:,search_list].values
#print(X.shape)

bhv_data = np.zeros([60, nsubs, 1], dtype=np.float64)   # [n_cons, n_subs, n_trials]
subindex = 0
for sub in subs:
    subid = int(sub[3:])
    df_data = df_bhv[df_bhv['subject']==subid]
    sub_bhv = df_data[df_data['trial_ID'].isin(index_order)].loc[:,search_list].values
    #print(subid, df_data.shape, sub_bhv.shape)
    bhv_data[:,subindex,:] = sub_bhv
    subindex += 1

# 'correlation'->Pearson相关系数；'euclidean'->欧氏距离；'mahalanobis'->马氏距离； 'absolute'->绝对值差
intents_rdm = bhvRDM(bhv_data, sub_opt=0, method="correlation", abs=False)
f = h5py.File("rdms/bhv_pearson_bytrials_absF_balance.h5", "w")
f.create_dataset("bhv_intents", data=intents_rdm)
f.close()
print(intents_rdm.shape)
from neurora.rsa_plot import plot_rdm
plot_rdm(intents_rdm)
