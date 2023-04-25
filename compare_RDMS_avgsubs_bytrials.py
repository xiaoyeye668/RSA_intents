import numpy as np
from neurora.corr_cal_by_rdm import rdms_corr
import h5py
import time
from neurora.rsa_plot import plot_rdm, plot_rdm_withvalue, plot_tbyt_decoding_acc,plot_corrs_by_time,plot_corrs_hotmap

#f = h5py.File("rdms/logauditory_pearson_absF_bytrials_6dim_balance.h5","r")
#f = h5py.File("rdms/logauditory_euclidean_bytrials_balance.h5","r")
#f = h5py.File("rdms/logauditory_pearson_bytrials_absF_balance.h5",'r')
#f = h5py.File("rdms/logauditory_euclidean_bytrials_6dim_balance.h5", "r")
#f = h5py.File("rdms/logauditory_euclidean_bytrials_6dim_balance_nonorm.h5", "r")
f = h5py.File("rdms/logauditory_euclidean_bytrials_3dim_balance.h5", "r")
#f = h5py.File("rdms/logauditory_euclidean_bytrials_3dim_balance_2.h5", 'r')
auditory_rdm = np.array(f["auditory"])
#plot_rdm(auditory_rdm)
f.close()

#f = h5py.File("rdms/emotion_euclidean_bytrials.h5", "r")
#f = h5py.File("rdms/emotion_pearson_bytrials_absF.h5", "r")
#f = h5py.File("rdms/emotion_euclidean_bytrials_2dim.h5","r")
#f = h5py.File("rdms/emotion_pearson_bytrials_absF_balance.h5", "r")
f = h5py.File("rdms/emotion_euclidean_bytrials_balance.h5", "r")
#f = h5py.File("rdms/emotion_pearson_bytrials_2dim_absF_balance.h5", "r")
#f = h5py.File("rdms/emotion_euclidean_bytrials_2dim_balance.h5", "r")
f = h5py.File("rdms/emotion_euclidean_bytrials_2dim_balance_new.h5", "r")

#emotion_rdm = np.array(f["auditory"])
emotion_rdm = np.array(f["emotion"])
#plot_rdm(emotion_rdm)
f.close()

#f = h5py.File("rdms/bhv_euclidean_bytrials.h5", "r")
#f = h5py.File("rdms/bhv_absolute_bytrials.h5", "r")
#f = h5py.File("rdms/bhv_pearson_bytrials_absF.h5", "r")
#f = h5py.File("rdms/bhv_euclidean_bytrials_balance.h5", "r")
f = h5py.File("rdms/bhv_absolute_bytrials_balance.h5", 'r')
#f = h5py.File("rdms/bhv_pearson_bytrials_absF_balance.h5", "r")
bhv_rdm = np.array(f["bhv_intents"])
#plot_rdm(bhv_rdm)
#plot_rdm_withvalue(bhv_rdm)
f.close()

#f = h5py.File("rdms/ERP_28subs_bytrials_balance_pearson_absF.h5", "r")
#f = h5py.File("rdms/ERP_28subs_bytrials_balance_euclidean.h5", "r")
f = h5py.File("rdms/ERP_28subs_bytrials_balance_pearson_absF_persub.h5", "r")
f = h5py.File('rdms/ERP_avgsub_avgchannel_bytrials_balance_euclidean.h5', 'r')
#f = h5py.File('rdms/ERP_persub_avgchannel_bytrials_balance_euclidean.h5', 'r')
#f = h5py.File('rdms/ERP_avgsub_avgchannel_bytrials_balance_pearson_absF.h5','r')
f = h5py.File('rdms/ERP_persub_avgchannel_bytrials_balance_pearson_absF.h5','r')
eeg_rdm = np.array(f["intents"])        #(170, 60, 60)
print(eeg_rdm.shape)    #(60, 170, 73, 73),ni(i=1, 2, 3) can be int(n_ts/timw_win), n_chls, n_subs.
#eeg_rdm = eeg_rdm.transpose(1,0,2,3)
#eeg_rdm = eeg_rdm[40:,:,:,:]
#eeg_rdm = eeg_rdm[:,40:140,:,:]
#eeg_rdm = eeg_rdm.transpose(2,1,0,3,4)
#eeg_rdm = eeg_rdm[40:140,:,:] 
eeg_rdm = eeg_rdm[:,40:140,:,:]   #(28, 60, 100, 60, 60)
#eeg_rdm = np.average(eeg_rdm, axis=1) #平均channel数据
#eeg_rdm = eeg_rdm[:,np.newaxis,:,:,:]
#for i in range(60):
#    plot_rdm(eeg_rdm[i,27,:,:].reshape([73,73]))
print(eeg_rdm.shape)
f.close()

#相似分析算法：spearman; kendall
print('<<<<<<<<<<<<< corrs caculation finish!')
print(auditory_rdm.shape, emotion_rdm.shape, bhv_rdm.shape, eeg_rdm.shape)
#(73, 73) (60, 90, 73, 73) (60, 90, 2)
#print(auditory_rdm.shape, corrs.shape)
#plot_corrs_by_time(corrs)
#plot_corrs_hotmap(corrs)

#相似分析算法：spearman; kendall
#corrs = np.zeros([3,100,2])
corrs = np.zeros([3,100,2])
corrs1 = rdms_corr(auditory_rdm, eeg_rdm, method="spearman")
corrs1 = np.average(rdms_corr(auditory_rdm, eeg_rdm, method="spearman"), axis=0)
corrs1 = corrs1[np.newaxis, :]
#corrs[0] = corrs1[::-1,:]
corrs[0] = corrs1
corrs2 = rdms_corr(emotion_rdm, eeg_rdm, method="spearman")
corrs2 = np.average(rdms_corr(emotion_rdm, eeg_rdm, method="spearman"), axis=0)
corrs2 = corrs2[np.newaxis, :]
corrs[1] = corrs2
corrs3 = rdms_corr(bhv_rdm, eeg_rdm, method="spearman")
corrs3 = np.average(rdms_corr(bhv_rdm, eeg_rdm, method="spearman"), axis=0)
corrs3 = corrs3[np.newaxis, :]
#corrs[2] = corrs3[::-1,:]
corrs[2] = corrs3
#If the shape of eeg_rdms is [n1, n2, n_cons, n_cons],the shape of corrs will be [n1, n2, 2]
print('<<<<<< 1 corrs.shape',corrs.shape)  
print('<<<<<<<<<<<<< corrs caculation finish!')
labels = ['corrs by auditory', 'corrs by emotion', 'corrs by intention behavior']
plot_corrs_by_time(corrs,labels=labels, time_unit=[0.2, 0.01])


