import numpy as np
from neurora.corr_cal_by_rdm import rdms_corr
import h5py
import time
from neurora.rsa_plot import plot_rdm, plot_rdm_withvalue, plot_tbyt_decoding_acc,plot_corrs_by_time,plot_corrs_hotmap

f = h5py.File("rdms/logauditory_euclidean_bytrials_4dim_balance.h5", "r")
auditory_rdm = np.array(f["auditory"])
#plot_rdm(auditory_rdm)
f.close()

f = h5py.File("rdms/emotion_euclidean_bytrials_2dim_balance_1.h5", "r")
emotion_rdm = np.array(f["emotion"])
#plot_rdm(emotion_rdm)
f.close()

f = h5py.File("rdms/bhv_pearson_bytrials_absF_balance.h5", "r")
bhv_rdm = np.array(f["bhv_intents"])
#plot_rdm(bhv_rdm)
#plot_rdm_withvalue(bhv_rdm)
f.close()

f = h5py.File("rdms/ERP_persub_avgchannel_bytrials_balance_pearson_absF.h5", "r")
eeg_rdm = np.array(f["intents"])        #(28, 170, 60, 60)
print(eeg_rdm.shape)    #(60, 170, 73, 73),ni(i=1, 2, 3) can be int(n_ts/timw_win), n_chls, n_subs.
eeg_rdm = eeg_rdm[:,40:140,:,:]
print(eeg_rdm.shape)
f.close()
#for i in range(60):
#    plot_rdm(eeg_rdm[i,27,:,:])

#相似分析算法：spearman; kendall
corrs = rdms_corr(auditory_rdm, eeg_rdm, method="spearman")
#corrs = rdms_corr(auditory_rdm, eeg_rdm, method="kendall")
#corrs = rdms_corr(emotion_rdm, eeg_rdm, method="spearman")
#corrs = rdms_corr(bhv_rdm, eeg_rdm, method="spearman")
#corrs = rdms_corr(bhv_rdm, eeg_rdm, method="spearman", rescale=False, permutation=True, iter=1000)

print('<<<<<< 1 corrs.shape',corrs.shape)  
print('<<<<<<<<<<<<< corrs caculation finish!')
print(auditory_rdm.shape, emotion_rdm.shape, bhv_rdm.shape, eeg_rdm.shape,corrs.shape)
#(73, 73) (60, 90, 73, 73) (60, 90, 2)
#print(auditory_rdm.shape, corrs.shape)
#plot_corrs_by_time(corrs)
#plot_corrs_hotmap(corrs)
'''
from neurora.stats_cal import stats
eeg_stats = stats(corrs)
#print(corrs)
'''

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#from neurora.stuff import get_affine, correct_by_threshold, get_bg_ch2, get_bg_ch2bet, \
#    clusterbased_permutation_1d_1samp_1sided, clusterbased_permutation_2d_1samp_1sided
from stuff import clusterbased_permutation_1d_1samp_1sided


def plot_tbytsim_withstats(task, similarities, start_time=0, end_time=1, smooth=True, p=0.05, cbpt=True, color='r',
                           lim=[-0.1, 0.8], figsize=[6.4, 3.6], x0=0, fontsize=16):

    """
    Plot the time-by-time Similarities with statistical results
    Parameters
    ----------
    task : str
        The rdm from which dimensional representation
    similarities : array
        The Similarities.
        The size of similarities should be [n_subs, n_ts] or [n_subs, n_ts, 2]. n_subs, n_ts represent the number of
        subjects and number of time-points. 2 represents the similarity and a p-value.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    smooth : bool True or False. Default is True.
        Smooth the results or not.
    p : float. Default is 0.05.
        The threshold of p-values.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    color : matplotlib color or None. Default is 'r'.
        The color for the curve.
    lim : array or list [min, max]. Default is [-0.1, 0.8].
        The corrs view lims.
    figsize : array or list, [size_X, size_Y]. Default is [6.4, 3.6].
        The size of the figure.
    x0 : float. Default is 0.
        The Y-axis is at x=x0.
    fontsize : int or float. Default is 16.
        The fontsize of the labels.
    """
    print('<<<<<<<<<< similarity', similarities.shape)  #(130, 60, 2)
    if len(np.shape(similarities)) < 2 or len(np.shape(similarities)) > 3:

        return "Invalid input!"

    n = len(np.shape(similarities))

    minlim = lim[0]
    maxlim = lim[1]

    if n == 3:
        similarities = similarities[:, :, 0]    #[n_channels, n_ts, 2]

    nsubs = np.shape(similarities)[0]
    nts = np.shape(similarities)[1]

    tstep = float((end_time-start_time)/nts)
    if task == 'emotion' or task == 'bhv':
        similarities = similarities*-1

    if smooth is True:
        for sub in range(nsubs):
            for t in range(nts):

                if t<=1:
                    similarities[sub, t] = np.average(similarities[sub, :t+3])
                if t>1 and t<(nts-2):
                    similarities[sub, t] = np.average(similarities[sub, t-2:t+3])
                if t>=(nts-2):
                    similarities[sub, t] = np.average(similarities[sub, t-2:])

    avg = np.average(similarities, axis=0)
    err = np.zeros([nts], dtype=np.float64)

    for t in range(nts):
        err[t] = np.std(similarities[:, t], ddof=1)/np.sqrt(nsubs)
    
    print(similarities.shape)
    if cbpt == True:
        ps,p_value = clusterbased_permutation_1d_1samp_1sided(similarities, level=0, p_threshold=p, iter=1000)
    else:
        ps = np.zeros([nts])
        for t in range(nts):
            ps[t] = ttest_1samp(similarities[:, t], 0, alternative="greater")[1]
            if ps[t] < p:
                ps[t] = 1
            else:
                ps[t] = 0
    tmp_p,count = 0,0
    for t in range(nts):
        if ps[t] == 1:
            plt.plot(t*tstep+start_time, (maxlim-minlim)*0.9+minlim, 's', color=color, alpha=1)
            xi = [t*tstep+start_time, t*tstep+tstep+start_time]
            ymin = [0]
            ymax = [avg[t]-err[t]]
            plt.fill_between(xi, ymax, ymin, facecolor=color, alpha=0.1)
            print(str((t*tstep+start_time)*1000)+'ms', p_value[t])
            tmp_p += p_value[t]
            count += 1
            #plt.text(t, ymax, p_value[t], ha='center', va='bottom', fontsize=20)
    if count > 0:
        print('average P is ', tmp_p/count)
    else:
        print('There is no significance section')
    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_position(("data", x0))
    ax.spines["bottom"].set_linewidth(3)
    ax.spines['bottom'].set_position(('data', 0))

    x = np.arange(start_time+0.5*tstep, end_time+0.5*tstep, tstep)
    plt.fill_between(x, avg + err, avg - err, facecolor=color, alpha=0.8)
    plt.ylim(minlim, maxlim)
    plt.xlim(start_time, end_time)
    plt.tick_params(labelsize=12)
    plt.xlabel("Time (s)", fontsize=fontsize)
    plt.ylabel("Representational Similarity", fontsize=fontsize)
    plt.show()

    return 0
print('<<<<<< 2 corrs.shape',corrs.shape)  
plot_tbytsim_withstats('auditory', corrs, start_time=0.2, end_time=1.2, p=0.05, cbpt=True, figsize=[8.3, 4], x0=0.2, lim=[-0.05, 0.05])
#plot_tbytsim_withstats('emotion', corrs, start_time=0.2, end_time=1.2, p=0.05, cbpt=True, figsize=[8.3, 4], x0=0.2, lim=[-0.05, 0.05])
#plot_tbytsim_withstats('bhv', corrs, start_time=0.2, end_time=1.2, p=0.05, cbpt=True, figsize=[8.3, 4], x0=0.2, lim=[-0.05, 0.05])
#plot_tbytsim_withstats(corrs, start_time=-0.2, end_time=1.5, time_interval=0.01, 
#                       p=0.05, cbpt=True, stats_time=[0, 1.5], xlim=[-0.2, 1.5], ylim=[-1.0, 1.0])

