import numpy as np
from neurora.corr_cal_by_rdm import rdms_corr
import h5py
import time
from neurora.rsa_plot import plot_rdm

#f = h5py.File("rdms/emotion_euclidean_bycon.h5", "r")
f = h5py.File("rdms/emotion_pearson_bycon.h5", "r")
emotion_rdm = np.array(f["emotion"])
#plot_rdm(emotion_rdm)
f.close()
f = h5py.File("rdms/auditory_euclidean_bycon.h5", "r")
#f = h5py.File("rdms/auditory_pearson_bycon.h5", "r")
auditory_rdm = np.array(f["auditory"])
plot_rdm(auditory_rdm)
f.close()
#f = h5py.File("rdms/bhv_euclidean_bycon.h5", "r")
f = h5py.File("rdms/bhv_pearson_bycon.h5", "r")
bhv_rdm = np.array(f["bhv_intents"])
f.close()
f = h5py.File("rdms/ERP_28subs_bycon.h5", "r")
eeg_rdm = np.array(f["intents"])
print(eeg_rdm.shape)
eeg_rdm = eeg_rdm[:,40:,:,:]
print(eeg_rdm.shape)
f.close()

#f = h5py.File("corrs/euclideanemotion_eeg_spearman_bycon.h5", "w")
#f = h5py.File("corrs/pearsonemotion_eeg_spearman_bycon.h5", "w")
f = h5py.File("corrs/euclideanauditory_eeg_spearman_bycon.h5", "w")
#f = h5py.File("corrs/pearsonauditory_eeg_spearman_bycon.h5", "w")
#f = h5py.File("corrs/euclideanbhv_eeg_spearman_bycon.h5", "w")
#f = h5py.File("corrs/pearsonbhv_eeg_spearman_bycon.h5", "w")
#corrs = rdms_corr(emotion_rdm, eeg_rdm)
corrs = rdms_corr(auditory_rdm, eeg_rdm)
#corrs = rdms_corr(bhv_rdm, eeg_rdm)
#corrs = rdms_corr(emotion_rdm, eeg_rdm, method="spearman", rescale=False, permutation=True, iter=1000)
#f.create_dataset("emotion_eeg", data=corrs)
f.create_dataset("auditory_eeg", data=corrs)
#f.create_dataset("bhv_eeg", data=corrs)

print('<<<<<<<<<<<<< corrs caculation finish!')
print(emotion_rdm.shape, eeg_rdm.shape,corrs.shape)
#print(auditory_rdm.shape, corrs.shape)

from neurora.stats_cal import stats
eeg_stats = stats(corrs)
#print(corrs)

#from neurora.rsa_plot import plot_tbytsim_withstats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from neurora.stuff import get_affine, correct_by_threshold, get_bg_ch2, get_bg_ch2bet, \
    clusterbased_permutation_1d_1samp_1sided, clusterbased_permutation_2d_1samp_1sided

def plot_tbytsim_withstats(similarities, start_time=0, end_time=1, smooth=True, p=0.05, cbpt=True, color='r',
                           lim=[-0.1, 0.8], figsize=[6.4, 3.6], x0=0, fontsize=16):

    """
    Plot the time-by-time Similarities with statistical results
    Parameters
    ----------
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

    if len(np.shape(similarities)) < 2 or len(np.shape(similarities)) > 3:

        return "Invalid input!"

    n = len(np.shape(similarities))

    minlim = lim[0]
    maxlim = lim[1]

    if n == 3:
        similarities = similarities[:, :, 0]

    nsubs = np.shape(similarities)[0]
    nts = np.shape(similarities)[1]

    tstep = float((end_time-start_time)/nts)

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
    print(similarities,similarities.shape)
    if cbpt == True:
        ps = clusterbased_permutation_1d_1samp_1sided(similarities, level=0, p_threshold=p, iter=5000)
        print(ps)
    else:
        ps = np.zeros([nts])
        for t in range(nts):
            ps[t] = ttest_1samp(similarities[:, t], 0, alternative="greater")[1]
            if ps[t] < p:
                ps[t] = 1
            else:
                ps[t] = 0

    for t in range(nts):
        if ps[t] == 1:
            plt.plot(t*tstep+start_time, (maxlim-minlim)*0.9+minlim, 's', color=color, alpha=1)
            xi = [t*tstep+start_time, t*tstep+tstep+start_time]
            ymin = [0]
            ymax = [avg[t]-err[t]]
            plt.fill_between(xi, ymax, ymin, facecolor=color, alpha=0.1)

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
plot_tbytsim_withstats(corrs, start_time=0, end_time=1.5, p=0.05, cbpt=True, lim=[-0.5, 0.5])
#plot_tbytsim_withstats(corrs, start_time=-0.2, end_time=1.5, time_interval=0.01, 
#                       p=0.05, cbpt=True, stats_time=[0, 1.5], xlim=[-0.2, 1.5], ylim=[-1.0, 1.0])
