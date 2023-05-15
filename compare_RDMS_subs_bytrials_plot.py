import numpy as np
import matplotlib.pyplot as plt
from neurora.corr_cal_by_rdm import rdms_corr
import h5py
import time
from neurora.rsa_plot import plot_rdm, plot_rdm_withvalue, plot_corrs_hotmap
from scipy.interpolate import interp1d

def plot_corrs_by_time(corrs, labels=None, time_unit=[0, 0.1]):

    """
    plot the correlation coefficients by time sequence

    corrs : array
        The correlation coefficients time-by-time.
        The shape of corrs must be [n, ts, 2] or [n, ts]. n represents the number of curves of the correlation
        coefficient by time sequence. ts represents the time-points. If shape of corrs is [n, ts 2], each time-point
        of each correlation coefficient curve contains a r-value and a p-value. If shape is [n, ts], only r-values.
    label : string-array or string-list or None. Default is None.
        The label for each corrs curve.
        If label=None, no legend in the figure.
    time_unit : array or list [start_t, t_step]. Default is [0, 0.1]
        The time information of corrs for plotting
        start_t represents the start time and t_step represents the time between two adjacent time-points. Default
        time_unit=[0, 0.1], which means the start time of corrs is 0 sec and the time step is 0.1 sec.
    """

    if len(np.shape(corrs)) < 2 or len(np.shape(corrs)) > 3:

        return "Invalid input!"

    # get the number of curves
    n = corrs.shape[0]

    # get the number of time-points
    ts = corrs.shape[1]

    # get the start time and the time step
    start_t = time_unit[0]
    tstep = time_unit[1]

    # calculate the end time
    end_t = start_t + ts * tstep

    # initialize the x
    x = np.arange(start_t, end_t, tstep)

    # interp1d t
    t = ts * 50

    # initialize the interp1d x
    x_soft = np.linspace(x.min(), x.max(), t)

    # initialize the interp1d y
    y_soft = np.zeros([n, t])

    # interp1d
    for i in range(n):
        if len(corrs.shape) == 3:
            f = interp1d(x, corrs[i, :, 0], kind='cubic')
            y_soft[i] = f(x_soft)
        if len(corrs.shape) == 2:
            f = interp1d(x, corrs[i, :], kind='cubic')
            y_soft[i] = f(x_soft)

    # get the max value
    vmax = np.max(y_soft)
    # get the min value
    vmin = np.min(y_soft)

    if vmax <= 1/1.1:
        ymax = np.max(y_soft)*1.1
    else:
        ymax = 1

    if vmin >= 0:
        ymin = -0.1
    elif vmin < 0 and vmin > -1/1.1:
        ymin = np.min(y_soft)*1.1
    else:
        ymin = -1

    fig, ax = plt.subplots()

    for i in range(n):

        if labels:
            plt.plot(x_soft, y_soft[i], linewidth=3, label=labels[i])
        else:
            plt.plot(x_soft, y_soft[i], linewidth=3)

    plt.ylim(ymin, ymax)
    plt.ylabel("Similarity", fontsize=20)
    plt.xlabel("Time (s)", fontsize=20)
    plt.tick_params(labelsize=18)

    if labels:
        plt.legend()

    #隐藏右边和上边
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #调整左边和下边坐标轴位置
    #ax.spines['left'].set_position(("outward", -20))
    #ax.spines['bottom'].set_position(("outward", -20))
    ax.spines['left'].set_position(('data', 0.2))
    #ax.spines['bottom'].set_position(('data', 0))

    plt.show()

    return 0

f = h5py.File("rdms/logauditory_euclidean_bytrials_4dim_balance.h5", "r")
auditory_rdm = np.array(f["auditory"])
#plot_rdm(auditory_rdm)
f.close()

f = h5py.File("rdms/emotion_euclidean_bytrials_2dim_balance_1.h5", "r")
#emotion_rdm = np.array(f["auditory"])
emotion_rdm = np.array(f["emotion"])
#plot_rdm(emotion_rdm)
f.close()

#f = h5py.File("rdms/bhv_euclidean_bytrials_balance.h5", "r")
f = h5py.File("rdms/bhv_pearson_bytrials_absF_balance.h5", "r")
bhv_rdm = np.array(f["bhv_intents"])
plot_rdm(bhv_rdm)
#plot_rdm_withvalue(bhv_rdm)
f.close()
#f = h5py.File("rdms/ERP_28subs_bytrials.h5", "r")
f = h5py.File("rdms/ERP_avgsub_avgchannel_bytrials_balance_pearson_absF.h5", "r")
#f = h5py.File("rdms/ERP_persub_avgchannel_bytrials_balance_pearson_absF.h5", "r")
eeg_rdm = np.array(f["intents"])        #(28, 170, 60, 60)
print(eeg_rdm.shape)   
eeg_rdm = eeg_rdm[40:140,:,:]
#eeg_rdm = eeg_rdm[:,40:140,:,:]
#for i in range(60):
#    plot_rdm(eeg_rdm[i,27,:,:].reshape([73,73]))
print(eeg_rdm.shape)
f.close()

#f = h5py.File("corrs/euclideanauditory_eeg_spearman_bycon.h5", "w")
#f = h5py.File("corrs/pearsonauditory_eeg_spearman_bycon.h5", "w")

#f = h5py.File("corrs/euclideanemotion_eeg_spearman_bycon.h5", "w")
#f = h5py.File("corrs/pearsonemotion_eeg_spearman_bycon.h5", "w")

#f = h5py.File("corrs/euclideanbhv_eeg_spearman_bycon.h5", "w")
#f = h5py.File("corrs/pearsonbhv_eeg_spearman_bycon.h5", "w")

#相似分析算法：spearman; kendall
#corrs = np.zeros([3,100,2])
corrs = np.zeros([3,100,2])
#corrs1 = np.average(rdms_corr(auditory_rdm, eeg_rdm, method="spearman"), axis=0)
corrs1 = rdms_corr(auditory_rdm, eeg_rdm, method="spearman")
print(corrs1.shape)
corrs1 = corrs1[np.newaxis, :]
corrs[0] = corrs1
#corrs2 = np.average(rdms_corr(emotion_rdm, eeg_rdm, method="spearman"), axis=0)
corrs2 = rdms_corr(emotion_rdm, eeg_rdm, method="spearman")
corrs2 = corrs2[np.newaxis, :]
corrs[1] = corrs2*-1
#corrs3 = np.average(rdms_corr(bhv_rdm, eeg_rdm, method="spearman"), axis=0)
corrs3 = rdms_corr(bhv_rdm, eeg_rdm, method="spearman")
corrs3 = corrs3[np.newaxis, :]
corrs[2] = corrs3*-1
#corrs = rdms_corr(auditory_rdm, eeg_rdm, method="kendall")
#corrs = rdms_corr(emotion_rdm, eeg_rdm, method="spearman")
#corrs = rdms_corr(bhv_rdm, eeg_rdm, method="spearman")
#corrs = rdms_corr(bhv_rdm, eeg_rdm, method="kendall")
#corrs = rdms_corr(bhv_rdm, eeg_rdm, method="spearman", rescale=False, permutation=True, iter=1000)
#If the shape of eeg_rdms is [n1, n2, n_cons, n_cons],the shape of corrs will be [n1, n2, 2]
print(corrs.shape)   #corrs.shape:(60, 100, 2)
#f.create_dataset("auditory_eeg", data=corrs)
#f.create_dataset("emotion_eeg", data=corrs)
#f.create_dataset("bhv_eeg", data=corrs)
print('<<<<<< 1 corrs.shape',corrs.shape)  
print('<<<<<<<<<<<<< corrs caculation finish!')
print(auditory_rdm.shape, emotion_rdm.shape, bhv_rdm.shape, eeg_rdm.shape,corrs.shape)
#(73, 73) (60, 90, 73, 73) (60, 90, 2)
#print(auditory_rdm.shape, corrs.shape)

print(corrs.shape) 
labels = ['corrs by auditory', 'corrs by emotion', 'corrs by intention behavior']
plot_corrs_by_time(corrs,labels=labels, time_unit=[0.2, 0.01])
#plot_corrs_hotmap(corrs)

