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
from neurora.stuff import show_progressbar

def eegRDM(EEG_data, sub_opt=1, chl_opt=0, time_opt=1, time_win=5, time_step=5, method="correlation", abs=True):

    """
    Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) based on EEG-like data
    Parameters
    ----------
    EEG_data : array
        The EEG/MEG/fNIRS data.
        The shape of EEGdata must be [n_cons, n_subs, n_trials, n_chls, n_ts].
        n_cons, n_subs, n_trials, n_chls & n_ts represent the number of conidtions, the number of subjects, the number
        of trials, the number of channels & the number of time-points, respectively.
    sub_opt: int 0 or 1. Default is 1.
        Return the subject-result or average-result.
        If sub_opt=0, return the average result.
        If sub_opt=1, return the results of each subject.
    chl_opt : int 0 or 1. Default is 0.
        Calculate the RDM for each channel or not.
        If chl_opt=0, calculate the RDM based on all channels'data.
        If chl_opt=1, calculate the RDMs based on each channel's data respectively.
    time_opt : int 0 or 1. Default is 0.
        Calculate the RDM for each time-point or not
        If time_opt=0, calculate the RDM based on whole time-points' data.
        If time_opt=1, calculate the RDMs based on each time-points respectively.
    time_win : int. Default is 5.
        Set a time-window for calculating the RDM for different time-points.
        Only when time_opt=1, time_win works.
        If time_win=5, that means each calculation process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of calculating.
        Only when time_opt=1, time_step works.
    method : string 'correlation' or 'euclidean' or 'mahalanobis'. Default is 'correlation'.
        The method to calculate the dissimilarities.
        If method='correlation', the dissimilarity is calculated by Pearson Correlation.
        If method='euclidean', the dissimilarity is calculated by Euclidean Distance, the results will be normalized.
        If method='mahalanobis', the dissimilarity is calculated by Mahalanobis Distance, the results will be normalized.
    abs : boolean True or False. Default is True.
        Calculate the absolute value of Pearson r or not.
    Returns
    -------
    RDM(s) : array
        The EEG/MEG/fNIR RDM.
        If sub_opt=0 & chl_opt=0 & time_opt=0, return only one RDM.
            The shape is [n_cons, n_cons].
        If sub_opt=0 & chl_opt=0 & time_opt=1, return int((n_ts-time_win)/time_step)+1 RDM.
            The shape is [int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
        If sub_opt=0 & chl_opt=1 & time_opt=0, return n_chls RDM.
            The shape is [n_chls, n_cons, n_cons].
        If sub_opt=0 & chl_opt=1 & time_opt=1, return n_chls*(int((n_ts-time_win)/time_step)+1) RDM.
            The shape is [n_chls, int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
        If sub_opt=1 & chl_opt=0 & time_opt=0, return n_subs RDM.
            The shape is [n_subs, n_cons, n_cons].
        If sub_opt=1 & chl_opt=0 & time_opt=1, return n_subs*(int((n_ts-time_win)/time_step)+1) RDM.
            The shape is [n_subs, int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
        If sub_opt=1 & chl_opt=1 & time_opt=0, return n_subs*n_chls RDM.
            The shape is [n_subs, n_chls, n_cons, n_cons].
        If sub_opt=1 & chl_opt=1 & time_opt=1, return n_subs*n_chls*(int((n_ts-time_win)/time_step)+1) RDM.
            The shape is [n_subs, n_chls, int((n_ts-time_win)/time_step)+1, n_cons, n_cons].
    Notes
    -----
    Sometimes, the numbers of trials under different conditions are not same. In NeuroRA, we recommend users to average
    the trials under a same condition firstly in this situation. Thus, the shape of input (EEG_data) should be
    [n_cons, n_subs, 1, n_chls, n_ts].
    """

    if len(np.shape(EEG_data)) != 5:

        print("The shape of input for eegRDM() function must be [n_cons, n_subs, n_trials, n_chls, n_ts].\n")

        return "Invalid input!"

    # get the number of conditions, subjects, trials, channels and time points
    cons, subs, trials, chls, ts = np.shape(EEG_data)

    if time_opt == 1:

        print("\nComputing RDMs")

        # the time-points for calculating RDM
        ts = int((ts - time_win) / time_step) + 1

        # initialize the data for calculating the RDM
        data = np.zeros([subs, chls, ts, cons, time_win], dtype=np.float64)

        # assignment
        for i in range(subs):
            for j in range(chls):
                for k in range(ts):
                    for l in range(cons):
                        for m in range(time_win):
                            # average the trials
                            data[i, j, k, l, m] = np.average(EEG_data[l, i, :, j, k * time_step + m])

        if chl_opt == 1:

            total = subs*chls*ts

            # initialize the RDMs
            rdms = np.zeros([subs, chls, ts, cons, cons], dtype=np.float64)

            # calculate the values in RDMs
            for i in range(subs):
                for j in range(chls):
                    for k in range(ts):

                        # show the progressbar
                        percent = (i*chls*ts+j*ts+k) / total * 100
                        show_progressbar("Calculating", percent)

                        for l in range(cons):
                            for m in range(cons):
                                if method is 'correlation':
                                    # calculate the Pearson Coefficient
                                    r = pearsonr(data[i, j, k, l], data[i, j, k, m])[0]
                                    # calculate the dissimilarity
                                    if abs == True:
                                        rdms[i, j, k, l, m] = limtozero(1 - np.abs(r))
                                    else:
                                        rdms[i, j, k, l, m] = limtozero(1 - r)
                                elif method is 'euclidean':
                                    rdms[i, j, k, l, m] = np.linalg.norm(data[i, j, k, l] - data[i, j, k, m])
                                elif method is 'mahalanobis':
                                    X = np.transpose(np.vstack((data[i, j, k, l], data[i, j, k, m])), (1, 0))
                                    X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
                                    rdms[i, j, k, l, m] = np.linalg.norm(X[:, 0] - X[:, 1])
                        if method is 'euclidean' or method is 'mahalanobis':
                            max = np.max(rdms[i, j, k])
                            min = np.min(rdms[i, j, k])
                            rdms[i, j, k] = (rdms[i, j, k] - min) / (max - min)

            # time_opt=1 & chl_opt=1 & sub_opt=1
            if sub_opt == 1:

                print("\nRDMs computing finished!")

                return rdms

            # time_opt=1 & chl_opt=1 & sub_opt=0
            if sub_opt == 0:

                rdms = np.average(rdms, axis=0)

                print("\nRDMs computing finished!")

                return rdms

        # if chl_opt = 0

        data = np.transpose(data, (0, 2, 3, 4, 1))
        data = np.reshape(data, [subs, ts, cons, time_win*chls])

        rdms = np.zeros([subs, ts, cons, cons], dtype=np.float64)

        total = subs * ts

        # calculate the values in RDMs
        for i in range(subs):
            for k in range(ts):

                # show the progressbar
                percent = (i * ts + j) / total * 100
                show_progressbar("Calculating", percent)

                for l in range(cons):
                    for m in range(cons):
                        if method is 'correlation':
                            # calculate the Pearson Coefficient
                            r = pearsonr(data[i, k, l], data[i, k, m])[0]
                            # calculate the dissimilarity
                            if abs == True:
                                rdms[i, k, l, m] = limtozero(1 - np.abs(r))
                            else:
                                rdms[i, k, l, m] = limtozero(1 - r)
                        elif method is 'euclidean':
                            rdms[i, k, l, m] = np.linalg.norm(data[i, k, l] - data[i, k, m])
                        elif method is 'mahalanobis':
                            X = np.transpose(np.vstack((data[i, k, l], data[i, k, m])), (1, 0))
                            X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
                            rdms[i, k, l, m] = np.linalg.norm(X[:, 0] - X[:, 1])
                if method is 'euclidean' or method is 'mahalanobis':
                    max = np.max(rdms[i, k])
                    min = np.min(rdms[i, k])
                    rdms[i, k] = (rdms[i, k] - min) / (max - min)

        # time_opt=1 & chl_opt=0 & sub_opt=1
        if sub_opt == 1:

            print("\nRDMs computing finished!")

            return rdms

        # time_opt=1 & chl_opt=0 & sub_opt=0
        if sub_opt == 0:

            rdms = np.average(rdms, axis=0)

            print("\nRDM computing finished!")

            return rdms


    # if time_opt = 0

    if chl_opt == 1:

        print("\nComputing RDMs")

        # average the trials
        data = np.average(EEG_data, axis=2)

        # initialize the RDMs
        rdms = np.zeros([subs, chls, cons, cons], dtype=np.float64)

        total = subs * chls

        # calculate the values in RDMs
        for i in range(subs):
            for j in range(chls):

                # show the progressbar
                percent = (i * chls + j) / total * 100
                show_progressbar("Calculating", percent)

                for k in range(cons):
                    for l in range(cons):
                        if method is 'correlation':
                            # calculate the Pearson Coefficient
                            r = pearsonr(data[k, i, j], data[l, i, j])[0]
                            # calculate the dissimilarity
                            if abs == True:
                                rdms[i, j, k, l] = limtozero(1 - np.abs(r))
                            else:
                                rdms[i, j, k, l] = limtozero(1 - r)
                        elif method is 'euclidean':
                            rdms[i, j, k, l] = np.linalg.norm(data[k, i, j] - data[k, i, j])
                        elif method is 'mahalanobis':
                            X = np.transpose(np.vstack((data[k, i, j], data[l, i, j])), (1, 0))
                            X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
                            rdms[i, j, k, l] = np.linalg.norm(X[:, 0] - X[:, 1])
                if method is 'euclidean' or method is 'mahalanobis':
                    max = np.max(rdms[i, j])
                    min = np.min(rdms[i, j])
                    rdms[i, j] = (rdms[i, j] - min) / (max - min)

        # time_opt=0 & chl_opt=1 & sub_opt=1
        if sub_opt == 1:

            print("\nRDM computing finished!")

            return rdms

        # time_opt=0 & chl_opt=1 & sub_opt=0
        if sub_opt == 0:

            rdms = np.average(rdms, axis=0)

            print("\nRDM computing finished!")

            return rdms

    # if chl_opt = 0

    if sub_opt == 1:

        print("\nComputing RDMs")

    else:

        print("\nComputing RDM")

    # average the trials
    data = np.average(EEG_data, axis=2)

    # flatten the data for different calculating conditions
    data = np.reshape(data, [cons, subs, chls * ts])

    # initialize the RDMs
    rdms = np.zeros([subs, cons, cons], dtype=np.float64)

    # calculate the values in RDMs
    for i in range(subs):
        for j in range(cons):
            for k in range(cons):
                if method is 'correlation':
                    # calculate the Pearson Coefficient
                    r = pearsonr(data[j, i], data[k, i])[0]
                    # calculate the dissimilarity
                    if abs is True:
                        rdms[i, j, k] = limtozero(1 - np.abs(r))
                    else:
                        rdms[i, j, k] = limtozero(1 - r)
                elif method is 'euclidean':
                    rdms[i, j, k] = np.linalg.norm(data[j, i] - data[k, i])
                elif method is 'mahalanobis':
                    X = np.transpose(np.vstack((data[j, i], data[k, i])), (1, 0))
                    X = np.dot(X, np.linalg.inv(np.cov(X, rowvar=False)))
                    rdms[i, j, k] = np.linalg.norm(X[:, 0] - X[:, 1])
        if method is 'euclidean' or method is 'mahalanobis':
            max = np.max(rdms[i])
            min = np.min(rdms[i])
            rdms[i] = (rdms[i] - min) / (max - min)

    if sub_opt == 1:

        print("\nRDMs computing finished!")

        return rdms

    if sub_opt == 0:

        rdms = np.average(rdms, axis=0)

        print("\nRDM computing finished!")

        return rdms

subs = ["sub01", "sub02", "sub03", "sub05", "sub06", 'sub07','sub10','sub11',
'sub12','sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19','sub20','sub21',
'sub22','sub23','sub24','sub25','sub26','sub27','sub28','sub29','sub30','sub31']
nsubs = len(subs)
data = np.zeros([3, nsubs, 41, 60, 851], dtype=np.float64)

index = 0
for sub in subs:
    f = h5py.File("data_for_RSA/ERP/"+sub+".h5", "r")
    subdata = np.array(f["intents"])
    f.close()
    data[:, index, :, :, :] = subdata
    index = index + 1

#eegrdms = eegRDM(data, sub_opt=1, time_opt=1, time_win=5, time_step=5)
eegrdms = eegRDM(data, sub_opt=0, chl_opt=1, time_opt=1, time_win=5, time_step=5, method="correlation", abs=True)
f = h5py.File("rdms/ERP_28subs_bycon.h5", "w")
f.create_dataset("intents", data=eegrdms)
f.close()

