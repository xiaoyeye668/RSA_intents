import numpy as np
import scipy.io as sio
import h5py
from collections import defaultdict
import mne
import pandas as pd

def read_from_mat(file_path):
    raw_data = sio.loadmat(file_path)
    data = raw_data['data_2visual']
    chas_arr = data['label'][0][0]
    fsample_arr = data['fsample'][0][0]
    features_arr = data['trial'][0][0].T
    label_arr = data['trialinfo'][0][0]
    return chas_arr,fsample_arr,features_arr,label_arr

preprocessed_path = 'enroll_data'
# 被试id
subs = ["sub01", "sub02", "sub03", "sub05", "sub06", 'sub07','sub10','sub11',
'sub12','sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19','sub20','sub21',
'sub22','sub23','sub24','sub25','sub26','sub27','sub28','sub29','sub30','sub31']
#subs = ["sub02"]
event_dic = {'0': 1, '11': 2, '12': 3, '13': 4, '51': 5, '52': 6}

df_voice = pd.read_excel('./behave_28subs_new.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
#X = np.average(df_voice[df_voice['subject']=='JG'].loc[:,search_list].values, axis=0)
df_index = pd.read_excel('./index_order.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
index_order = df_index['trial_ID'].values
#print(index_order)
for sub in subs:
   subid = int(sub[3:])
   trial_ID = df_voice[df_voice['subject']==subid].loc[:,'trial_ID'].values
   index_order = [i for i in index_order if i in trial_ID]
   #print(len(index_order))
print(len(index_order), index_order)
for sub in subs:
   dictmp = defaultdict()
   data_path = preprocessed_path + '/cleandata_' + sub + '.mat'
   chas_arr,fsample_arr, features_arr, label_arr = read_from_mat(data_path)
   #print(features_arr.shape, label_arr.shape)
   subid = int(sub[3:])
   trial_ID = df_voice[df_voice['subject']==subid].loc[:,'trial_ID'].values
   # data.shape: n_cons, n_trials, n_channels, n_times
   #subdata = np.zeros([3, 41, 60, 851], dtype=np.float64)
   subdata = np.zeros([73, 1, 60, 851], dtype=np.float64) 
   #Epochs object from numpy array.
   data = features_arr
   for i in range(len(trial_ID)):
      dictmp[trial_ID[i]] = features_arr[i][0]

   subdata_index = 0
   for order in index_order:
      if order in dictmp.keys():
         subdata[subdata_index,:,:,:] = dictmp[order]
         subdata_index += 1
   print(subdata_index, index_order[0],index_order[30],index_order[-1])
   '''
   ch_names = np.loadtxt('Xiaoqing60_AF7.txt', dtype=str, usecols=-1)
   ch_names = list(ch_names)
   ch_names.remove('COMNT')
   ch_names.remove('SCALE') 
   sfreq = 500 #采样率
   info = mne.create_info(ch_names, sfreq, ch_types = "eeg") #创建信号的信息
    #event_dic = {'0': 1, '11': 2, '12': 3, '13': 4, '51': 5, '52': 6}
    #events.shape (n_events, 3) [timestamp,0,label]
   #events = np.zeros([y.shape[0],3])
   events = np.zeros([10,3])
   events[:,-1] = trial_ID[:10]
   events[:,0] = np.array([int(i) for i in range(10)])
   events=events.astype(int)
   event_id = dict(JG=2, MM=3, JY=4)
   # 定义输出顺序
   order = index_order[:10]
   # 选择和排序事件
   print(events,event_id)
   selected_events = mne.pick_events(events, include=order)
   epochs = mne.EpochsArray(data, info, events=events, tmin=0, event_id=event_id)
   # 根据选择和排序的事件创建新的Epochs对象
   selected_epochs = mne.Epochs(raw, selected_events, event_id, tmin, tmax, baseline=baseline, picks=picks, preload=True)

   # 查看输出的数据顺序
   print(selected_epochs.events[:, 2])
   f = h5py.File("data_for_RSA/ERP_order/"+sub+".h5", "w")
   f.create_dataset("intents", data=selected_epochs.get_data())
   '''
   '''
   labelindex = np.zeros([3], dtype=np.int32)  #分别记录类别index
   #nSamples = features_arr.shape[0]
   for i in range(123):
      #'11': 2, '12': 3, '13': 4
      label = event_dic[str(label_arr[i][0])]
      if label not in dictmp:
         dictmp[label] = 1
      else:
         dictmp[label] += 1
      if labelindex[label-2] < 41:
         subdata[label-2, labelindex[label-2]] = features_arr[i][0] #-0.2-1.5s
         #subdata[label-2, labelindex[label-2]] = features_arr[i][0][:,:701]  #-0.2-1.2s
         labelindex[label-2] += 1
   #print(sub,dictmp)
   #print(labelindex)
   '''
   f = h5py.File("data_for_RSA/ERP_order/"+sub+".h5", "w")
   f.create_dataset("intents", data=subdata)
   f.close()