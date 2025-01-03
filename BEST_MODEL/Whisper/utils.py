import os
import time
import random
import argparse
import pickle
import copy
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.utils.rnn as rmm_utils
import torch.utils.data.dataset as Dataset
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold


class subDataset(Dataset.Dataset):
    def __init__(self,Data_1,Data_2,Label):
        self.Data_1 = Data_1
        self.Data_2 = Data_2
        self.Label = Label
    def __len__(self):
        return len(self.Data_1)
    def __getitem__(self, item):
        data_1 = torch.Tensor(self.Data_1[item])
        data_2 = torch.Tensor(self.Data_2[item])
        label = torch.Tensor(self.Label[item])
        return data_1,data_2,label

def STD(input_fea):
    data_1 = []
    for i in range(len(input_fea)):
        data_1.append(input_fea[i])
    a = []
    for i in range(len(data_1)):
        a.extend(data_1[i])
    scaler_1 = preprocessing.StandardScaler().fit(a)
    #print(scaler_1.mean_)
    #print(scaler_1.var_)
    for i in range(len(input_fea)):
        input_fea[i]   = scaler_1.transform(input_fea[i])
    return input_fea

def Feature(data,args):
    input_data_spec = []
    for i in range(len(data)):
        input_data_spec.append(np.array(data[i]['whisper']).reshape(1,-1))
    #input_data_spec = STD(input_data_spec)

    '''
    a = [0.0 for i in range(args.utt_insize)]
    a = np.array(a)
    input_data_spec_CNN = []
    
    for i in range(len(input_data_spec)):
        ha = []
        if(len(input_data_spec[i]) < 300):
            for z in range(len(input_data_spec[i])):
                ha.append(np.array(input_data_spec[i][z]))
            len_zero = 300 - len(input_data_spec[i])
            for x in range(len_zero):
                ha.append(np.array(a))
        if(len(input_data_spec[i]) >= 300):
            for z in range(len(input_data_spec[i])):
                if(z < 300):
                    ha.append(np.array(input_data_spec[i][z]))
        ha = np.array(ha)
        input_data_spec_CNN.append(ha)
    input_data_spec_CNN = STD(input_data_spec_CNN)
    '''

    input_label = []
    for i in range(len(data)):
        input_label.append(data[i]['label'])
    input_label_1 = []
    for i in range(len(data)):
        input_label_1.append(data[i]['label'])
    input_data_id= []
    for i in range(len(data)):
        input_data_id.append(data[i]['id'])
    input_label_org = []
    for i in range(len(data)):
        input_label_org.append(data[i]['label'])
    return input_data_spec,input_data_spec,input_label,input_label_1,input_data_id,input_label_org

def Get_data(data,train_data,test_data,args):

    input_train_data_spec,input_train_data_spec_CNN,input_train_label,input_train_label_1,_,_ = Feature(train_data,args)
    input_test_data_spec,input_test_data_spec_CNN, input_test_label,input_test_label_1,input_test_data_id,input_test_label_org = Feature(test_data,args)

    # 将标签转换为numpy数组以便计算权重
    labels_array = np.array(input_train_label).reshape(-1)
    
    # 计算类别权重
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    #label = np.array(input_train_label, dype='int64').reshape(-1,1)
    label = np.array(input_train_label).reshape(-1, 1)
    label_test = np.array(input_test_label).reshape(-1,1)
    label_1 = np.array(input_train_label_1).reshape(-1, 1)
    label_test_1 = np.array(input_test_label_1).reshape(-1,1)
    train_dataset = subDataset(input_train_data_spec_CNN,label,label_1)
    test_dataset = subDataset(input_test_data_spec_CNN,label_test,label_test_1)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,drop_last=False, shuffle=False)
    return train_loader,test_loader,input_test_data_id,input_test_label_org,class_weights_tensor