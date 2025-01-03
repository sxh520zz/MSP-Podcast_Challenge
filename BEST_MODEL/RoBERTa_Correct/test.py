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
import torch.optim as optim
from utils import Get_data
from torch.autograd import Variable
from models import SpeechRecognitionModel
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold


with open('/home/shixiaohan-toda/Desktop/Challenge/SHI/Train_data_RoBerta_Correct.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=16, metavar='N')
parser.add_argument('--log_interval', type=int, default=100, metavar='N')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--attention', action='store_true', default=False, help='use attention on top of lstm')
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=8)
parser.add_argument('--utt_insize_1', type=int, default=1024)
parser.add_argument('--utt_insize_3', type=int, default=128)
args = parser.parse_args()

torch.manual_seed(args.seed)


def Train(epoch):
    train_loss = 0
    model.train()
    for batch_idx, (data_1, data_2, target, ids, att)  in enumerate(train_loader):
        if args.cuda:
            data_1, data_2, target, ids, att = data_1.cuda(), data_2.cuda(), target.cuda(), ids.cuda(), att.cuda()
        data_1, data_2, target, ids, att = Variable(data_1), Variable(data_2), Variable(target),Variable(ids),Variable(att)
        model_optim.zero_grad()
        target = target.squeeze()
        #data_1 = data_1.squeeze()
        #data_2 = data_2.squeeze()

        line_out = model(ids, att)

        loss_fn = torch.nn.CrossEntropyLoss()  # 使用权重
        loss = loss_fn(line_out, target.long())

        loss.backward()

        model_optim.step()

        train_loss += loss

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval
            ))
            train_loss = 0


def Test():
    model.eval()
    label_pre = []
    label_true = []
    Fea_true = []
    with torch.no_grad():
        for batch_idx, (data_1, data_2, target, ids, att) in enumerate(test_loader):
            if args.cuda:
                data_1, data_2, target, ids, att = data_1.cuda(), data_2.cuda(), target.cuda(), ids.cuda(), att.cuda()
            data_1, data_2, target, ids, att = Variable(data_1), Variable(data_2), Variable(target),Variable(ids),Variable(att)
            target = target.squeeze()
            model_optim.zero_grad()

            #data_1 = data_1.squeeze()
            #data_2 = data_2.squeeze()

            line_out = model(ids, att)
            output = torch.argmax(line_out, dim=1)
            Fea_true.extend(line_out.cpu().data.numpy())
            label_true.extend(target.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())
        accuracy_recall = metrics.f1_score(label_true, label_pre, average='macro')
        accuracy_f1 = metrics.f1_score(label_true, label_pre, average='macro')
        CM_test = confusion_matrix(label_true, label_pre)
        print(accuracy_recall)
        print(accuracy_f1)
        print(CM_test)
    return accuracy_recall, label_pre, label_true, Fea_true


Final_result = []
Fineal_f1 = []
result_label = []


train = data[0]
test = data[1]

print(len(train))
print(len(test))
train_loader, test_loader, input_test_data_id, input_test_label_org,class_weights = Get_data(data, train, test, args)
model = SpeechRecognitionModel(args.utt_insize_1, args.hidden_layer, args.out_class, args)
name_1 = '/home/shixiaohan-toda/Desktop/Challenge/SHI/Baseline_Text_SSL/model_Correct.pkl'
model.load_state_dict(torch.load(name_1))

if args.cuda:
    model = model.cuda()
    class_weights = class_weights.cuda()
lr = args.lr
model_optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
model_optim = optim.Adam(model.parameters(), lr=lr)
recall = 0
for epoch in range(1, args.epochs + 1):
    accuracy_recall, pre_label, true_label, label_pre = Test()
    if (accuracy_recall > recall):
        predict = copy.deepcopy(input_test_label_org)
        num = 0
        for x in range(len(predict)):
            predict[x] = pre_label[num]
            num = num + 1
        result_label = predict
        recall = accuracy_recall
    print("Best Result Until Now:")
    print(recall)

onegroup_result = []
for i in range(len(input_test_data_id)):
    a = {}
    a['id'] = input_test_data_id[i]
    a['Predict_label'] = result_label[i]
    a['True_label'] = input_test_label_org[i]
    a['Predict_fea'] = label_pre[i]
    onegroup_result.append(a)
Final_result.append(onegroup_result)
Fineal_f1.append(recall)

file = open('/home/shixiaohan-toda/Desktop/Challenge/SHI/Baseline_Text_SSL/Final_result_Correct.pickle', 'wb')
pickle.dump(Final_result,file)
file.close()
