import os
import time
import random
import argparse
import pickle
import csv
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.utils.rnn as rmm_utils
import torch.utils.data.dataset as Dataset
import torch.optim as optim
from torch.optim import AdamW
from utils import Get_data
from torch.autograd import Variable
from models import SpeechRecognitionModel
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from transformers import Wav2Vec2Model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False

with open('/home/shixiaohan-toda/Desktop/Challenge/SHI/Ensemble/Data_test.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="RNN_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=16, metavar='N')
parser.add_argument('--log_interval', type=int, default=100, metavar='N')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optim', type=str, default='AdamW')
parser.add_argument('--attention', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=8)
parser.add_argument('--utt_insize', type=int, default=1024)
args = parser.parse_args()

torch.manual_seed(args.seed)


def Train(epoch):
    train_loss = 0
    model.train()
    for batch_idx, (data_1, data_2, target) in enumerate(train_loader):
        if args.cuda:
            data_1, data_2, target = data_1.cuda(), data_2.cuda(), target.cuda()
        data_1, data_2, target = Variable(data_1), Variable(data_2), Variable(target)
        target = target.squeeze()
        data_2 = data_2.squeeze()
        utt_optim.zero_grad()
        data_1 = data_1.squeeze()
        utt_out = model(data_1)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)  # 使用权重
        loss = loss_fn(utt_out, data_2.long())
        loss.backward()
        utt_optim.step()
        train_loss += loss.item()  # 注意调整为.item()获取实际的loss数值

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss / args.log_interval
            ))
            train_loss = 0

def Test():
    model.eval()
    label_pre = []
    label_true = []
    label_pre_1 = []
    label_true_1 = []

    with torch.no_grad():
        for batch_idx, (data_1, data_2, target)  in enumerate(test_loader):
            if args.cuda:
                data_1, data_2, target = data_1.cuda(), data_2.cuda(),target.cuda()
            data_1, data_2, target = Variable(data_1), Variable(data_2), Variable(target)
            target = target.squeeze(1)
            data_2 = data_2.squeeze(1)
            utt_optim.zero_grad()
            data_1 = data_1.squeeze(1)
            data_1 = data_1.squeeze(1)
            utt_out = model(data_1)
            output = torch.argmax(utt_out, dim=1)

            label_pre.extend(output.cpu().data.numpy())

    return label_pre

Final_result = []
Fineal_f1 = []
result_label = []


train = data[0]
test = data[0]

print(len(train))
print(len(test))
train_loader, test_loader, input_test_data_id, input_test_label_org,class_weights = Get_data(data, train, test, args)
model = SpeechRecognitionModel(args)
name_1 = '/home/shixiaohan-toda/Desktop/Challenge/SHI/Ensemble/model.pkl'
model.load_state_dict(torch.load(name_1))

if args.cuda:
    model = model.cuda()
    class_weights = class_weights.cuda()
lr = args.lr
utt_optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
utt_optim = optim.Adam(model.parameters(), lr=lr)

pre_label = Test()
onegroup_result = []
for i in range(len(input_test_data_id)):
    a = {}
    a['id'] = input_test_data_id[i]
    a['Predict_label'] = pre_label[i]
    onegroup_result.append(a)
Final_result.append(onegroup_result)


# Define the emotion mapping
emotion_mapping = {
    "Angry": "A",
    "Sad": "S",
    "Happy": "H",
    "Surprise": "U",
    "Fear": "F",
    "Disgust": "D",
    "Contempt": "C",
    "Neutral": "N"
}

# Prepare the data for CSV
csv_data = []
for item in onegroup_result:
    id = item['id']
    label = item['Predict_label']
    
    # Map the label to the corresponding emotion class
    emotion_class = list(emotion_mapping.keys())[label]
    emo_class = emotion_mapping[emotion_class]
    
    # Prepare the row with filename and emotion class
    filename = f"{id}.wav"
    csv_data.append([filename, emo_class])

# Write the data to a CSV file
csv_file = '/home/shixiaohan-toda/Desktop/Challenge/SHI/Ensemble/output_emotion_classes.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['FileName', 'EmoClass'])
    writer.writerows(csv_data)

print(f"CSV file saved as {csv_file}")