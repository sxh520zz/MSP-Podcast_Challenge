import torch
import torch.nn as nn
'''
class SpeechRecognitionModel(nn.Module):
    def __init__(self, args):
        super(SpeechRecognitionModel, self).__init__()
        
        # 定义多个线性层
        self.linear1 = nn.Linear(1280, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.out_layer = nn.Linear(128, args.out_class)
        
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, input_waveform):
        x = self.linear1(input_waveform)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        logits = self.out_layer(x)
        return logits
'''
class SpeechRecognitionModel(nn.Module):
    def __init__(self, args):
        super(SpeechRecognitionModel, self).__init__()
        self.out_layer = nn.Linear(1280, args.out_class)
    def forward(self, input_waveform):
        logits = self.out_layer(input_waveform)
        return logits
