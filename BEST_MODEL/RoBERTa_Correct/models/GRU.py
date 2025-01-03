import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

# 指定本地模型路径 这两个是服务bert用的
model_path = "roBERTa-large"


class SpeechRecognitionModel(nn.Module):
    def __init__(self, input_size_1, hidden_size, output_size, args):
        super(SpeechRecognitionModel, self).__init__()
        self.feature_extractor = RobertaModel.from_pretrained(model_path)
        self.out_layer = nn.Linear(1024, args.out_class)

        # 冻结 SSL 模型的所有参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
            
    def forward(self, input_ids,attention_mask):
        # 提取特征
        input_ids = input_ids.to(torch.int).to(torch.device("cuda"))
        output = self.feature_extractor(input_ids, attention_mask)
        features_text = output.last_hidden_state
        mean_features = torch.mean(features_text, dim=1)       
        logits = self.out_layer(mean_features)
        return logits