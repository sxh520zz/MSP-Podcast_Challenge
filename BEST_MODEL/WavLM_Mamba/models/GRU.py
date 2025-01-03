import torch
import torch.nn as nn
from transformers import WavLMModel
from mamba_ssm import Mamba
import torch.nn.functional as F

class Mamba_block(nn.Module):
    def __init__(self, input_size, args): 
        super(Mamba_block, self).__init__()
        self.bigru = Utt_net_1(input_size, args)
        self.Mamba  = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=512, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
            )
        self.FC = nn.Linear(512, input_size)
        self.layer_norm = nn.LayerNorm(input_size)
    def forward(self, features):
        hidden_features = self.bigru(features)
        hidden_features = self.Mamba(hidden_features)
        hidden_features = self.FC(hidden_features)
        tensor = torch.add(features,hidden_features)
        tensor = self.layer_norm(tensor)
        return tensor
    
class Utt_net_1(nn.Module):
    def __init__(self, input_size, args):
        super(Utt_net_1, self).__init__()
        self.hidden_dim = args.hidden_layer
        #  dropout
        self.dropout = nn.Dropout(args.dropout)
        # gru
        self.attention = args.attention
        self.num_layers = args.dia_layers
        self.bigru = nn.GRU(input_size, self.hidden_dim, dropout=args.dropout, 
                            batch_first=True, num_layers=self.num_layers, bidirectional=True)
        if self.attention:
            self.matchatt = MatchingAttention(2 * 256, 2 * 256, att_type='dot')

    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.bigru(U)
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=None)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            emotions = att_emotions
            #hidden = F.relu(self.linear(att_emotions))
        else:
            #hidden = F.relu(self.linear(emotions))
            emotions = emotions
        return emotions

class MatchingAttention(nn.Module):
    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            #torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2) # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
            #import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim
        return attn_pool, alpha
       
class SpeechRecognitionModel(nn.Module):
    def __init__(self, args):
        super(SpeechRecognitionModel, self).__init__()
        self.feature_extractor = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-large")
        self.out_layer = nn.Linear(1024, args.out_class)
        self.Mamba_speech  = Mamba_block(1024, args)

        # 冻结 SSL 模型的所有参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
            
    def forward(self, input_waveform):
        # 提取特征
        features = self.feature_extractor(input_waveform).last_hidden_state
        Mamba_features = self.Mamba_speech(features)
        mean_features = torch.mean(Mamba_features, dim=1)
        logits = self.out_layer(mean_features)
        return logits, logits