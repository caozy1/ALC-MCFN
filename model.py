
from sklearn.model_selection import train_test_split

import warnings
import seaborn as sns
import matplotlib
from sklearn.metrics import accuracy_score, recall_score,f1_score,roc_curve, auc,precision_score ,classification_report,confusion_matrix

matplotlib.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
# 忽略警告
warnings.filterwarnings('ignore')

import torch
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.WARNING)



number_class=10

# 定义模型，获取bert输出并加上全连接层实现分类任务
class Model_bert(torch.nn.Module):
    def __init__(self,pretrained):
        super().__init__()

        self.fc = torch.nn.Linear(768, number_class)
        self.pretrained=pretrained
    def forward(self, input_ids):
        with torch.no_grad():
            out = self.pretrained(**input_ids)

        #out = self.pretrained(**input_ids)
        out = self.fc(out.last_hidden_state[:, 0])


        return out

import torch.nn.functional as F

# text cnn 模型
class Model_cnn(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()

        self.pretrained = pretrained  # BERT 模型
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 768))  # 3-gram卷积层
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, 768))  # 4-gram卷积层
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(5, 768))  # 5-gram卷积层
        self.fc = torch.nn.Linear(300, number_class)  # 全连接层，输入是三个卷积层的输出维度总和
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, input_ids, attention_mask=None):
        # 使用 BERT 提取特征
        with torch.no_grad():
            bert_output = self.pretrained(**input_ids)

         # 获取 BERT 的最后隐藏状态
        hidden_states = bert_output.last_hidden_state  # shape: (batch_size, seq_len, 768)

        # 添加一个通道维度以适配卷积层
        hidden_states = hidden_states.unsqueeze(1)  # shape: (batch_size, 1, seq_len, 768)

        # 通过卷积层
        conv1_out = F.relu(self.conv1(hidden_states)).squeeze(3)  # shape: (batch_size, 100, seq_len-2)
        conv2_out = F.relu(self.conv2(hidden_states)).squeeze(3)  # shape: (batch_size, 100, seq_len-3)
        conv3_out = F.relu(self.conv3(hidden_states)).squeeze(3)  # shape: (batch_size, 100, seq_len-4)

        # 池化操作
        pooled1 = F.max_pool1d(conv1_out, conv1_out.size(2)).squeeze(2)  # shape: (batch_size, 100)
        pooled2 = F.max_pool1d(conv2_out, conv2_out.size(2)).squeeze(2)  # shape: (batch_size, 100)
        pooled3 = F.max_pool1d(conv3_out, conv3_out.size(2)).squeeze(2)  # shape: (batch_size, 100)

        # 拼接池化结果
        combined = torch.cat((pooled1, pooled2, pooled3), dim=1)  # shape: (batch_size, 300)
        combined = self.dropout(combined)
        # 通过全连接层得到最终输出
        out = self.fc(combined)

        return out
    
# tcn模型
class TCNBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation):
        super().__init__()
        self.conv = torch.nn.Conv1d(input_size, output_size, kernel_size=kernel_size, 
                                     padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))
    
    
class Model_tcn(torch.nn.Module):
    def __init__(self, pretrained, num_layers=1, num_heads=1, hidden_dim=512, dropout_rate=0.5):
        super().__init__()

        self.pretrained = pretrained  # BERT 模型
        self.tcn1 = TCNBlock(input_size=768, output_size=256, kernel_size=3, dilation=1)
        self.tcn2 = TCNBlock(input_size=256, output_size=256, kernel_size=3, dilation=2)
        self.tcn3 = TCNBlock(input_size=256, output_size=256, kernel_size=3, dilation=4)
        self.text_cnn = TextCNN(dropout_rate=dropout_rate)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(256, number_class)  # 300 是 TextCNN 输出，256 是 TCN 输出

    def forward(self, input_ids, attention_mask=None):
        # 使用 BERT 提取特征
        with torch.no_grad():
            bert_output = self.pretrained(**input_ids)

        # 获取 BERT 的最后隐藏状态
        hidden_states = bert_output.last_hidden_state  # shape: (batch_size, seq_len, 768)

        # TCN 特征提取
        tcn_out1 = self.tcn1(hidden_states.permute(0, 2, 1))  # 转置为 (batch_size, 768, seq_len)
        tcn_out2 = self.tcn2(tcn_out1)
        tcn_out3 = self.tcn3(tcn_out2)
        #tcn_out3 =tcn_out3.mean(dim=2)
        tcn_out3 = self.pool(tcn_out3).squeeze(-1) 

        # 通过全连接层得到最终输出
        out = self.fc(tcn_out3)

        return out

class TextCNN(torch.nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 100, (3, 768))  # 3-gram卷积
        self.conv2 = torch.nn.Conv2d(1, 100, (4, 768))  # 4-gram卷积
        self.conv3 = torch.nn.Conv2d(1, 100, (5, 768))  # 5-gram卷积
        self.dropout = torch.nn.Dropout(dropout_rate)  # Dropout层

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        conv1_out = F.relu(self.conv1(x)).squeeze(3)  # shape: (batch_size, 100, seq_len-2)
        conv2_out = F.relu(self.conv2(x)).squeeze(3)  # shape: (batch_size, 100, seq_len-3)
        conv3_out = F.relu(self.conv3(x)).squeeze(3)  # shape: (batch_size, 100, seq_len-4)

        pooled1 = F.max_pool1d(conv1_out, conv1_out.size(2)).squeeze(2)  # shape: (batch_size, 100)
        pooled2 = F.max_pool1d(conv2_out, conv2_out.size(2)).squeeze(2)  # shape: (batch_size, 100)
        pooled3 = F.max_pool1d(conv3_out, conv3_out.size(2)).squeeze(2)  # shape: (batch_size, 100)

        combined = torch.cat((pooled1, pooled2, pooled3), dim=1)  # shape: (batch_size, 300)
        combined = self.dropout(combined)  # 应用 Dropout
        return combined
    
from torch import nn
    

# 融合模型（创新模型）
class Model(nn.Module):
    def __init__(self, pretrained, num_layers=1, num_heads=1, hidden_dim=512, dropout_rate=0.5):
        super().__init__()

        self.pretrained = pretrained  # BERT 模型
        self.tcn1 = TCNBlock(input_size=768, output_size=256, kernel_size=3, dilation=1)
        self.tcn2 = TCNBlock(input_size=256, output_size=256, kernel_size=3, dilation=2)
        self.tcn3 = TCNBlock(input_size=256, output_size=256, kernel_size=3, dilation=4)
        self.text_cnn = TextCNN(dropout_rate=dropout_rate)
        self.lstm = nn.LSTM(input_size=556, hidden_size=556, batch_first=True)

        # Transformer Encoder Layer
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=556, nhead=num_heads)
        self.fc = nn.Linear(556, number_class)

    def forward(self, input_ids, attention_mask=None):
        # 使用 BERT 提取特征
        with torch.no_grad():
            bert_output = self.pretrained(**input_ids)

        # 获取 BERT 的最后隐藏状态
        hidden_states = bert_output.last_hidden_state  # shape: (batch_size, seq_len, 768)

        # TCN 特征提取
        tcn_out1 = self.tcn1(hidden_states.permute(0, 2, 1))  # 转置为 (batch_size, 768, seq_len)
        tcn_out2 = self.tcn2(tcn_out1)
        tcn_out3 = self.tcn3(tcn_out2)
        tcn_out3 = tcn_out3.mean(dim=2)

        # TextCNN 特征提取
        textcnn_out = self.text_cnn(hidden_states)

        # 特征融合和跳跃连接
        combined_features = torch.cat((tcn_out3, textcnn_out), dim=1)  # shape: (batch_size, 556)
        lstm_out, _ = self.lstm(combined_features.unsqueeze(1))
        #print( lstm_out.squeeze(1).shape)
        #print(combined_features.shape)
        
        skip_connection = lstm_out.squeeze(1) + combined_features
        #print(skip_connection.unsqueeze(1).permute(1, 0, 2).shape)
        # 通过 Transformer Encoder 处理融合特征
        transformer_out = self.transformer_encoder(skip_connection.unsqueeze(1).permute(1, 0, 2))
        transformer_out = transformer_out[-1]  # shape: (batch_size, feature_dim)

        # 通过全连接层得到最终输出
        out = self.fc(transformer_out)
        return out




