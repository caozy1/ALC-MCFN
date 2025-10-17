
from utils import *
from model import *
matplotlib.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
# 忽略警告
warnings.filterwarnings('ignore')

import pickle
import os

import torch
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import logging

logging.disable(logging.WARNING)



df2 = pd.read_excel('train.xlsx')

df2.shape

# In[4]:


df2['label'].value_counts()

# In[18]:


data = df2[df2['label'].isin([2, 16, 15, 5, 9, 3, 12, 14, 10, 4])].copy()
# 创建标签映射字典
label_mapping = {2: 0, 16: 1, 15: 2, 5: 3, 9: 4, 3: 5, 12: 6, 14: 7, 10: 8, 4: 9}

# 应用标签映射
data['label'] = data['label'].map(label_mapping)

# In[19]:


import random
import pandas as pd

df2_o, df21 = data.copy(), data.copy()
print(df21['label'].value_counts())

# In[20]:


from transformers import BertTokenizer

# 加载字典和中文分词工具
# 将每个汉字作为一个单词
token = BertTokenizer.from_pretrained('bert-base-chinese')  # 从预训练模型加载BERT中文分词器

token  # 查看分词器对象





train_loader, val_loader, test_loader = pre_process(data)
len(train_loader)

# In[23]:


from transformers import BertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备为GPU（如果可用）否则为CPU

# 加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese').to(device)  # 从预训练模型加载BERT中文模型并移动到设备上

# 我们只需要使用它而不是训练它
for param in pretrained.parameters():
    param.requires_grad_(False)  # 冻结模型参数，不进行梯度更新

# 查看数据
for i, (input_ids, labels) in enumerate(train_loader):
    break

print(len(train_loader))
# 测试预训练数据
out = pretrained(**input_ids.to(device))  # 使用预训练模型进行前向传播

# 16批量大小，128最大长度，768是编码维度
out.last_hidden_state.shape  # 查看最后一层隐藏状态的形状

# # 模型框架构建

# In[24]:


number_class = 10



# # 融合模型训练

# In[42]:


torch.manual_seed(42)

# 如果使用 GPU 还需要设置 CUDA 的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # 如果有多个 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese').to(device)  # 从预训练模型加载BERT中文模型并移动到设备上

# 我们只需要使用它而不是训练它
for param in pretrained.parameters():
    param.requires_grad_(False)  # 冻结模型参数，不进行梯度更新

model = Model(pretrained)
num_epochs = 5
learning_rate = 1e-4

# metrics=train_bert(model, train_loader, test_loader,test_loader,  num_epochs=num_epochs, learning_rate=learning_rate)
metrics = train_bert(model, train_loader, val_loader, test_loader, num_epochs=num_epochs, learning_rate=learning_rate)

# In[45]:


import pickle

pickle.dump(metrics, open('/kaggle/working/model_tcn_cnn_att.pkl', 'wb'))

# In[46]:


plot_metrics(metrics)

# In[47]:


all_labels, all_preds, all_probs = predict_model(model, test_loader)
evaluate_result(all_labels, all_preds, all_probs)

# # CNN模型训练

# In[48]:


torch.manual_seed(42)

# 如果使用 GPU 还需要设置 CUDA 的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # 如果有多个 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese').to(device)  # 从预训练模型加载BERT中文模型并移动到设备上

# 我们只需要使用它而不是训练它
for param in pretrained.parameters():
    param.requires_grad_(False)  # 冻结模型参数，不进行梯度更新

model_cnn = Model_cnn(pretrained)
num_epochs = 5
learning_rate = 1 * 1e-4

metrics_cnn = train_bert(model_cnn, train_loader, val_loader, test_loader, num_epochs=num_epochs,
                         learning_rate=learning_rate)

# In[49]:


import pickle

pickle.dump(metrics_cnn, open('/kaggle/working/model_cnn.pkl', 'wb'))
plot_metrics(metrics_cnn)

# In[50]:


all_labels, all_preds, all_probs = predict_model(model_cnn, test_loader)
evaluate_result(all_labels, all_preds, all_probs)

# # tcn模型训练

# In[34]:


# 加载预训练模型
torch.manual_seed(42)

# 如果使用 GPU 还需要设置 CUDA 的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # 如果有多个 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
pretrained = BertModel.from_pretrained('bert-base-chinese').to(device)  # 从预训练模型加载BERT中文模型并移动到设备上

# 我们只需要使用它而不是训练它
for param in pretrained.parameters():
    param.requires_grad_(False)  # 冻结模型参数，不进行梯度更新

model_tcn = Model_tcn(pretrained)
num_epochs = 5
learning_rate = 1e-4

metrics_tcn = train_bert(model_tcn, train_loader, val_loader, test_loader, num_epochs=num_epochs,
                         learning_rate=learning_rate)

# In[ ]:


pickle.dump(metrics_tcn, open('/kaggle/working/model_tcn.pkl', 'wb'))
plot_metrics(metrics_tcn)

# In[38]:


all_labels, all_preds, all_probs = predict_model(model_tcn, test_loader)
evaluate_result(all_labels, all_preds, all_probs)

# # bert模型训练

# In[39]:


# 加载预训练模型
torch.manual_seed(42)

# 如果使用 GPU 还需要设置 CUDA 的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # 如果有多个 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
pretrained = BertModel.from_pretrained('bert-base-chinese').to(device)  # 从预训练模型加载BERT中文模型并移动到设备上

# 我们只需要使用它而不是训练它
for param in pretrained.parameters():
    param.requires_grad_(False)  # 冻结模型参数，不进行梯度更新
num_epochs = 5
model_bert = Model_bert(pretrained)
learning_rate = 1e-4

metrics_bert = train_bert(model_bert, train_loader, val_loader, test_loader, num_epochs=num_epochs,
                          learning_rate=learning_rate)

# In[40]:


pickle.dump(metrics_bert, open('/kaggle/working/model_bert.pkl', 'wb'))
plot_metrics(metrics_bert)

# In[41]:


all_labels, all_preds, all_probs = predict_model(model_bert, test_loader)
evaluate_result(all_labels, all_preds, all_probs)

# # 模型对比结果

# In[51]:


metrics_dict = {'Model_tcn_cnn_att': metrics, 'Model_cnn': metrics_cnn,
                'Model_tcn': metrics_tcn, 'Model_bert': metrics_bert}

plot_metrics_compare(metrics_dict)

