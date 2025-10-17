
from sklearn.model_selection import train_test_split

import warnings

import seaborn as sns

import matplotlib
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, precision_score, \
    classification_report, confusion_matrix


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




def collate_fn(data):
    # 分离句子和标签
    # sents = [i[0] for i in data]
    sents = [(str(i[0][0]), str(i[0][1])) for i in data]

    # sents = [f"{i[0][0]} {i[0][1]} {i[0][2]}" for i in data]
    labels = [i[1] for i in data]
    # print(sents)
    # 编码，加载中文的token
    # 批量编码
    input_ids = token.batch_encode_plus(sents,  # 这是句子
                                        truncation='longest_first',  # 如果超过最大长度，将被截断
                                        padding='max_length',  # 填充零到最大长度
                                        max_length=256,  # 句子的最大长度
                                        return_tensors='pt',
                                        # truncation='longest_first',
                                        # 返回的类型（pytorch或tensorflow），这里用pt
                                        )

    # 修改标签的类型
    labels = torch.LongTensor(labels)

    return input_ids, labels


def pre_process(data):
    X = data[['text1', 'text3']].values
    y = data['label'].values

    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0,stratify=y)
    train_set = []
    test_set = []

    # 使用小批量训练往往会收敛到平缓的最小化
    batch_size = 64

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp)

    # 构建数据集
    train_set = [(i, j) for i, j in zip(X_train, y_train)]
    val_set = [(i, j) for i, j in zip(X_val, y_val)]
    test_set = [(i, j) for i, j in zip(X_test, y_test)]

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, collate_fn=collate_fn,
                                               shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, collate_fn=collate_fn,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, collate_fn=collate_fn,
                                              shuffle=False)

    return train_loader, val_loader, test_loader

    # return train_loader,test_loader


# In[22]:




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def train_bert(model, train_loader, test_loader, test_loader1, num_epochs=5, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to keep track of progress
    epoch_train_losses = []
    epoch_train_accuracies = []
    epoch_train_f1_scores = []  # List to track F1 scores during training
    epoch_test_losses = []
    epoch_test_accuracies = []
    epoch_test_f1_scores = []  # List to track F1 scores during testing

    epoch_test_losses2 = []
    epoch_test_accuracies2 = []
    epoch_test_f1_scores2 = []  # List to track F1 scores during testing
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0
        train_true_labels = []  # To store true labels during training
        train_predicted_labels = []  # To store predicted labels during training

        for inputs, targets in train_loader:
            # print(1)
            inputs, targets = inputs.to(device), targets.to(device)
            # print(inputs)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_train_loss += loss.item() * targets.size(0)  # Adjust loss by batch size
            _, predicted = torch.max(outputs, 1)
            total_train_samples += targets.size(0)
            total_train_correct += (predicted == targets).sum().item()

            # Collect labels for F1 score
            train_true_labels.extend(targets.cpu().numpy())
            train_predicted_labels.extend(predicted.cpu().numpy())

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / total_train_samples
        train_accuracy = 100 * total_train_correct / total_train_samples
        train_f1 = f1_score(train_true_labels, train_predicted_labels, average='macro')

        epoch_train_losses.append(avg_train_loss)
        epoch_train_accuracies.append(train_accuracy)
        epoch_train_f1_scores.append(train_f1)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Macro F1: {train_f1:.4f}')

        # Testing
        model.eval()
        total_test_loss = 0.0
        total_test_correct = 0
        total_test_samples = 0
        test_true_labels = []  # To store true labels during testing
        test_predicted_labels = []  # To store predicted labels during testing

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item() * targets.size(0)  # Adjust loss by batch size
                _, predicted = torch.max(outputs, 1)
                total_test_samples += targets.size(0)
                total_test_correct += (predicted == targets).sum().item()

                # Collect labels for F1 score
                test_true_labels.extend(targets.cpu().numpy())
                test_predicted_labels.extend(predicted.cpu().numpy())

        avg_test_loss = total_test_loss / total_test_samples
        test_accuracy = 100 * total_test_correct / total_test_samples
        test_f1 = f1_score(test_true_labels, test_predicted_labels, average='macro')

        epoch_test_losses.append(avg_test_loss)
        epoch_test_accuracies.append(test_accuracy)
        epoch_test_f1_scores.append(test_f1)

        total_test_loss2 = 0.0
        total_test_correct2 = 0
        total_test_samples2 = 0
        test_true_labels2 = []  # To store true labels during testing
        test_predicted_labels2 = []  # To store predicted labels during testing

        with torch.no_grad():
            for inputs, targets in test_loader1:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_test_loss2 += loss.item() * targets.size(0)  # Adjust loss by batch size
                _, predicted = torch.max(outputs, 1)
                total_test_samples2 += targets.size(0)
                total_test_correct2 += (predicted == targets).sum().item()

                # Collect labels for F1 score
                test_true_labels2.extend(targets.cpu().numpy())
                test_predicted_labels2.extend(predicted.cpu().numpy())

        avg_test_loss2 = total_test_loss2 / total_test_samples2
        test_accuracy2 = 100 * total_test_correct2 / total_test_samples2
        test_f2 = f1_score(test_true_labels2, test_predicted_labels2, average='macro')

        epoch_test_losses2.append(avg_test_loss2)
        epoch_test_accuracies2.append(test_accuracy2)
        epoch_test_f1_scores2.append(test_f2)

        print(
            f'Valid Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%, Macro F1: {test_f1:.4f}')
        print(
            f'Testing Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_test_loss2:.4f}, Accuracy: {test_accuracy2:.2f}%, Macro F1: {test_f2:.4f}')

    # Return the collected statistics
    return {
        'train_losses': epoch_train_losses,
        'train_accuracies': epoch_train_accuracies,
        'train_f1_scores': epoch_train_f1_scores,

        'valid_losses': epoch_test_losses,
        'valid_accuracies': epoch_test_accuracies,
        'valid_f1_scores': epoch_test_f1_scores,

        'test_losses': epoch_test_losses2,
        'test_accuracies': epoch_test_accuracies2,
        'test_f1_scores': epoch_test_f1_scores2
    }


def plot_metrics(metrics):
    epochs = range(1, len(metrics['train_losses']) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_losses'], label='Training Loss')
    plt.plot(epochs, metrics['valid_losses'], label='Validation Loss')
    plt.plot(epochs, metrics['test_losses'], label='Testing Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_accuracies'], label='Training Accuracy')
    plt.plot(epochs, metrics['valid_accuracies'], label='Validation Accuracy')
    plt.plot(epochs, metrics['test_accuracies'], label='Testing Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def data_convert(y):
    K = len(np.unique(y.tolist()))
    eyes_mat = np.eye(K)
    y_onehot = np.zeros((y.shape[0], K))
    for i in range(0, y.shape[0]):
        y_onehot[i] = eyes_mat[y[i]]
    return y_onehot


def roc_auc_curve(y_onehot, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('class ' + str(i))
        plt.legend(loc="lower right")
        plt.show()


def evaluate_result(test_labels, test_predictions, y_score):
    # Calculate the confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)

    # Plot the confusion matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False, square=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    y_onehot = data_convert(np.array(test_labels))
    roc_auc_curve(y_onehot, y_score)

    # Output the classification report
    target_names = list(range(40))
    classification_rep = classification_report(test_labels, test_predictions)
    print('Classification Report:\n', classification_rep)


def predict_model(model, test_loader):
    model.eval()  # 设置模型为评估模式

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():  # 在测试过程中不需要计算梯度
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 获取模型输出
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            # 获取预测标签
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 将结果转换为 numpy 数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    return all_labels, all_preds, all_probs


def plot_metrics_compare(metrics_dict):
    # 创建子图，用于分别展示 train_losses、test_losses、train_accuracies、test_accuracies
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle('Model Performance Metrics', fontsize=16)

    # 分别绘制每个模型的 train_losses、test_losses、train_accuracies、test_accuracies
    for model_name, metrics in metrics_dict.items():
        # 绘制 train_losses
        axes[0, 0].plot(metrics['train_losses'], label=model_name)
        axes[0, 0].set_title('Train Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')

        # 绘制 test_losses
        axes[0, 1].plot(metrics['test_losses'], label=model_name)
        axes[0, 1].set_title('Test Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')

        # 绘制 train_accuracies
        axes[1, 0].plot(metrics['train_accuracies'], label=model_name)
        axes[1, 0].set_title('Train Accuracies')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')

        # 绘制 test_accuracies
        axes[1, 1].plot(metrics['test_accuracies'], label=model_name)
        axes[1, 1].set_title('Test Accuracies')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')

    # 设置每个子图的图例
    for ax in axes.flat:
        ax.legend()

    # 自动调整子图布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 调整标题和子图间的空间
    plt.show()


