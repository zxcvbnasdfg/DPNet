
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import read_directory
import matplotlib.pyplot as plt
import torchvision.models as models
import os

from sklearn.metrics import confusion_matrix

from timm.loss import LabelSmoothingCrossEntropy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# if torch.cuda.is_available():
torch.backends.cudnn.deterministic = True
from scipy.io import loadmat, savemat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# In[2] 加载数据
num_classes = 2
height = 64
width = 64

# 小波时频图---2D-CNN输入
x_train, y_train = read_directory('E:/数据/小波/train1_img', height, width, normal=1)
x_valid, y_valid = read_directory('E:/数据/小波/valid1_img', height, width, normal=1)

# FFT频域信号--1D-CNN输入
datafft = loadmat('E:/数据/FFT.mat')
x_train2 = datafft['train_X']
x_valid2 = datafft['valid_X']
ss2 = StandardScaler().fit(x_train2)
x_train2 = ss2.transform(x_train2)
x_valid2 = ss2.transform(x_valid2)

x_train2 = x_train2.reshape(x_train2.shape[0], 1, -1)
x_valid2 = x_valid2.reshape(x_valid2.shape[0], 1, -1)

# 转换为torch的输入格式
train_features = torch.tensor(x_train).type(torch.FloatTensor)
valid_features = torch.tensor(x_valid).type(torch.FloatTensor)

train_features2 = torch.tensor(x_train2).type(torch.FloatTensor)
valid_features2 = torch.tensor(x_valid2).type(torch.FloatTensor)

train_labels = torch.tensor(y_train).type(torch.LongTensor)
valid_labels = torch.tensor(y_valid).type(torch.LongTensor)

print(train_features.shape)
print(train_features2.shape)
print(train_labels.shape)
N = train_features.size(0)

# In[3]: 参数设置
learning_rate = 0.005          # 学习率
num_epochs = 1            # 迭代次数
batch_size = 64             # batchsize

# In[4]:
# 模型设置

class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()


        self.resnet18 = models.resnet18(pretrained=True)

        for param in self.resnet18.parameters():
            param.requires_grad = False

        self.net3 = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=5),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(6),
            nn.Conv1d(6, 16, kernel_size=3),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )



        self.net3 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 6, kernel_size=1),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(6),
            nn.Conv1d(6, 6, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(6),
            nn.Conv1d(6, 16, kernel_size=1),
            nn.MaxPool1d(kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )


        self.feature_layer = nn.Sequential(
            nn.Linear(1000 + 1216, 120),

            nn.ReLU(),
            nn.Linear(120, 84),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x1, x2):
        x1 = self.resnet18(x1)
        x2 = self.net3(x2)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat([x1, x2], dim=1)
        fc = self.feature_layer(x)
        logits = self.classifier(fc)
        probas = torch.softmax(logits, dim=1)
        return logits, probas, x1, x2, fc


def compute_accuracy(model, feature, feature1, labels):
    correct_pred, num_examples = 0, 0
    l = 0
    N = feature.size(0)
    total_batch = int(np.ceil(N / batch_size))
    indices = np.arange(N)
    np.random.shuffle(indices)
    for i in range(total_batch):
        start_idx = batch_size * i
        end_idx = min(batch_size * (i + 1), N)
        rand_index = indices[start_idx:end_idx]
        # rand_index = indices[batch_size*i:batch_size*(i+1)]
        features = feature[rand_index, :]
        features1 = feature1[rand_index, :]

        targets = labels[rand_index]

        features = features.to(device)
        features1 = features1.to(device)
        targets = targets.to(device)

        logits, probas, _, _, _ = model(features, features1)
        cost = loss(logits, targets)
        _, predicted_labels = torch.max(probas, 1)

        num_examples += targets.size(0)
        l += cost.item()
        correct_pred += (predicted_labels == targets).sum()

    return l / num_examples, correct_pred.float() / num_examples * 100



model = ConvNet(num_classes=num_classes)
model = model.to(device)  # 传进device
# print(model)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # sgd优化器

loss = LabelSmoothingCrossEntropy()

train_loss, valid_loss = [], []
train_acc, valid_acc = [], []

best_accuracy = 0

for epoch in range(num_epochs):
    model = model.train()
    total_batch = int(np.ceil(N / batch_size))
    indices = np.arange(N)
    np.random.shuffle(indices)
    avg_loss = 0
    for i in range(total_batch):
        start_idx = batch_size * i
        end_idx = min(batch_size * (i + 1), N)
        rand_index = indices[start_idx:end_idx]
        # rand_index = indices[batch_size*i:batch_size*(i+1)]
        features = train_features[rand_index, :]
        features2 = train_features2[rand_index, :]

        targets = train_labels[rand_index]

        features = features.to(device)
        features2 = features2.to(device)
        targets = targets.to(device)

        logits, probas, _, _, _ = model(features, features2)
        cost = loss(logits, targets)


        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    model = model.eval()  # 关闭dropout
    # print('valid_features.shape',valid_features.shape)
    # print('valid_features2.shape',valid_features2.shape)
    # print('valid_labels.shape',valid_labels.shape)
    with torch.set_grad_enabled(False):  # save memory during inference
        trl, trac = compute_accuracy(model, train_features, train_features2, train_labels)
        val, vaac = compute_accuracy(model, valid_features, valid_features2, valid_labels)
        print('Epoch: %03d/%03d training accuracy: %.2f%%  training loss:  %f  validing accuracy: %.2f%% validing loss:  %f' % (
            epoch + 1, num_epochs,
            trac, trl,
            vaac, val))
        if vaac > best_accuracy:
            best_accuracy = vaac
            torch.save(model.state_dict(), 'model/RESNET_DW2.pkl')  # 保存最佳模型参数
            print("Now save this model as best model:", epoch+1)

    train_loss.append(trl)
    valid_loss.append(val)

    train_acc.append(trac)
    valid_acc.append(vaac)

# In[6]: 利用训练好的模型 对测试集进行分类


plt.figure()
plt.plot(np.array(train_loss),label='train')
plt.plot(np.array(valid_loss),label='valid')
plt.title('loss curve')
plt.legend()
plt.show()




# 提取测试集图片
x_test, y_test = read_directory('E:/数据/小波/test1_img', height, width, normal=1)
x_test2 = datafft['test_X']
x_test2 = ss2.transform(x_test2)
x_test2 = x_test2.reshape(x_test2.shape[0], 1, -1)

test_features = torch.tensor(x_test).type(torch.FloatTensor)
test_features2 = torch.tensor(x_test2).type(torch.FloatTensor)
test_labels = torch.tensor(y_test).type(torch.LongTensor)


model = ConvNet(num_classes= num_classes)
model.load_state_dict(torch.load("model/RESNET_DW2.pkl"))
model.to(device)
model = model.eval()
_, teac = compute_accuracy(model, test_features, test_features2, test_labels)
print('测试集正确率为：', teac.item(), '%')


