import json
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import math
import time
import os
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, precision_recall_fscore_support, \
    roc_curve
from sklearn.utils import shuffle
from torch.utils.data.sampler import WeightedRandomSampler
from torch import argmax
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import argparse

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

onehot_encoder = OneHotEncoder(sparse=False)

# Define super-parameter
parser = argparse.ArgumentParser(description='Attack_PI')
parser.add_argument('--dataset', type=str, default='utkface', help="dataset")
parser.add_argument('--model', type=str, default='lenet', help="model")
parser.add_argument('--acti', type=str, default='leakyrelu', help="acti")
parser.add_argument('--attributes', type=str, default="race_gender", help="For attrinf, two attributes should be in format x_y e.g. race_gender")
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--client', default='VFL_client1', type=str, help='client')
parser.add_argument('--epochs_attack', default=50, type=int, help='epochs_attack')
parser.add_argument('--epochs', default=100, type=int, help='epochs')
parser.add_argument('--out_dim', default=2, type=int, help='out_dim')
parser.add_argument('--lr', default=1e-4, type=float, help='lr') 
parser.add_argument('--attackepoch', default=30, type=int, help='attackepoch')
parser.add_argument('--num_cutlayer', default=1000, type=int, help='num_cutlayer')
parser.add_argument('--new_shadow', default=2000, type=int, help='new_shadow')
parser.add_argument('--end_shadow', default=100, type=int, help='end_shadow')
parser.add_argument('--attack_label', default='y1', type=str, help='attack_label')
args = parser.parse_args()

dataset = args.dataset
model = args.model
acti = args.acti
batch_size = args.batch_size
attributes = args.attributes
epochs_attack = args.epochs_attack
epochs = args.epochs
attack_label = args.attack_label
out_dim = args.out_dim
num_cutlayer =args.num_cutlayer
lr = args.lr
client_c = args.client
end_shadow= args.end_shadow
newshadow=args.new_shadow

attackepoch=args.attackepoch-45
batch_sizeattck=64
attacklen = 10
minibatch = 200


def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)

def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


def split_data(data):
    data_len = len(data[:,-1])
    data_dim=len(data[-1,:])
    print(data.shape)
    train_data = np.zeros((newshadow*attacklen,data_dim))
    test_data = np.zeros(((batch_size*minibatch-newshadow)*(attacklen),data_dim))
    for j in range(attackepoch,attackepoch+attacklen,1):
      k=0
      for i in range(minibatch):
        if(k<newshadow):
          k+=end_shadow
          train_data[newshadow*(j-attackepoch)+i*end_shadow:newshadow*(j-attackepoch)+end_shadow*(i+1),:]=data[j*newshadow+i*batch_size:j*newshadow+i*batch_size+end_shadow,:]
          test_data[(batch_size*minibatch-newshadow)*(j-attackepoch)+i*(batch_size-end_shadow):(batch_size*minibatch-newshadow)*(j-attackepoch)+(batch_size-end_shadow)*(i+1),:]=data[j*minibatch*batch_size+i*batch_size+end_shadow:j*minibatch*batch_size+(i+1)*batch_size,:]
        else:
          test_data[(batch_size*minibatch-newshadow)*(j-attackepoch)+i*(batch_size)-newshadow:(batch_size*minibatch-newshadow)*(j-attackepoch)+(batch_size)*(i+1)-newshadow,:]=data[j*minibatch*batch_size+i*batch_size:j*minibatch*batch_size+(i+1)*batch_size,:]
    train_data=torch.tensor(train_data,dtype=float)
    test_data=torch.tensor(test_data,dtype=float)
    return train_data,  test_data

# Define model
class LinearNet_mul(nn.Module):
    def __init__(self, in_dim=64, n_hidden_1=500, n_hidden_2=128, n_hidden_3=128, n_hidden_4 = 128, n_hidden_5 = 256, out_dim=2):
        super(LinearNet_mul, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.5),
                                    )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.5),  
                                    )
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.5), 
                                    )
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.5),  
                                    )
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.5),)

        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim))

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer6_out


def train(dataloader, model):
    size = train_size
    model.train()
    correct = 0
    for batch, data in enumerate(dataloader):
        X = data[:, :data_dim - 1].to(device)
        X = X.to(torch.float32)
        y = data[:, -1].to(device).long()
        pred = model(X)
        loss1 = loss_func(pred, y)
        l2 = l2_regularization(model, 10)
        loss = loss1+l2
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 10 == 0 and batch != 0:
            correct_train = correct / ((batch + 1) * len(X))
            loss, current = loss.item(), (batch +1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%")


def test(dataloader, model):
    n_classes = out_dim
    size = test_size
    num_batches = len(dataloader)
    print('size:', size)
    print('num_batches:', num_batches)
    model.eval()
    test_loss, correct = 0, 0
    TP, FP, TN, FN = 0, 0, 0, 0
    ypred = []
    ytrue = []
    y_pred = []
    y_true = []

    for data in dataloader:
        X = data[:, :data_dim - 1].to(device)
        X = X.to(torch.float32)
        y = data[:, -1].to(device).long()
        pred = model(X)
        test_loss += loss_func(pred, y).item()
        correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

        ypred.extend(np.array(pred.argmax(1).cpu(), dtype=int))
        ytrue.extend(np.array(y.cpu(), dtype=int))

 
        y_one_hot = torch.randint(1, (batch_sizeattck, n_classes)).to(device).scatter_(1, y.view(-1, 1), 1)
        y_true.extend(y_one_hot.tolist())
        y_pred.extend(pred.softmax(dim=-1).tolist())

    cm = confusion_matrix(ytrue, ypred, labels=range(n_classes))
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    print(f'TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}')

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    acc = (TP + TN) / (TP + FP + TN + FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F_meature = 2.0 * precision * recall / (precision + recall)

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (FP + TN)

    # auc
    auc_score = roc_auc_score(y_true, y_pred, multi_class='ovr')

    return acc, precision, recall, F_meature, TPR, FPR, TNR, TP, FP, TN, FN, auc_score


# Define data
if dataset == 'utkface':
    data = pd.read_csv(f'Results/{dataset}/{model}/{client_c}_c{num_cutlayer}_{acti}_b{batch_size}_E{epochs}.csv', sep=',', header=None, error_bad_lines=False)
    data=np.array(data)

# Pre-processing data
data_dim = len(data[-1, :])
acc1 = ['epochs', 'acc', 'precision', 'recall', 'f1score', 'TPR', 'FPR', 'TNR', 'TP', 'FP', 'TN', 'FN', 'AUC']
train_set, test_set = split_data(data)


# Sample data
train_iter = Data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_sizeattck,  
                    drop_last=True,)
test_iter = Data.DataLoader(
                    dataset=test_set, 
                    batch_size=batch_sizeattck,  
                    drop_last=True,)
train_size = len(train_iter)*batch_sizeattck
test_size = len(test_iter)*batch_sizeattck



# Define model
Attack_model = LinearNet_mul(in_dim=num_cutlayer, n_hidden_1=1024, n_hidden_2=512, n_hidden_3=512, n_hidden_4 = 256,  n_hidden_5 = 256, out_dim=out_dim).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Attack_model.parameters(), lr=lr)


# Define rocords
save_path = f'Results_attack/{dataset}/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

filename = f'Results_attack/{dataset}/{client_c}_{dataset}_{model}_c{num_cutlayer}_b{batch_size}_{attack_label}_{newshadow}_{attackepoch}.txt'
f2 = open(filename, 'w')
f2.write('Attack_' + str(dataset) + 'num_cutlayer'+ str(num_cutlayer)+ 'batch_size'+ str(batch_size)+ '_epochs_attack' + str(epochs_attack) + '_Adam_lr_' + str(lr) + '_end_shadow_' + str(
        end_shadow) + '_' + str(attack_label) + '\n')
f2.write(str(acc1) + '\n')


for t in range(epochs_attack):
    print(f"Epoch {t + 1}")
    train(train_iter, Attack_model)
    acc, precision, recall, fscore, TPR, FPR, TNR, TP, FP, TN, FN, auc = test(test_iter, Attack_model)
    acc1 = {'Epochs': t,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'TPR': TPR,
            'FPR': FPR,
            'TNR': TNR,
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'AUC': auc}
    for key in acc1:
        f2.write('\n')
        f2.writelines('"' + str(key) + '": ' + str(acc1[key]))
    f2.write('\n')
    f2.write('\n')
    f2.write('\n')
    print(acc1)
f2.close()
print("Done!")