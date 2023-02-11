import json
import pandas as pd
import torch
import torch.nn as nn
import numpy as  np
import math
import time
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, precision_recall_fscore_support, roc_curve
from sklearn.utils import shuffle
from torch import argmax
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import argparse
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
onehot_encoder = OneHotEncoder(sparse=False)


# Define super-parameter
parser = argparse.ArgumentParser(description='Attack_VFL_PI')
parser.add_argument('--dataset', type=str, default='bank_marketing', help="dataset") 
parser.add_argument('--number_client', default=2, type=int, help='number_client')
parser.add_argument('--acti', type=str, default='leakyrelu_2', help="activate")  
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--client', default='VFL_client2', type=str, help='client')
parser.add_argument('--epochs_attack', default=200, type=int, help='epochs')
parser.add_argument('--epochs', default=50, type=int, help='epochs')
parser.add_argument('--end_shadow', default=1, type=int, help='end_shadow')
parser.add_argument('--attack_label', default='y1', type=str, help='attack_label')
parser.add_argument('--num_smooth_epoch', default=1, type=int, help='num_smooth_epoch')
parser.add_argument('--out_dim', default=2, type=int, help='out_dim')
parser.add_argument('--num_data', default=7936, type=int, help='num_data')
parser.add_argument('--lr', default=1e-4, type=float, help='lr')  
parser.add_argument('--attackepoch', default=30, type=int, help='attackepoch')
parser.add_argument('--num_cutlayer', default=200, type=int, help='num_cutlayer')
parser.add_argument('--new_shadow', default=100, type=int, help='new_shadow')
parser.add_argument('--attacklen', default=10, type=int, help='attacklen')
args = parser.parse_args() 

dataset = args.dataset
number_client = args.number_client
acti = args.acti
num_smooth_epoch = args.num_smooth_epoch
batch_size = args.batch_size
epochs_attack =args.epochs_attack
epochs=args.epochs
end_shadow = args.end_shadow
attack_label = args.attack_label
cutlayer=args.num_cutlayer
out_dim = args.out_dim
lr = args.lr
num_data = args.num_data
client_c=args.client
attackepoch=args.attackepoch
newshadow=args.new_shadow
attacklen=args.attacklen

minibatch=num_data/batch_size
minibatch=int(minibatch)

def split_data2(data):
    sample_dim=out_dim
    sample_num=int(newshadow/sample_dim)
    data_len = len(data[:,-1])
    data_dim=len(data[-1,:])
    train_data = np.zeros([sample_dim*sample_num*(attacklen),data_dim])
    test_data = []
    sample0=0
    for k in range(attackepoch, attackepoch + attacklen, 1):
      countnum = np.zeros([out_dim])
      for i in range(num_data):
        for j in range(sample_dim):
            if ((data[k*num_data+i,-1] == j)):
                if (countnum[j] < sample_num):
                    countnum[j] += 1
                    train_data[sample0,:]=data[k*minibatch*batch_size+i,:]
                    sample0+=1
    test_data=data
    train_data=torch.tensor(train_data,dtype=float)
    test_data=torch.tensor(test_data,dtype=float)
    return train_data,test_data



# Define record path
save_path1 = f'Results_attack/{dataset}/num_client{number_client}/blackbox_smooth{num_smooth_epoch}/c{cutlayer}_s{newshadow}'
if not os.path.exists(save_path1):
  os.makedirs(save_path1)

if dataset=='bank_marketing':
  data = pd.read_csv(f'Results/{dataset}/num_client{number_client}/{client_c}_{acti}_c{cutlayer}.csv', sep=',',header=None)
  data=np.array(data)
  print('data.shape', data.shape)
  if client_c=='VFL_client2':
    if attack_label == 'y1':  
      data=np.delete(data,[cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+7,cutlayer+8, cutlayer+9], 1)

data=np.array(data)
data_dim=len(data[-1,:])
for i in range(data_dim-1):
  data[:,i] = (data[:,i]-data[:,i].mean())/data[:,i].std()
data[:,-1]=data[:,-1]-1
sensordata_num,sensor_num = data.shape
acc1=['epochs', 'acc','precision','recall','f1score','TPR','FPR', 'TNR', 'TP','FP','TN','FN','AUC']


# Define model
class LinearNet_mul(nn.Module):
    def __init__(self, in_dim=cutlayer, n_hidden_1=500, n_hidden_2=128, out_dim=2):
        super(LinearNet_mul, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                    nn.LeakyReLU()
                                    )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.LeakyReLU()
                                    )
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out

train_set,test_set=split_data2(data)
model = LinearNet_mul(in_dim=cutlayer, n_hidden_1=500, n_hidden_2=128, out_dim=out_dim).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(dataloader, model):
    size = len(train_set)
    model.train()
    correct = 0
    for batch, data in enumerate(dataloader):
        X=data[:,:data_dim-1].to(device)
        X=X.to(torch.float32)
        y=data[:,-1].to(device).long()
        print('X', X.shape)
        pred = model(X)
        loss = loss_func(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 10 == 0 and batch != 0:
            correct_train = correct / ((batch+1) * batch_size)
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%")

def test(dataloader, model):
    n_classes=out_dim
    size = len(test_set)
    num_batches = len(dataloader)
    print('size:', size)
    print('num_batches:', num_batches)
    model.eval()
    test_loss, correct = 0, 0
    TP,FP,TN,FN=0,0,0,0
    ypred = []
    ytrue = []
    y_pred = []
    y_true = []

    with torch.no_grad():
      for data in dataloader:
        X = data[:, :data_dim-1].to(device)
        X = X.to(torch.float32)
        y = data[:, -1].to(device).long()
        pred= model(X)
        test_loss += loss_func(pred, y).item()
        correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

        # for TP FP
        ypred.extend(np.array(pred.argmax(1).cpu(),dtype=int))
        ytrue.extend(np.array(y.cpu(),dtype=int))
        
        # for auc
        y_one_hot = torch.randint(1,(batch_size, n_classes)).to(device).scatter_(1,y.view(-1, 1),1)
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
    correct /=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    acc = (TP+TN)/(TP + FP + TN + FN)
    precision = 1.0*TP/(TP + FP)
    recall = 1.0*TP/(TP + FN)
    F_meature = 2.0*precision*recall/(precision + recall)
    acc1=accuracy_score(ytrue,ypred)
    acc2=accuracy_score(ytrue, ypred, normalize=True, sample_weight=None)
    f1=f1_score(ytrue,ypred,average='macro')
    precision1=precision_score(ytrue,ypred,average='macro')
    recall1=recall_score(ytrue,ypred,average='macro')
    f2=f1_score(ytrue,ypred,average='weighted')
    precision2=precision_score(ytrue,ypred,average='weighted')
    recall2=recall_score(ytrue,ypred,average='weighted')
    TPR=TP/(TP+FN)
    FPR=FP/(FP+TN)
    TNR = TN/(FP+TN)


    #auc_score=0
    auc_score=roc_auc_score(y_true, y_pred, multi_class='ovr')
    return acc,precision,recall,F_meature,TPR, FPR,TNR, TP,FP,TN,FN,auc_score,acc1,precision1,f1,recall1,precision2,f2,recall2,acc2


train_iter = Data.DataLoader(
    dataset=train_set, 
    batch_size=batch_size,  
)

test_iter = Data.DataLoader(
    dataset=test_set,  
    batch_size=batch_size,  
    drop_last=True,
)


filename=f'Results_attack/{dataset}/num_client{number_client}/blackbox_smooth{num_smooth_epoch}/c{cutlayer}_s{newshadow}_{acti}_{client_c}_{attack_label}_{attackepoch}.txt'
f2 = open(filename, 'w')


f2.write(str(acc1)+'\n')
for t in range(epochs_attack):
    print(f"Epoch {t+1}")
    train(train_iter, model)
    acc,precision,recall,fscore,TPR,FPR,TNR,TP,FP,TN,FN,auc,meacc,meprecission,mef1,merecall,precision2,f12,merecall2,acc2=test(test_iter, model)
    acc1 ={'Epochs':t,
            'acc':acc, 
            'precision':precision, 
            'recall':recall,
            'fscore':fscore, 
            'TPR':TPR, 
            'FPR':FPR, 
            'TNR':TNR, 
            'TP':TP,
            'FP':FP, 
            'TN':TN, 
            'FN':FN, 
            'AUC':auc,
            'MEACC':meacc,
            'MEPrecision':meprecission,
            'MEF1':mef1,
            'MErecall':merecall,
            'MEPrecision2':precision2,
            'MEF12':f12,
            'MErecall2':merecall2,
            'ACC2':acc2
            }
    for key in acc1:
        f2.write('\n')
        f2.writelines('"' + str(key) + '": ' + str(acc1[key]))        
    f2.write('\n')
    f2.write('\n')
    f2.write('\n')
    print(acc1)
f2.close()
print("Done!")