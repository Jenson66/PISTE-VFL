import random
import time
from datetime import datetime
from torch.utils.data.sampler import  WeightedRandomSampler
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from model.Linear_NN_2 import *
from utils_tabular import *
from torch import nn
from sys import argv
import os
import argparse
import math
import copy
from collections import defaultdict
import csv


# Define A random_seed
def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_random_seed(1234)



# Define parameter
def parse_args():
    parser = argparse.ArgumentParser(description='VFL1')
    parser.add_argument('--dataset', type=str, default='credit', help="dataset") # [bank_marketing, credit, census, cancer]
    parser.add_argument('--acti', type=str, default='leakyrelu_2', help="acti")  
    parser.add_argument('--number_client', default=2, type=int, help='number_client')
    parser.add_argument('--attack_mode', type=str, default='graybox', help="attack_mode")
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--num_cutlayer', default=200, type=int, help='num_cutlayer')  
    parser.add_argument('--num_recover', default=1, type=int, help='num_recover')
    parser.add_argument('--num_smooth_epoch', default=1, type=int, help='num_smooth_epoch')
    parser.add_argument('--num_shadow', default=100, type=int, help='num_shadow') 
    parser.add_argument('--attack_time', default=48, type=int, help='attack_time')
    parser.add_argument('--attack_batch', default=0, type=int, help='attack_batch') 
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='lr')  
    return parser.parse_args(argv[1:])

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


def l2_regularization(model, l2_alpha):
    reg_loss = None
    for param in model.parameters():
        if reg_loss is None:
            reg_loss = l2_alpha * torch.sum(param**2)
        else:
            reg_loss = reg_loss + l2_alpha * param.norm(2)**2
    return reg_loss


def true_server(client1_fx, client2_fx, y, batch, correct, t, size, sum_batch):
    save_server = f'Results/{dataset}/num_client{number_client}/server_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time}_b{attack_batch}.pth'
    server_model = torch.load(save_server)
    global train_true
    global test_true

    server_model.eval()
    correct = correct

    # Data of (output_cutlayer, y) for server
    client1_fx = client1_fx.to(device)
    client2_fx = client2_fx.to(device)
    y = y.to(device)

    # eval
    fx_server = server_model(client1_fx, client2_fx)
    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()

    if t ==0:
        train_true.extend(fx_server.argmax(1).tolist())
    if t ==1:
        test_true.extend(fx_server.argmax(1).tolist())
  
    correct_train = correct / size
    current = (batch+1) * len(client1_fx)
    if batch == sum_batch-1:
      print(f"acc: [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%" , file=filename1)

    return correct


# Test_Client Side Program
def true_client(dataloader, client_model_1, client_model_2, t):
    save_path1 = f'Results/{dataset}/num_client{number_client}/client1_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time}_b{attack_batch}.pth'
    save_path2 = f'Results/{dataset}/num_client{number_client}/client2_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time}_b{attack_batch}.pth'

    client_model_1 = torch.load(save_path1)
    client_model_2 = torch.load(save_path2) 

    if t ==0:
        size = size_train
    if t ==1:
        size = size_test
    sum_batch = len(dataloader)
        
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # a = torch.chunk(X, 2, dim=1)
        if dataset == 'bank_marketing':
            X1 = X[:, :10]
            X2 = X[:, 10:]

        # client1--train and update
        fx1 = client_model_1(X1, 1)
        client1_fx = fx1.clone().detach().requires_grad_(True)

        # client2--train and update
        fx2 = client_model_2(X2, 2)
        client2_fx = fx2.clone().detach().requires_grad_(True)

        # Sending activations to server and receiving gradients from server
        correct = true_server(client1_fx, client2_fx, y, batch, correct, t, size, sum_batch)

    correct /= size
    print(f"True Error: {t}: \n Accuracy: {(100 * correct):>0.1f}% \n", file=filename1)

def fake_server(client1_fx, client2_fx, y, batch, correct, t, size, sum_batch):
    save_server = f'Results/{dataset}/num_client{number_client}/server_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time}_b{attack_batch}.pth'
    server_model = torch.load(save_server)
    global train_fake
    global test_fake

    server_model.eval()
    correct = correct

    # Data of (output_cutlayer, y) for server
    client1_fx = client1_fx.to(device)
    client2_fx = client2_fx.to(device)
    y = y.to(device)

    # eval
    fx_server = server_model(client1_fx, client2_fx)

    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()
    
    if t ==0:
        train_fake.extend(fx_server.argmax(1).tolist())
    if t ==1:
        test_fake.extend(fx_server.argmax(1).tolist())

    correct_train = correct / size
    current =  (batch+1) * len(client1_fx)
    if batch == sum_batch-1:
      print(f"ttest-loss: [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%" , file=filename1)


    return correct


# Test_Client Side Program
def fake_client(dataloader, client_model_1, client_model_2, t):
    save_path1 = f'Results_attack/{dataset}/num_client{number_client}/{attack_mode}_smooth{num_smooth_epoch}/model1_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_r_{num_recover}_epoch{attack_time}.pth'
    save_path2 = f'Results_attack/{dataset}/num_client{number_client}/{attack_mode}_smooth{num_smooth_epoch}/model2_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_r_{num_recover}_epoch{attack_time}.pth'

    client_model_1 = torch.load(save_path1)
    client_model_2 = torch.load(save_path2) 

    if t ==0:
        size = size_train
    if t ==1:
        size = size_test
    sum_batch = len(dataloader)
        
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # a = torch.chunk(X, 2, dim=1)
        if dataset == 'bank_marketing':
            X1 = X[:, :10]
            X2 = X[:, 10:]
 

        # client1--train and update
        fx1 = client_model_1(X1, 1)
        client1_fx = fx1.clone().detach().requires_grad_(True)


        # client2--train and update
        fx2 = client_model_2(X2, 2)
        client2_fx = fx2.clone().detach().requires_grad_(True)


        # Sending activations to server and receiving gradients from server
        correct = fake_server(client1_fx, client2_fx, y, batch, correct, t, size, sum_batch)

    correct /= size
    print(f"True Error: {t}: \n Accuracy: {(100 * correct):>0.1f}% \n", file=filename1)



if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    batch_size = args.batch_size
    number_client = args.number_client
    attack_mode = args.attack_mode
    num_cutlayer = args.num_cutlayer
    num_recover = args.num_recover
    num_shadow = args.num_shadow
    attack_time = args.attack_time
    attack_batch = args.attack_batch
    num_smooth_epoch = args.num_smooth_epoch

    epochs = args.epochs
    lr = args.lr
    dataset=args.dataset
    acti = args.acti
    time_start_load_everything = time.time()


    save_path = f'Results_attack/{dataset}/num_client{number_client}/{attack_mode}_smooth{num_smooth_epoch}/'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    
    filename1 = open(f'Results_attack/{dataset}/num_client{number_client}/{attack_mode}_smooth{num_smooth_epoch}/Aggrement_{acti}_c{num_cutlayer}_b{batch_size}.txt', 'w+')

    # Define data
    train_iter, test_iter, size_train, size_test = load_data(dataset, batch_size)


    # Define model
    if attack_mode == 'graybox':
        if dataset == 'bank_marketing':
            if acti == 'leakyrelu_2':
              client_model1 = Client_LeakyreluNet_2_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=1).to(device)
              client_model2 = Client_LeakyreluNet_2_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=2).to(device)
              server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*2, n_hidden_3=num_cutlayer, n_hidden_4=64, n_hidden_5=32, out_dim=2).to(device)

              optimizer_client1 = torch.optim.Adam(client_model1.parameters(), lr=lr)
              optimizer_client2 = torch.optim.Adam(client_model2.parameters(), lr=lr)
              optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)

            if acti == 'leakyrelu_3':
              client_model1 = Client_LeakyreluNet_3_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=1).to(device)
              client_model2 = Client_LeakyreluNet_3_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=2).to(device)
              server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*2, n_hidden_3=num_cutlayer, n_hidden_4=64, n_hidden_5=32, out_dim=2).to(device)

              optimizer_client1 = torch.optim.Adam(client_model1.parameters(), lr=lr)
              optimizer_client2 = torch.optim.Adam(client_model2.parameters(), lr=lr)
              optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)

            if acti == 'linear':
              client_model1 = Client_LinearNet_2_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=1).to(device)
              client_model2 = Client_LinearNet_2_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=2).to(device)
              server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*2, n_hidden_3=num_cutlayer, n_hidden_4=64, n_hidden_5=32, out_dim=2).to(device)

              optimizer_client1 = torch.optim.Adam(client_model1.parameters(), lr=lr)
              optimizer_client2 = torch.optim.Adam(client_model2.parameters(), lr=lr)
              optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)
    
    if attack_mode == 'blackbox':
        if dataset == 'bank_marketing':
            client_model1 = Client_LeakyreluNet_4_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=1).to(device)
            client_model2 = Client_LeakyreluNet_4_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=2).to(device)
            server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*2, n_hidden_3=num_cutlayer, n_hidden_4=64, n_hidden_5=32, out_dim=2).to(device)

            optimizer_client1 = torch.optim.Adam(client_model1.parameters(), lr=lr)
            optimizer_client2 = torch.optim.Adam(client_model2.parameters(), lr=lr)
            optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)


    # start training
    for t in range(1):
        print(f"Epoch {t+1}\n-------------------------------", file=filename1)
        train_true = []
        test_true = []

        true_client(train_iter, client_model1, client_model2, 0)
        true_client(test_iter, client_model1, client_model2, 1)

        train_fake = []
        test_fake = []
        fake_client(train_iter, client_model1, client_model2, 0)
        fake_client(test_iter, client_model1, client_model2, 1)


    train_aggrement = 0
    test_aggrement = 0
    for i in range(len(train_true)):
        if train_true[i]==train_fake[i]:
            train_aggrement +=1

    for i in range(len(test_true)):
        if test_true[i]==test_fake[i]:
            test_aggrement +=1

    print('len(train_true)', len(train_true),  file=filename1)
    print('len(test_true)', len(test_true),  file=filename1)

    print('train_aggrement', train_aggrement,  file=filename1)
    print('train_aggrement', train_aggrement/ len(train_true),  file=filename1)

    print('test_aggrement', test_aggrement,  file=filename1)
    print('test_aggrement', test_aggrement/ len(test_true),  file=filename1)


    print("Done!", file=filename1)

    







