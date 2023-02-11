import random
import time
from model.Linear_NN_2 import *
from utils_tabular import *
from torch import nn
from sys import argv
import os
import argparse
import copy


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
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--num_cutlayer', default=200, type=int, help='num_cutlayer')  
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


# Train_Server Side Program
def train_server(client1_fx, client2_fx, y, t,  batch,correct):
    server_model.train()
    correct = correct
    size = size_train

    # Data of (output_cutlayer, y) for server
    client1_fx = client1_fx.to(device)
    client2_fx = client2_fx.to(device)
    y = y.to(device)

    # train and update
    optimizer_server.zero_grad()
    fx_server = server_model(client1_fx, client2_fx)

    loss = criterion(fx_server, y) + l2_regularization(server_model, 0.00005) +l2_regularization(client_model_1, 0.00005) +l2_regularization(client_model_2, 0.00005) 
    loss.backward()
    optimizer_server.step()
  
    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()
    correct_train = correct / size
    loss, current = loss.item(), (batch+1) * len(client1_fx.grad.clone().detach())

    #  record for attack
    if t>=45 and batch<=2:
      save_server = f'Results/{dataset}/num_client{number_client}/server_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch}.pth'
      torch.save(server_model, save_server)
      yt = copy.deepcopy(fx_server.detach().cpu().numpy())
      np.save(f'Results/{dataset}/num_client{number_client}/yt_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch}.npy', yt)

    return client1_fx.grad, client2_fx.grad, correct


# Train_Client Side Program
def train_client(dataloader, client_model_1, client_model_2,  t):
    client_model_1.train()
    client_model_2.train()
    print('size_train', size_train, file=filename1)

    correct = 0
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)
      # a = torch.chunk(X, 2, dim=1)
      if dataset == 'bank_marketing':
          X1 = X[:, :10]
          X2 = X[:, 10:]

      # client1--train and update
      optimizer_client1.zero_grad()
      fx1 = client_model_1(X1, 1)
      client1_fx = fx1.clone().detach().requires_grad_(True)

      # client2--train and update
      optimizer_client2.zero_grad()
      fx2 = client_model_2(X2, 2)
      client2_fx = fx2.clone().detach().requires_grad_(True)
 
      # Sending activations to server and receiving gradients from server
      g_fx1, g_fx2, correct = train_server(client1_fx, client2_fx, y, t, batch, correct)  
 
      # backward prop
      (client1_fx).backward(g_fx1) 
      (client2_fx).backward(g_fx2) 

      optimizer_client1.step()
      optimizer_client2.step()

      
      # record for attack
      n1 = torch.cat([fx1, X1], dim=1)
      n2 = torch.cat([fx2, X2], dim=1)
      n1 = n1.cpu().detach().numpy()
      n2 = n2.cpu().detach().numpy()
      writer_1.writerows(n1)
      writer_2.writerows(n2)

      if t>=45 and batch<=2:
        save_path1 = f'Results/{dataset}/num_client{number_client}/client1_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch}.pth'
        save_path2 = f'Results/{dataset}/num_client{number_client}/client2_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch}.pth'
       
        torch.save(client_model_1, save_path1)
        torch.save(client_model_2, save_path2)
      
        X1_s = copy.deepcopy(X1.detach().cpu().numpy())
        X2_s = copy.deepcopy(X2.detach().cpu().numpy())
        client1_fx_before = copy.deepcopy((fx1).detach().cpu().numpy())
        client2_fx_before = copy.deepcopy((fx2).detach().cpu().numpy())

        np.save(f'Results/{dataset}/num_client{number_client}/X1_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch}.npy', X1_s)
        np.save(f'Results/{dataset}/num_client{number_client}/X2_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch}.npy', X2_s)
    
        np.save(f'Results/{dataset}/num_client{number_client}/fx1_before{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch}.npy', client1_fx_before)
        np.save(f'Results/{dataset}/num_client{number_client}/fx2_before{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch}.npy', client2_fx_before)


def test_server(client1_fx, client2_fx, y, batch, correct):
    server_model.eval()
    correct = correct
    size = size_test

    # Data of (output_cutlayer, y) for server
    client1_fx = client1_fx.to(device)
    client2_fx = client2_fx.to(device)
    y = y.to(device)

    # eval
    fx_server = server_model(client1_fx, client2_fx)
    loss = criterion(fx_server, y)
    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()

    correct_train = correct / size
    loss, current = loss.item(), (batch+1) * len(client1_fx)
    if batch == len(test_iter)-1:
      print(f"ttest-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%" , file=filename1)

    return correct


# Test_Client Side Program
def test_client(dataloader, client_model_1, client_model_2, t):
    client_model_1.eval()
    client_model_2.eval()

    correct = 0
    size = size_test
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        if dataset == 'bank_marketing':
          X1 = X[:, :10]
          X2 = X[:, 10:]

        if dataset == 'credit':
          X1 = X[:, :14]
          X2 = X[:, 14:]

        # client1--train and update
        optimizer_client1.zero_grad()
        fx1 = client_model_1(X1, 1)
        client1_fx = fx1.clone().detach().requires_grad_(True)

        # client2--train and update
        optimizer_client2.zero_grad()
        fx2 = client_model_2(X2, 2)
        client2_fx = fx2.clone().detach().requires_grad_(True)
 
        # Sending activations to server and receiving gradients from server
        correct = test_server(client1_fx, client2_fx, y, batch, correct)

    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}% \n", file=filename1)


if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    batch_size = args.batch_size
    num_cutlayer = args.num_cutlayer
    epochs = args.epochs
    lr = args.lr
    dataset=args.dataset
    acti = args.acti
    number_client = args.number_client
    time_start_load_everything = time.time()

    # Define record path
    save_path = f'Results/{dataset}/num_client{number_client}/'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    
    writer_1, writer_2 = records_path(acti, dataset, number_client, num_cutlayer)
    
    filename1 = open(f'Results/{dataset}/num_client{number_client}/{acti}_c{num_cutlayer}_b{batch_size}.txt', 'w+')


    # Define data
    train_iter, test_iter, size_train, size_test = load_data(dataset, batch_size)
    
    
    # Define model
    if dataset == 'bank_marketing':
        if acti == 'leakyrelu_2':
          client_model_1 = Client_LeakyreluNet_2_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_2_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=2).to(device)
          server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*2, n_hidden_3=num_cutlayer, n_hidden_4=64, n_hidden_5=32, out_dim=2).to(device)

          optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=lr)
          optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=lr)
          optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)

        if acti == 'leakyrelu_3':
          client_model_1 = Client_LeakyreluNet_3_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_3_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=2).to(device)
          server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*2, n_hidden_3=num_cutlayer, n_hidden_4=64, n_hidden_5=32, out_dim=2).to(device)

          optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=lr)
          optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=lr)
          optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)

        if acti == 'linear':
          client_model_1 = Client_LinearNet_2_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LinearNet_2_em_bm(in_dim=10, n_hidden_1=256,  n_hidden_2=num_cutlayer, client=2).to(device)
          server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*2, n_hidden_3=num_cutlayer, n_hidden_4=64, n_hidden_5=32, out_dim=2).to(device)

          optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=lr)
          optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=lr)
          optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)
    
  
    # Define criterion
    criterion = nn.CrossEntropyLoss()     

    # start training
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------", file=filename1)
        train_client(train_iter, client_model_1, client_model_2, t)
        test_client(test_iter, client_model_1, client_model_2, t)

    print("Done!", file=filename1)

    







