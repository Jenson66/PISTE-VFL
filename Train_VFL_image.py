import random
import time
from model.lenet import *
from utils_image import *
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
    parser = argparse.ArgumentParser(description='VFL')
    parser.add_argument('--dataset', type=str, default='utkface', help="dataset")
    parser.add_argument('--model', type=str, default='lenet', help="model")
    parser.add_argument('--acti', type=str, default='leakyrelu', help="acti")
    parser.add_argument('--attack_label', type=int, default='0')
    parser.add_argument('--attributes', type=str, default="race_gender", help="For attrinf, two attributes should be in format x_y e.g. race_gender")
    parser.add_argument('--lr', default=1e-4, type=float, help='lr')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--num_cutlayer', default=1000, type=int, help='num_cutlayer')
    return parser.parse_args(argv[1:])


# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Train_Server Side Program
def train_server(client1_fx, client2_fx, Y_1, t, batch_id, correct, size):
    server_model.train()
    if t >= 45 and batch_id <= 30:
        save_path3 = f'Results/{dataset}/{model}/server_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch_id}.pth'
        torch.save(server_model, save_path3)

    correct = correct

    # train and update
    optimizer_server.zero_grad()
    fx_server = server_model(client1_fx, client2_fx)
    loss = criterion(fx_server, Y_1)

    # backward
    loss.backward()
    dfx1_client = client1_fx.grad.clone().detach().to(device)
    dfx2_client = client2_fx.grad.clone().detach().to(device)
    optimizer_server.step()
    correct += (fx_server.argmax(1) == Y_1).type(torch.float).sum().item()

    correct_train = correct / size
    loss, current = loss.item(), (batch_id + 1) * len(dfx1_client)
    print(f"train-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
          file=filename)
    return dfx1_client, dfx2_client, correct


# Train_Client Side Program
def train_client(dataloader, client_model_1, client_model_2, t):
    client_model_1.train()
    client_model_2.train()
    correct = 0
    size = len(dataloader.dataset)

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(X)
        Y_1 = target[0].to(device)
        Y_2 = target[1].view(-1, 1).to(device)


        if t >= 45 and batch_id <= 30:
            save_path1 = f'Results/{dataset}/{model}/client1_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch_id}.pth'
            save_path2 = f'Results/{dataset}/{model}/client2_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch_id}.pth'
            torch.save(client_model_1, save_path1)
            torch.save(client_model_2, save_path2)

        # client1--train and update
        fx1 = client_model_1(X_1)
        fx2 = client_model_2(X_2)
        client1_fx = (fx1).clone().detach().requires_grad_(True)
        client2_fx = (fx2).clone().detach().requires_grad_(True)

        # Sending activations to server and receiving gradients from server
        g_fx1, g_fx2, correct = train_server(client1_fx, client2_fx, Y_1, t, batch_id, correct, size)

        # backward prop
        optimizer_client1.zero_grad()
        optimizer_client2.zero_grad()
        (fx1).backward(g_fx1)
        (fx2).backward(g_fx2)

        optimizer_client1.step()
        optimizer_client2.step()

        # record for attack
        if t >= 45 and batch_id <= 200:
            # for property inference
            n1 = torch.cat([fx1, Y_2], dim=1)
            n1 = n1.cpu().detach().numpy()
            writer_1.writerows(n1)

            if batch_id <= 30:   
                X1_s = copy.deepcopy(X_1.detach().cpu().numpy())
                X2_s = copy.deepcopy(X_2.detach().cpu().numpy())
                client1_fx_s = copy.deepcopy(client1_fx.detach().cpu().numpy())
                client2_fx_s = copy.deepcopy(client2_fx.detach().cpu().numpy())

                np.save(f'Results/{dataset}/{model}/X1_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch_id}.npy',
                        X1_s)
                np.save(f'Results/{dataset}/{model}/X2_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch_id}.npy',
                        X2_s)
                np.save(f'Results/{dataset}/{model}/fx1_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch_id}.npy',
                        client1_fx_s)
                np.save(f'Results/{dataset}/{model}/fx2_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch_id}.npy',
                        client2_fx_s)


# Test_Server Side Program
def test_server(client1_fx, client2_fx, y, batch_id, correct, size):
    server_model.train()
    correct = correct

    # train and update
    optimizer_server.zero_grad()
    fx_server = server_model(client1_fx, client2_fx)
    loss = criterion(fx_server, y)

    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()
    correct_train = correct / size
    loss, current = loss.item(), (batch_id + 1) * len(y)
    if batch_id == len(test_data) - 1:
        print(f"ttest-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
              file=filename)
    return correct


# Test_Client Side Program
def test_client(dataloader, client_model_1, client_model_2, t):
    client_model_1.eval()
    client_model_2.eval()
    correct = 0
    size = len(dataloader.dataset)

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(X)

        Y_1 = target[0].to(device)

        # client1--train and update
        fx1 = client_model_1(X_1)
        fx2 = client_model_2(X_2)
        client1_fx = fx1.clone().detach().requires_grad_(True)
        client2_fx = fx2.clone().detach().requires_grad_(True)

        # Sending activations to server and receiving gradients from server
        correct = test_server(client1_fx, client2_fx, Y_1, batch_id, correct, size)

    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}% \n", file=filename)


if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    dataset = args.dataset
    model = args.model
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    acti = args.acti
    attributes = args.attributes
    attack_label = args.attack_label
    num_cutlayer = args.num_cutlayer
    time_start_load_everything = time.time()

    if dataset == 'utkface':
        attributes = 'race_gender'

    if dataset == 'celeba':
        attributes = "attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr"

    # Define record path
    save_path_1 = f'Results/{dataset}/{model}/'
    if not os.path.exists(save_path_1):
        os.makedirs(save_path_1)
    filename = open(f'Results/{dataset}/{model}/c{num_cutlayer}_{acti}_b{batch_size}.txt', 'w+')
    
    writer_1, writer_2 = records_path(model, dataset, acti, num_cutlayer, batch_size, epochs)

    ### Load data
    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    train_data, test_data, num_classes1, num_classes2, channel, hideen = load_data(args.dataset, args.attack_label,
                                                                                   args.attributes, data_path,
                                                                                   batch_size)

    # Define model
    if model == 'lenet':
        if acti == 'leakyrelu':
            # Define client-side model
            client_model = Client_LeNet(channel=channel, hideen1=hideen, hideen2=num_cutlayer).to(device)
            client_model_1 = copy.deepcopy(client_model).to(device)
            client_model_2 = copy.deepcopy(client_model).to(device)
            optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=lr)
            optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=lr)

            # Define server-side model
            server_model = Server_LeNet(hideen2=num_cutlayer * 2, hideen3=256, hideen4=128, hideen5=64,
                                        num_classes=num_classes1).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)

        if acti == 'linear':
            # Define client-side model
            client_model = Client_LeNet_linear(channel=channel, hideen1=hideen, hideen2=num_cutlayer).to(device)
            client_model_1 = copy.deepcopy(client_model).to(device)
            client_model_2 = copy.deepcopy(client_model).to(device)
            optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=lr)
            optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=lr)

            # Define server-side model
            server_model = Server_LeNet(hideen2=num_cutlayer * 2, hideen3=256, hideen4=128, hideen5=64,
                                        num_classes=num_classes1).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)

    if model == 'resnet':
        if acti == 'leakyrelu':
            # Define client-side model
            client_model = ResNet18(num_classes=num_cutlayer)
            client_model_1 = copy.deepcopy(client_model).to(device)
            client_model_2 = copy.deepcopy(client_model).to(device)
            optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=lr)
            optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=lr)

            # Define server-side model
            server_model = Server_ResNet(hideen2=num_cutlayer * 2, hideen3=256, hideen4=128, hideen5=64,
                                         num_classes=num_classes1).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)


    # Define criterion
    criterion = nn.CrossEntropyLoss()

    # start training
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------", file=filename)
        train_client(train_data, client_model_1, client_model_2, t)
        test_client(test_data, client_model_1, client_model_2, t)
    print("Done!", file=filename)







