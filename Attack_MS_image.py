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
    parser = argparse.ArgumentParser(description='VFL1_attack_Model_Steal') # [utkface, celebA]
    parser.add_argument('--dataset', type=str, default='utkface', help="dataset")
    parser.add_argument('--model', type=str, default='lenet', help="model")
    parser.add_argument('--acti', type=str, default='leakyrelu', help="acti")
    parser.add_argument('--num_smooth_epoch', default=5, type=int, help='num_smooth_epoch')
    parser.add_argument('--attack_mode', type=str, default='graybox', help="attack_mode")
    parser.add_argument('--attack_label', type=int, default='0', help="num of l_class_num")
    parser.add_argument('--attributes', type=str, default="race_gender", help="For attrinf, two attributes should be in format x_y e.g. race_gender")
    parser.add_argument('--num_shadow', default=100, type=int, help='num_shadow')
    parser.add_argument('--lr', default=1e-4, type=float, help='lr')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--num_recover', default=1, type=int, help='num_recover')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--num_cutlayer', default=1000, type=int, help='num_cutlayer')
    parser.add_argument('--num_bs', default=20, type=int, help='num_bs')
    parser.add_argument('--attack_time', default=50, type=int, help='attack_time')
    parser.add_argument('--attack_batch', default=10, type=int, help='attack_batch')
    return parser.parse_args(argv[1:])

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


# Train_Server Side Program
def true_server(client1_fx, client2_fx, Y_1, t,  batch_id, correct, size, sum_batch):
    save_server = f'Results/{dataset}/{model}/server_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time}_b{attack_batch}.pth'
    server_model = torch.load(save_server)
    global train_true
    global test_true


    correct = correct
    fx_server = server_model(client1_fx, client2_fx) 
    if t ==0:
        train_true.extend(fx_server.argmax(1).tolist())
    if t ==1:
        test_true.extend(fx_server.argmax(1).tolist())

    # backward
    correct += (fx_server.argmax(1) == Y_1).type(torch.float).sum().item()

    correct_train = correct / size
    current = (batch_id+1) * len(client1_fx)
    if batch_id == sum_batch-1:
      print('loss-main', criterion(fx_server, Y_1), file=filename)
      print(f"acc: [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%", file=filename)

    if t ==0:
       train_true_save = copy.deepcopy(np.asarray(train_true))
       np.save(f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/train_true_c{num_cutlayer}_{acti}_{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}.npy', train_true_save)

    if t ==1:
       test_true_save = copy.deepcopy(np.asarray(test_true))
       np.save(f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/test_true_c{num_cutlayer}_{acti}_{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}.npy', test_true_save)


    return correct

# Train_Client Side Program
def true_client(dataloader, client_model_1, client_model_2, t):

    save_path1 = f'Results/{dataset}/{model}/client1_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time}_b{attack_batch}.pth'
    save_path2 = f'Results/{dataset}/{model}/client2_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time}_b{attack_batch}.pth'
            
    client_model_1 = torch.load(save_path1)
    client_model_2 = torch.load(save_path2) 

    correct = 0
    size = len(dataloader.dataset)
    sum_batch = len(dataloader)


    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X = X.to(device)
        X_1, X_2 = split_data(X)
        Y_1 = target[0].to(device)

        # client1--train and update
        fx1 = client_model_1(X_1)
        fx2 = client_model_2(X_2)

        # Sending activations to server and receiving gradients from server
        correct = true_server(fx1, fx2, Y_1, t,  batch_id, correct, size, sum_batch)


# Train_Server Side Program
def fake_server(client1_fx, client2_fx, Y_1, t,  batch_id, correct, size, sum_batch):
    save_server = f'Results/{dataset}/{model}/server_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time}_b{attack_batch}.pth'
    server_model = torch.load(save_server)
    global train_fake
    global test_fake


    correct = correct
    fx_server = server_model(client1_fx, client2_fx) 
    if t ==0:
        train_fake.extend(fx_server.argmax(1).tolist())
    if t ==1:
        test_fake.extend(fx_server.argmax(1).tolist())

    # backward
    correct += (fx_server.argmax(1) == Y_1).type(torch.float).sum().item()

    correct_train = correct / size
    current = (batch_id+1) * len(client1_fx)
    if batch_id == sum_batch-1:
      print('train—loss-main', criterion(fx_server, Y_1), file=filename)
      print(f"train-loss: [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%", file=filename)

    if t ==0:
       train_fake_save = copy.deepcopy(np.asarray(train_fake))
       np.save(f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/train_fake_c{num_cutlayer}_{acti}_{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}.npy', train_fake_save)

    if t ==1:
       test_fake_save = copy.deepcopy(np.asarray(test_fake))
       np.save(f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/test_fake_c{num_cutlayer}_{acti}_{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}.npy', test_fake_save)

    return correct

# Train_Client Side Program
def fake_client(dataloader, client_model_1, client_model_2, t):
    save_path1 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/fake_model1_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}_shadow2.pth'
    save_path2 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_2/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/fake_model2_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}_shadow2.pth'

    client_model_1 = torch.load(save_path1)
    client_model_2 = torch.load(save_path2) 

    correct = 0
    sum_batch = len(dataloader)
    size = len(dataloader.dataset)

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X = X.to(device)
        X_1, X_2 = split_data(X)
        Y_1 = target[0].to(device)

        # client1--train and update
        fx1 = client_model_1(X_1)
        fx2 = client_model_2(X_2)

        # Sending activations to server and receiving gradients from server
        correct = fake_server(fx1, fx2, Y_1, t,  batch_id, correct, size, sum_batch)


if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    dataset = args.dataset
    model = args.model
    num_smooth_epoch = args.num_smooth_epoch
    batch_size = args.batch_size
    epochs = args.epochs
    num_shadow = args.num_shadow
    lr = args.lr
    attack_mode = args.attack_mode
    acti = args.acti
    attributes = args.attributes
    attack_label = args.attack_label
    num_cutlayer =args.num_cutlayer
    attack_time = args.attack_time
    attack_batch = args.attack_batch
    num_recover = args.num_recover
    num_bs = args.num_bs
    time_start_load_everything = time.time()


    # Define record path
    save_path = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/Aggrement_{acti}_{batch_size}_epoch{attack_time}.txt', 'w+')

    ### Load data 
    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    train_data, test_data, num_classes1, num_classes2, channel, hideen = load_data(args.dataset, args.attack_label, args.attributes, data_path, batch_size)
    
    # Define model
    if attack_mode == 'graybox':
        if model == 'lenet':
            if acti == 'leakyrelu':
                # Define client-side model
                client_model = Client_LeNet(channel=channel, hideen1=hideen, hideen2=num_cutlayer).to(device)
                client_model_1 = copy.deepcopy(client_model).to(device)
                client_model_2 = copy.deepcopy(client_model).to(device)
                # Define server-side model
                server_model = Server_LeNet(hideen2=num_cutlayer*2, hideen3=256, hideen4=128, hideen5=64, num_classes=num_classes1).to(device)

            if acti == 'linear':
                # Define client-side model
                client_model = Client_LeNet_linear(channel=channel, hideen1=hideen, hideen2=num_cutlayer).to(device)
                client_model_1 = copy.deepcopy(client_model).to(device)
                client_model_2 = copy.deepcopy(client_model).to(device)

                # Define server-side model
                server_model = Server_LeNet(hideen2=num_cutlayer*2, hideen3=256, hideen4=128, hideen5=64, num_classes=num_classes1).to(device)

        if model == 'resnet':
            if acti == 'leakyrelu':
                # Define client-side model
                client_model = ResNet18(num_classes=num_cutlayer)
                client_model_1 = copy.deepcopy(client_model).to(device)
                client_model_2 = copy.deepcopy(client_model).to(device)

                # Define server-side model
                server_model = Server_ResNet(hideen2=num_cutlayer*2, hideen3=256, hideen4=128, hideen5=64, num_classes=num_classes1).to(device)

    if attack_mode == 'blackbox':
        if model == 'lenet':
            if acti == 'leakyrelu':
                # Define client-side model
                client_model = Client_LeNet(channel=channel, hideen1=hideen, hideen2=num_cutlayer).to(device)
                client_model_1 = copy.deepcopy(client_model).to(device)
                client_model_2 = copy.deepcopy(client_model).to(device)
                # Define client-side model
                client_model_s = general_LeNet(channel=channel, hideen1=hideen, hideen2=num_cutlayer).to(device)
                client_model_1_s = copy.deepcopy(client_model_s).to(device)
                client_model_2_s = copy.deepcopy(client_model_s).to(device)

                # Define server-side model
                server_model = Server_LeNet(hideen2=num_cutlayer * 2, hideen3=256, hideen4=128, hideen5=64,
                                            num_classes=num_classes1).to(device)

            if acti == 'linear':
                # Define client-side model
                client_model = Client_LeNet_linear(channel=channel, hideen1=hideen, hideen2=num_cutlayer).to(device)
                client_model_1 = copy.deepcopy(client_model).to(device)
                client_model_2 = copy.deepcopy(client_model).to(device)
                # Define client-side model
                client_model_s = general_LeNet(channel=channel, hideen1=hideen, hideen2=num_cutlayer).to(device)
                client_model_1_s = copy.deepcopy(client_model_s).to(device)
                client_model_2_s = copy.deepcopy(client_model_s).to(device)

                # Define server-side model
                server_model = Server_LeNet(hideen2=num_cutlayer * 2, hideen3=256, hideen4=128, hideen5=64,
                                            num_classes=num_classes1).to(device)
            if model == 'resnet':
                if acti == 'leakyrelu':
                    # Define client-side model
                    client_model = ResNet18(num_classes=num_cutlayer)
                    client_model_1 = copy.deepcopy(client_model).to(device)
                    client_model_2 = copy.deepcopy(client_model).to(device)

                    client_model_s = ResNet18_general(num_classes=num_cutlayer)
                    client_model_1_s = copy.deepcopy(client_model_s).to(device)
                    client_model_2_s = copy.deepcopy(client_model_s).to(device)

                    # Define server-side model
                    server_model = Server_ResNet(hideen2=num_cutlayer * 2, hideen3=256, hideen4=128, hideen5=64,
                                                 num_classes=num_classes1).to(device)

    criterion = nn.CrossEntropyLoss()
    # start training
    for t in range(1):
        train_true = []
        test_true = []
        train_fake = []
        test_fake = []

        if attack_mode == 'graybox':
            true_client(train_data, client_model_1, client_model_2, 0)
            true_client(test_data, client_model_1, client_model_2, 1)

            fake_client(train_data, client_model_1, client_model_2, 0)
            fake_client(test_data, client_model_1, client_model_2, 1)

        if attack_mode == 'blackbox':
            true_client(train_data, client_model_1, client_model_2, 0)
            true_client(test_data, client_model_1, client_model_2, 1)

            fake_client(train_data, client_model_1_s, client_model_2, 0)
            fake_client(test_data, client_model_1_s, client_model_2, 1)


    train_aggrement = 0
    test_aggrement = 0
    for i in range(len(train_true)):
        if train_true[i]==train_fake[i]:
            train_aggrement +=1

    for i in range(len(test_true)):
        if test_true[i]==test_fake[i]:
            test_aggrement +=1

    print('len(train_true)', len(train_true),  file=filename)
    print('len(test_true)', len(test_true),  file=filename)

    print('train_aggrement', train_aggrement,  file=filename)
    print('train_aggrement', train_aggrement/ len(train_true),  file=filename)

    print('test_aggrement', test_aggrement,  file=filename)
    print('test_aggrement', test_aggrement/ len(test_true),  file=filename)


    print("Done!", file=filename)

    



