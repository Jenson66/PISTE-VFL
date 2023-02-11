import random
import time
from model.Linear_NN_2 import *
from utils_tabular import *
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
    parser.add_argument('--acti', type=str, default='non', help="acti")  # [leakrelu, non]
    parser.add_argument('--attack_mode', type=str, default='graybox', help="attack_mode")
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--num_cutlayer', default=200, type=int, help='num_cutlayer')
    parser.add_argument('--num_shadow', default=100, type=int, help='num_shadow')
    parser.add_argument('--number_client', default=2, type=int, help='number_client')
    parser.add_argument('--num_smooth_epoch', default=1, type=int, help='num_smooth_epoch')
    parser.add_argument('--num_recover', default=1, type=int, help='num_recover')
    parser.add_argument('--noise_scale', default=0.0, type=float, help='noise_scale')
    parser.add_argument('--Iteration', default=30, type=int, help='Iteration') #3000
    parser.add_argument('--attack_time', default=48, type=int, help='attack_time')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--bs', default=10, type=int, help='bs')
    parser.add_argument('--lr', default=1e-1, type=float, help='lr')  # [1e-4, ]
    return parser.parse_args(argv[1:])

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


def value_class(x):
    z1 = x.clone()
    z2 = x.clone()
    zeros = torch.zeros_like(z1)
    z1 = torch.where(z1>=0, zeros, z1)
    z2 = torch.where(z2<=11, zeros, z2)
    loss = torch.norm(z1) + torch.norm(z2)
    return loss.to(device)

def value_reg(x):
    z1 = x.clone()
    z2 = x.clone()
    zeros = torch.zeros_like(z1)
    z1 = torch.where(z1>=0, zeros, z1)
    z2 = torch.where(z2<=1, zeros, z2)
    loss = torch.norm(z1) + torch.norm(z2)
    return loss.to(device)


if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    batch_size = args.batch_size
    num_cutlayer = args.num_cutlayer
    epochs = args.epochs
    lr = args.lr
    bs = args.bs
    num_recover = args.num_recover
    noise_scale = args.noise_scale
    dataset=args.dataset
    number_client = args.number_client
    num_smooth_epoch = args.num_smooth_epoch
    num_shadow =args.num_shadow
    Iteration= args.Iteration
    attack_time = args.attack_time
    acti = args.acti
    attack_mode = args.attack_mode
    time_start_load_everything = time.time()

    #Define record path
    save_path3 = f'Results_attack/{dataset}/num_client{number_client}/{attack_mode}_smooth{num_smooth_epoch}/file/recover{num_recover}'
    if not os.path.exists(save_path3):
        os.makedirs(save_path3)
    filename=open(f'Results_attack/{dataset}/num_client{number_client}/{attack_mode}_smooth{num_smooth_epoch}/{dataset}_c{num_cutlayer}_{acti}_recover{num_recover}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_b{bs}_0.txt', 'w+')
 

    # Load shadow data
    num_b = 1
    for i in range(num_smooth_epoch):
        for j in range(num_b):
            x1_true = np.load(f'Results/{dataset}/num_client{number_client}/X1_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time+(int(num_smooth_epoch/2))-i}_b{j}.npy')
            x2_true = np.load(f'Results/{dataset}/num_client{number_client}/X2_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time+(int(num_smooth_epoch/2))-i}_b{j}.npy')
            yb1_true = np.load(f'Results/{dataset}/num_client{number_client}/fx1_before{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time+(int(num_smooth_epoch/2))-i}_b{j}.npy')
            yb2_true = np.load(f'Results/{dataset}/num_client{number_client}/fx2_before{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time+(int(num_smooth_epoch/2))-i}_b{j}.npy')

            if i == 0:
                yb1_true_sum = yb1_true
                yb2_true_sum = yb2_true
            else:
                yb1_true_sum += yb1_true
                yb2_true_sum += yb2_true

    x1_true = torch.tensor(x1_true).to(device).requires_grad_(True) 
    x2_true = torch.tensor(x2_true).to(device).requires_grad_(True)

    yb1_true_sum = torch.tensor(yb1_true_sum/num_smooth_epoch).to(device).requires_grad_(True) 
    yb2_true_sum = torch.tensor(yb2_true_sum/num_smooth_epoch).to(device).requires_grad_(True) 

    # Define data
    x1_syn = x1_true[:num_shadow,:]
    x1_victim = x1_true[num_shadow:,:]

    x2_syn = x2_true[:num_shadow,:]
    x2_victim = x2_true[num_shadow:,:]

    yb1_smooth = yb1_true_sum[:num_shadow,:]
    yb1_victim = yb1_true_sum[num_shadow:,:]

    yb2_smooth = yb2_true_sum[:num_shadow,:]
    yb2_victim = yb2_true_sum[num_shadow:,:]


    # # Define data
    x1_victim_true = copy.deepcopy(x1_victim.detach().cpu().numpy())
        
    np.save(f'Results_attack/{dataset}/num_client{number_client}/{attack_mode}_smooth{num_smooth_epoch}/file/x_victim_1_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}.npy', x1_victim_true)
    x2_victim_true = copy.deepcopy(x2_victim.detach().cpu().numpy())
    np.save(f'Results_attack/{dataset}/num_client{number_client}/{attack_mode}_smooth{num_smooth_epoch}/file/x_victim_2_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}.npy', x2_victim_true)

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



    # Define optimizer_m1
    optimizer_m1 = torch.optim.Adam(client_model1.parameters(),  lr=0.0001)   
    for iters in range(Iteration*4):
        optimizer_m1.zero_grad()
        yb1_fake = client_model1(x1_syn, 1)
        loss1 = ((yb1_fake - yb1_smooth) ** 2).sum()
        loss = loss1 

        loss.backward()
        optimizer_m1.step()

        if iters%500 == 0 or iters == (Iteration*4-1):
            print('current_loss_m1:', loss.item(), file=filename)
  
    save_path1 = f'Results_attack/{dataset}/num_client{number_client}/{attack_mode}_smooth{num_smooth_epoch}/model1_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_r_{num_recover}_epoch{attack_time}.pth'
    torch.save(client_model1, save_path1)
    del loss1, loss


    # Define optimizer_m2
    optimizer_m2 = torch.optim.Adam(client_model2.parameters(),  lr=0.0001)   
    for iters in range(Iteration*4):
        optimizer_m2.zero_grad()
        yb2_fake = client_model2(x2_syn, 2)
        loss1 = ((yb2_fake - yb2_smooth) ** 2).sum()
        loss = loss1 

        loss.backward()
        optimizer_m2.step()

        if iters%500 == 0 or iters == (Iteration*4-1):
            print('current_loss_m2:', loss.item(), file=filename)
  
    save_path2 = f'Results_attack/{dataset}/num_client{number_client}/{attack_mode}_smooth{num_smooth_epoch}/model2_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_r_{num_recover}_epoch{attack_time}.pth'
    torch.save(client_model2, save_path2)
    del loss1, loss

   


    ####### attack victim data
    mse_list = []
    error_list = []
    iis = int(28/num_recover)
    for i in range(iis):
        # Define data
        x_1 = torch.rand(size=(num_recover, x1_victim.shape[1]) ).float()
        x_1 = x_1.to(device).requires_grad_(True)
        client_model1 = torch.load(save_path1)

        # Define optimizer1
        optimizer1 = torch.optim.LBFGS([x_1, ], lr=0.1)   
        for iters in range(Iteration):
            def closure():
                optimizer1.zero_grad()
                yb1_fake = client_model1(x_1, 1)
                loss1 = ((yb1_fake - yb1_victim[num_recover*i:num_recover*(i+1),:]) ** 2).sum()
                loss = loss1 + value_reg(x_1) 
                loss.backward()
                if iters%500 == 0 or iters ==Iteration-1:
                    print('current_loss_1:', loss.item(), file=filename)
                    print('mse_1:', loss1.item(), file=filename)

                return loss
            optimizer1.step(closure)
            current_loss = closure().item()


        print('x1_fake:', x_1, file=filename)
        print('x1_victim:', x1_victim[num_recover*i:num_recover*(i+1),:], file=filename)
        mse1 = ((x_1 - x1_victim[num_recover*i:num_recover*(i+1),:]) ** 2).sum()
        print('i', i, file=filename)
        print('mse1:', mse1/num_recover/len(x_1[0,:]), file=filename)

        mse_list.append( (mse1/num_recover/len(x_1[0,:])).item())
        print('mse_list', mse_list, file=filename)
        X1_fake = copy.deepcopy(x_1.detach().cpu().numpy())

        np.save(f'Results_attack/{dataset}/num_client{number_client}/{attack_mode}_smooth{num_smooth_epoch}/file/recover{num_recover}/X_fake_1_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_recovery{i}.npy', X1_fake)

        print('------------------------------------------------', file=filename)

    

    print('------------------------------------------------', file=filename)
    print('mse_list', mse_list, file=filename)
    print('mean(mse_list)', np.mean(mse_list), file=filename)
    print('std(mse_list)', np.std(mse_list), file=filename)

    print('==============================================', file=filename)



