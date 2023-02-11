import random
import time
import matplotlib.pyplot as plt
from model.lenet import *
from Generator.model import *
from torch.nn.utils import clip_grad_norm_ 
from torch import nn
from sys import argv
import os
import argparse
import copy
from torchvision.transforms import transforms


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
	parser = argparse.ArgumentParser(description='VFL_DR')
	parser.add_argument('--model', type=str, default='lenet', help="model")
	parser.add_argument('--acti', type=str, default='leakyrelu', help="acti")
	parser.add_argument('--dataset', type=str, default='utkface', help="dataset")
	parser.add_argument('--num_smooth_epoch', default=50, type=int, help='num_smooth_epoch')
	parser.add_argument('--attack_mode', type=str, default='graybox', help="attack_mode")
	parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
	parser.add_argument('--num_cutlayer', default=1000, type=int, help='num_cutlayer')
	parser.add_argument('--Iteration', default=600, type=int, help='Iteration')
	parser.add_argument('--num_shadow', default=100, type=int, help='num_shadow')
	parser.add_argument('--epochs', default=100, type=int, help='epochs')
	parser.add_argument('--num_exp', default=1, type=int, help='num_exp')
	parser.add_argument('--num_bs', default=20, type=int, help='num_bs')
	parser.add_argument('--attack_time', default=50, type=int, help='attack_time') 
	parser.add_argument('--num_recover', default=1, type=int, help='num_recover')
	parser.add_argument('--attack_batch', default=10, type=int, help='attack_batch')
	parser.add_argument('--lr', default=1e-1, type=float, help='lr')
	return parser.parse_args(argv[1:])

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def x_value2(x):
	z1 = x.clone()
	z2 = x.clone()
	zeros = torch.zeros_like(z1)
	z1 = 2*torch.where(z1>0, zeros, z1)
	z2 = torch.where(z2<1, zeros, z2)
	loss = (torch.norm(z1) + torch.norm(z2)).to(device)
	return loss

class TVLoss(nn.Module):
		def __init__(self,TVLoss_weight=1):
			super(TVLoss,self).__init__()
			self.TVLoss_weight = TVLoss_weight

		def forward(self,x):
			batch_size = x.size()[0]
			h_x = x.size()[2]
			w_x = x.size()[3]
			count_h = self._tensor_size(x[:,:,1:,:])
			count_w = self._tensor_size(x[:,:,:,1:])
			h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
			w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
			return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
				
		@staticmethod
		def _tensor_size(t):
				return t.size()[1]*t.size()[2]*t.size()[3]

tv_loss = TVLoss()


if __name__ == '__main__':
	print('Start training')
	args = parse_args()
	model = args.model
	num_smooth_epoch = args.num_smooth_epoch
	batch_size = args.batch_size
	num_cutlayer = args.num_cutlayer
	epochs = args.epochs
	lr = args.lr
	attack_mode = args.attack_mode
	dataset=args.dataset
	num_bs = args.num_bs
	num_recover = args.num_recover
	num_shadow =args.num_shadow
	num_exp =args.num_exp
	acti = args.acti
	Iteration = args.Iteration
	attack_time = args.attack_time
	attack_batch = args.attack_batch
	time_start_load_everything = time.time()
	tp = transforms.Compose([transforms.ToPILImage()])
	plot_num = 30	

	if acti =='linear':
		P_tv =1e-1
	if acti == 'leakyrelu':
		P_tv = 1

	# Define record path
	save_path_1 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/'
	save_path_2 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_2/sb_{num_bs}/s_{num_shadow}/r_{num_recover}'
	if not os.path.exists(save_path_1):
		os.makedirs(save_path_1)
	if not os.path.exists(save_path_2):
		os.makedirs(save_path_2)
	filename1 = open(f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/{acti}_{batch_size}_epoch{attack_time}.txt', 'w+')
	filename2 = open(f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_2/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/{acti}_{batch_size}_epoch{attack_time}.txt', 'w+')

	for i in range(num_smooth_epoch):
		for num_b in range(num_bs):
			# Load data
			x1_true = np.load(
				f'Results/{dataset}/{model}/X1_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time + (int(num_smooth_epoch / 2)) - i}_b{num_b}.npy')
			x2_true = np.load(
				f'Results/{dataset}/{model}/X2_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time + (int(num_smooth_epoch / 2)) - i}_b{num_b}.npy')
			yb1 = np.load(
				f'Results/{dataset}/{model}/fx1_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time + (int(num_smooth_epoch / 2)) - i}_b{num_b}.npy')
			yb2 = np.load(
				f'Results/{dataset}/{model}/fx2_c{num_cutlayer}_{acti}_{batch_size}_epoch{attack_time + (int(num_smooth_epoch / 2)) - i}_b{num_b}.npy')

			if num_b == 0:
				yb1_sum = yb1
				yb2_sum = yb2
				x1_true_sum = x1_true
				x2_true_sum = x2_true
			else:
				yb1_sum = np.concatenate((yb1_sum, yb1), axis=0)
				x1_true_sum = np.concatenate((x1_true_sum, x1_true), axis=0)

				yb2_sum = np.concatenate((yb2_sum, yb2), axis=0)
				x2_true_sum = np.concatenate((x2_true_sum, x2_true), axis=0)

		if i == 0:
			yb1_sum_smooth = yb1_sum
			yb2_sum_smooth = yb2_sum
		else:
			yb1_sum_smooth += yb1_sum
			yb2_sum_smooth += yb2_sum



		
	x1_true_sum = torch.tensor(x1_true_sum).to(device) 
	x2_true_sum = torch.tensor(x2_true_sum).to(device) 
	yb1_sum = torch.tensor(yb1_sum).to(device) 
	yb2_sum = torch.tensor(yb2_sum).to(device) 
	

	# Define data
	s_num = int(num_bs/2)
	x1_shadow1_1 = x1_true_sum[:batch_size*s_num,:]
	x1_shadow1_2 = x1_true_sum[batch_size*(s_num+1):,:]
	x1_shadow1 = torch.cat((x1_shadow1_1, x1_shadow1_2), axis=0)
	x1_shadow2 = x1_true_sum[batch_size*s_num:batch_size*s_num+num_shadow,:]
	x1_victim = x1_true_sum[batch_size*s_num+num_shadow : batch_size*(s_num+1),:]

	yb1_1 = yb1_sum[:batch_size*s_num,:]
	yb1_2 = yb1_sum[batch_size*(s_num+1):,:]
	yb1_shadow1 = torch.cat((yb1_1, yb1_2), axis=0)
	yb1_victim = yb1_sum[batch_size*s_num+num_shadow:batch_size*(s_num+1),:]
	yb1_shadow2 = yb1_sum[batch_size*s_num:batch_size*s_num+num_shadow,:]


	x1_shadow1 = x1_shadow1.to(device).requires_grad_(True)
	x1_shadow2 = x1_shadow2.to(device).requires_grad_(True)
	x1_victim = x1_victim.to(device).requires_grad_(True)

	yb1_shadow1 = yb1_shadow1.to(device).requires_grad_(True)
	yb1_shadow2 = yb1_shadow2.to(device).requires_grad_(True)
	yb1_victim = yb1_victim.to(device).requires_grad_(True)


	# Define data
	x2_shadow1_1 = x2_true_sum[:batch_size*s_num,:]
	x2_shadow1_2 = x2_true_sum[batch_size*(s_num+1):,:]
	x2_shadow1 = torch.cat((x2_shadow1_1, x2_shadow1_2), axis=0)
	x2_victim = x2_true_sum[batch_size*s_num+num_shadow:batch_size*(s_num+1),:]
	x2_shadow2 = x2_true_sum[batch_size*s_num:batch_size*s_num+num_shadow,:]

	yb2_1 = yb2_sum[:batch_size*s_num,:]
	yb2_2 = yb2_sum[batch_size*(s_num+1):,:]
	yb2_shadow1 = torch.cat((yb2_1, yb2_2), axis=0)
	yb2_victim = yb2_sum[batch_size*s_num+num_shadow:batch_size*(s_num+1),:]
	yb2_shadow2 = yb2_sum[batch_size*s_num:batch_size*s_num+num_shadow,:]


	x2_shadow1 = x2_shadow1.to(device).requires_grad_(True)
	x2_shadow2 = x2_shadow2.to(device).requires_grad_(True)
	x2_victim = x2_victim.to(device).requires_grad_(True)
	yb2_shadow1 = yb2_shadow1.to(device).requires_grad_(True)
	yb2_shadow2 = yb2_shadow2.to(device).requires_grad_(True)
	yb2_victim = yb2_victim.to(device).requires_grad_(True)


	#### Define data
	x_1 = torch.randn(size=x1_victim.shape).float()
	x_1 = x_1.to(device).requires_grad_(True)

	x_2 = torch.randn(size=x2_victim.shape).float()
	x_2 = x_2.to(device).requires_grad_(True)


	# Define model
	if attack_mode == 'graybox':
		if model == 'lenet':
			if acti == 'leakyrelu':
				# Define client-side model
				client_model = Client_LeNet(channel=3, hideen1=768, hideen2=num_cutlayer).to(device)
				client_model1 = copy.deepcopy(client_model).to(device)
				client_model2 = copy.deepcopy(client_model).to(device)

			if acti == 'linear':
				# Define client-side model
				client_model = Client_LeNet_linear(channel=3, hideen1=768, hideen2=num_cutlayer).to(device)
				client_model1 = copy.deepcopy(client_model).to(device)
				client_model2 = copy.deepcopy(client_model).to(device)

	if attack_mode == 'blackbox':
		# Define model
		if model == 'lenet':
			client_model = general_LeNet(channel=3, hideen1=768, hideen2=num_cutlayer).to(device)
			client_model1 = copy.deepcopy(client_model).to(device)
			client_model2 = copy.deepcopy(client_model).to(device)

	# steal client_model1_shadow1
	optimizer_m1 = torch.optim.Adam(client_model1.parameters(), lr=0.0001)   
	for iters in range(Iteration*8):
		client_model1.train()
		for i in range(num_bs-1):
			yb1_fake = client_model1(x1_shadow1[batch_size*i:batch_size*(i+1),:,:,:])
			loss1 = ((yb1_fake - yb1_shadow1[batch_size*i:batch_size*(i+1),:]) ** 2).sum()
			loss = loss1 

			optimizer_m1.zero_grad()
			loss.backward()
			optimizer_m1.step()

		if iters%500 == 0 or iters ==(Iteration*8-1) :
			print('current_loss_m1:', loss.item(), file=filename1)

	save_path1 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/fake_model1_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}_shadow1.pth'
	torch.save(client_model1, save_path1)
	del loss1, loss

	# steal client_model2_shadow1
	optimizer_m2 = torch.optim.Adam(client_model2.parameters(), lr=0.0001)
	for iters in range(Iteration * 8):
		client_model2.train()
		for i in range(num_bs - 1):
			yb2_fake = client_model2(x2_shadow1[batch_size * i:batch_size * (i + 1), :, :, :])
			loss1 = ((yb2_fake - yb2_shadow1[batch_size * i:batch_size * (i + 1), :]) ** 2).sum()
			loss = loss1

			optimizer_m2.zero_grad()
			loss.backward()
			optimizer_m2.step()

		if iters % 500 == 0 or iters == (Iteration * 8 - 1):
			print('current_loss_m2:', loss.item(), file=filename2)

	save_path2 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_2/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/fake_model2_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}_shadow1.pth'
	torch.save(client_model2, save_path2)
	del loss1, loss


	#### pretrain G1
	client_model1 = torch.load(save_path1)
	history = []
	for idx_exp in range(num_exp):
		g_in = 128
		b1_shadow1 = yb1_shadow1.shape[0]
		G_ran_in = torch.randn(b1_shadow1, g_in).to(device)# initialize GRNN input
		Gnet1 = Generator(channel=3, shape_img=32, batchsize=batch_size, g_in=g_in, iters=0).to(device)
		G_optimizer = torch.optim.RMSprop(Gnet1.parameters(), lr=0.0001, momentum=0.9)

		for iters in range(Iteration*2):
			for i in range(num_bs-1):
				Gout = Gnet1(G_ran_in[batch_size*i:batch_size*(i+1),:], batch_size, 0) # produce recovered data
				Gout = Gout.to(device)

				yb1_fake = client_model1(Gout)
				loss1 = ((yb1_fake - yb1_shadow1[batch_size*i:batch_size*(i+1),:]) ** 2).sum()
				loss = loss1 + x_value2(Gout[:,:,:,16:]) + P_tv* tv_loss(Gout[:,:,:,16:])

				G_optimizer.zero_grad()
				loss.backward()
				clip_grad_norm_(Gnet1.parameters(), max_norm=5, norm_type=2)
				G_optimizer.step()

				if iters % int(Iteration*2 / plot_num) == 0 and i==0:
					history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(batch_size)])

		save_G1 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/G1_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}_shadow1.pth'
		torch.save(Gnet1, save_G1)

		# visualization
		for imidx in range(batch_size):
			plt.figure(figsize=(12, 8))
			plt.subplot(plot_num//10, 10, 1)
			plt.imshow(tp(x1_shadow1[imidx].cpu()))
			for i in range(min(len(history), plot_num-1)):
				plt.subplot(plot_num//10, 10, i + 2)
				plt.imshow(history[i][imidx])
				plt.axis('off')

			if True:
				save_path = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/{acti}/attack_shadow1/{attack_time}/'
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				true_path = os.path.join(save_path, f'true_data_pretrain/')
				fake_path = os.path.join(save_path, f'fake_data_pretrain/')
				if not os.path.exists(true_path) or not os.path.exists(fake_path):
					os.makedirs(true_path)
					os.makedirs(fake_path)
				tp(x1_shadow1[imidx].cpu()).save(os.path.join(true_path, f'{num_b}_{imidx}.png'))
				history[-1][imidx].save(os.path.join(fake_path, f'{num_b}_{imidx}.png'))
				plt.savefig(save_path + '/pretrain_exp:%03d-imidx:%02d_b:%02d.png' % (idx_exp, imidx, num_b))
				plt.close()

	del loss1,  loss, Gout, history
	print('=======================================================', file=filename1)

	#### pretrain G2
	client_model2 = torch.load(save_path2)
	history = []
	for idx_exp in range(num_exp):
		g_in = 128
		b2_shadow1 = yb2_shadow1.shape[0]
		G_ran_in = torch.randn(b2_shadow1, g_in).to(device)  # initialize GRNN input
		Gnet2 = Generator(channel=3, shape_img=32, batchsize=batch_size, g_in=g_in, iters=0).to(device)
		G_optimizer = torch.optim.RMSprop(Gnet2.parameters(), lr=0.0001, momentum=0.9)

		for iters in range(Iteration * 2):
			for i in range(num_bs - 1):
				Gout = Gnet2(G_ran_in[batch_size * i:batch_size * (i + 1), :], batch_size, 0)
				Gout = Gout.to(device)

				yb2_fake = client_model2(Gout)
				loss1 = ((yb2_fake - yb2_shadow1[batch_size * i:batch_size * (i + 1), :]) ** 2).sum()
				loss = loss1 + x_value2(Gout[:, :, :, :16]) + P_tv * tv_loss(Gout[:, :, :, :16])

				G_optimizer.zero_grad()
				loss.backward()
				clip_grad_norm_(Gnet2.parameters(), max_norm=5, norm_type=2)
				G_optimizer.step()

				if iters % int(Iteration * 2 / plot_num) == 0 and i == 0:
					history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(batch_size)])

		save_G2 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_2/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/G2_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}_shadow1.pth'
		torch.save(Gnet2, save_G2)

		# visualization
		for imidx in range(batch_size):
			plt.figure(figsize=(12, 8))
			plt.subplot(plot_num // 10, 10, 1)
			plt.imshow(tp(x2_shadow1[imidx].cpu()))
			for i in range(min(len(history), plot_num - 1)):
				plt.subplot(plot_num // 10, 10, i + 2)
				plt.imshow(history[i][imidx])
				plt.axis('off')

			if True:
				save_path = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_2/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/{acti}/attack_shadow1/{attack_time}/'
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				true_path = os.path.join(save_path, f'true_data_pretrain/')
				fake_path = os.path.join(save_path, f'fake_data_pretrain/')
				if not os.path.exists(true_path) or not os.path.exists(fake_path):
					os.makedirs(true_path)
					os.makedirs(fake_path)
				tp(x2_shadow1[imidx].cpu()).save(os.path.join(true_path, f'{num_b}_{imidx}.png'))
				history[-1][imidx].save(os.path.join(fake_path, f'{num_b}_{imidx}.png'))
				plt.savefig(save_path + '/pretrain_exp:%03d-imidx:%02d_b:%02d.png' % (idx_exp, imidx, num_b))
				plt.close()

	del loss1, loss, Gout, history
	print('=======================================================', file=filename2)


	# Steal client_model1_shadow2
	optimizer_m1 = torch.optim.Adam(client_model1.parameters(), lr=0.0001)   
	for iters in range(Iteration*5):
		client_model1.train()
		yb1_fake = client_model1(x1_shadow2)
		loss1 = ((yb1_fake - yb1_shadow2[:num_shadow,:]) ** 2).sum()
		loss = loss1 

		optimizer_m1.zero_grad()
		loss.backward()
		optimizer_m1.step()

		if iters%500==0 or iters==(Iteration*5-1):
			print('loss_m1:', loss.item(), file=filename1)
	save_path1 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/fake_model1_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}_shadow2.pth'
	torch.save(client_model1, save_path1)
	del loss1, loss

	# Steal client_model2_shadow2
	optimizer_m2 = torch.optim.Adam(client_model2.parameters(), lr=0.0001)
	for iters in range(Iteration * 5):
		client_model2.train()
		yb2_fake = client_model2(x2_shadow2)
		loss1 = ((yb2_fake - yb2_shadow2[:num_shadow, :]) ** 2).sum()
		loss = loss1

		optimizer_m2.zero_grad()
		loss.backward()
		optimizer_m2.step()

		if iters % 500 == 0 or iters == (Iteration * 5 - 1):
			print('loss_m2:', loss.item(), file=filename2)
	save_path2 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_2/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/fake_model2_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_b{attack_batch}_shadow2.pth'
	torch.save(client_model2, save_path2)
	del loss1, loss


	# pre G1_shadow2
	client_model1 = torch.load(save_path1)
	history = []
	for idx_exp in range(num_exp):
		g_in = 128
		b_shadow2 = (num_shadow)
		G_ran_in = torch.randn(b_shadow2, g_in).to(device)# initialize GRNN input
		Gnet11 = Generator(channel=3, shape_img=32, batchsize=(b_shadow2), g_in=g_in, iters=0).to(device)
		Gnet11 = torch.load(save_G1)	
		G_optimizer = torch.optim.RMSprop(Gnet11.parameters(), lr=0.0001, momentum=0.9)	
		for iters in range(Iteration*5):
			Gnet11.train()
			Gout = Gnet11(G_ran_in, b_shadow2, 0) # produce recovered data
			Gout = Gout.to(device)	
			yb1_fake = client_model1(Gout)
			loss1 = ((yb1_fake - yb1_shadow2) ** 2).sum()
			loss = loss1 + x_value2(Gout[:,:,:,16:]) + P_tv* tv_loss(Gout[:,:,:,16:])
			G_optimizer.zero_grad()
			loss.backward()
			clip_grad_norm_(Gnet11.parameters(), max_norm=5, norm_type=2)
			G_optimizer.step()
			if iters % int(Iteration*5 / plot_num) == 0:
				history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(b_shadow2)])

		save_G11 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/G1_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_b{num_b}_shadow2.pth'
		torch.save(Gnet11, save_G11)

		# visualization
		for imidx in range(b_shadow2):
			plt.figure(figsize=(12, 8))
			plt.subplot(plot_num//10, 10, 1)
			plt.imshow(tp(x1_shadow2[imidx].cpu()))
			for i in range(min(len(history), plot_num-1)):
					plt.subplot(plot_num//10, 10, i + 2)
					plt.imshow(history[i][imidx])
					plt.axis('off')	
			if True:
				save_path = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/{acti}/attack_shadow2/{attack_time}'
				if not os.path.exists(save_path):
						os.makedirs(save_path)
				true_path = os.path.join(save_path, f'true_data/')
				fake_path = os.path.join(save_path, f'fake_data/')
				if not os.path.exists(true_path) or not os.path.exists(fake_path):
						os.makedirs(true_path)
						os.makedirs(fake_path)
				tp(x1_shadow2[imidx].cpu()).save(os.path.join(true_path, f'{num_bs}_{imidx}.png'))
				history[-1][imidx].save(os.path.join(fake_path, f'{num_bs}_{imidx}.png'))
				plt.savefig(save_path + '/exp:%03d-imidx:%02d-bs:%02d.png' % (idx_exp, imidx,num_bs))
				plt.close()	
	del loss1, loss, Gout, history
	print('=======================================================', file=filename1)





	# pre G2_shadow2
	client_model2 = torch.load(save_path2)
	history = []
	for idx_exp in range(num_exp):
		g_in = 128
		b_shadow2 = (num_shadow)
		G_ran_in = torch.randn(b_shadow2, g_in).to(device)  # initialize GRNN input
		Gnet22 = Generator(channel=3, shape_img=32, batchsize=(b_shadow2), g_in=g_in, iters=0).to(device)
		Gnet22 = torch.load(save_G2)
		G_optimizer = torch.optim.RMSprop(Gnet22.parameters(), lr=0.0001, momentum=0.9)
		for iters in range(Iteration * 5):
			Gnet22.train()
			Gout = Gnet22(G_ran_in, b_shadow2, 0)  # produce recovered data
			Gout = Gout.to(device)
			yb2_fake = client_model2(Gout)
			loss1 = ((yb2_fake - yb2_shadow2) ** 2).sum()
			loss = loss1 + x_value2(Gout[:, :, :, 16:]) + P_tv * tv_loss(Gout[:, :, :, 16:])
			G_optimizer.zero_grad()
			loss.backward()
			clip_grad_norm_(Gnet22.parameters(), max_norm=5, norm_type=2)
			G_optimizer.step()
			if iters % int(Iteration * 5 / plot_num) == 0:
				history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(b_shadow2)])

		save_G22 = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_2/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/G2_c{num_cutlayer}_{acti}_bs{batch_size}_s{num_shadow}_epoch{attack_time}_b{num_b}_shadow2.pth'
		torch.save(Gnet22, save_G22)

		# visualization
		for imidx in range(b_shadow2):
			plt.figure(figsize=(12, 8))
			plt.subplot(plot_num // 10, 10, 1)
			plt.imshow(tp(x2_shadow2[imidx].cpu()))
			for i in range(min(len(history), plot_num - 1)):
				plt.subplot(plot_num // 10, 10, i + 2)
				plt.imshow(history[i][imidx])
				plt.axis('off')
			if True:
				save_path = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_2/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/{acti}/attack_shadow2/{attack_time}'
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				true_path = os.path.join(save_path, f'true_data/')
				fake_path = os.path.join(save_path, f'fake_data/')
				if not os.path.exists(true_path) or not os.path.exists(fake_path):
					os.makedirs(true_path)
					os.makedirs(fake_path)
				tp(x2_shadow2[imidx].cpu()).save(os.path.join(true_path, f'{num_bs}_{imidx}.png'))
				history[-1][imidx].save(os.path.join(fake_path, f'{num_bs}_{imidx}.png'))
				plt.savefig(save_path + '/exp:%03d-imidx:%02d-bs:%02d.png' % (idx_exp, imidx, num_bs))
				plt.close()
	del loss1, loss, Gout, history
	print('=======================================================', file=filename2)


	# steal x1
	for idx_exp in range(num_exp):
		hs = int((batch_size - num_shadow)/num_recover)
		for h in range(hs):
			history = []
			g_in = 128
			b_victim = num_recover
			G_ran_in = torch.randn(b_victim, g_in).to(device)# initialize GRNN input
			Gnet11 = Generator(channel=3, shape_img=32, batchsize=(b_victim), g_in=g_in, iters=0).to(device)
			Gnet11 = torch.load(save_G11)
			G_optimizer = torch.optim.RMSprop(Gnet11.parameters(), lr=0.0001, momentum=0.9)
			for iters in range(Iteration*5):
				Gnet11.train()
				Gout = Gnet11(G_ran_in, b_victim, 0) # produce recovered data
				Gout = Gout.to(device)
				yb1_fake = client_model1(Gout)
				loss1 = ((yb1_fake - yb1_victim[h*num_recover:(h+1)*num_recover, :]) ** 2).sum()
				loss = loss1 + x_value2(Gout[:,:,:,16:]) + P_tv* tv_loss(Gout[:,:,:,16:])
				G_optimizer.zero_grad()
				loss.backward()
				clip_grad_norm_(Gnet11.parameters(), max_norm=5, norm_type=2)
				G_optimizer.step()

				if iters % int(Iteration*5 / plot_num) == 0:
					history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(b_victim)])


			# visualization
			for imidx in range(b_victim):
				plt.figure(figsize=(12, 8))
				plt.subplot(plot_num//10, 10, 1)
				plt.imshow(tp(x1_victim[h*num_recover+imidx].cpu()))
				for i in range(min(len(history), plot_num-1)):
						plt.subplot(plot_num//10, 10, i + 2)
						plt.imshow(history[i][imidx])
						plt.axis('off')
				if True:
					save_path = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_1/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/{acti}/attack/{attack_time}'
					if not os.path.exists(save_path):
							os.makedirs(save_path)
					true_path = os.path.join(save_path, f'true_data/')
					fake_path = os.path.join(save_path, f'fake_data/')
					if not os.path.exists(true_path) or not os.path.exists(fake_path):
							os.makedirs(true_path)
							os.makedirs(fake_path)
					tp(x1_victim[h*num_recover+imidx].cpu()).save(os.path.join(true_path, f'{h}_{num_bs}_{imidx}.png'))
					history[-1][imidx].save(os.path.join(fake_path, f'{h}_{num_bs}_{imidx}.png'))
					plt.savefig(save_path + '/h:%03d-exp:%03d-imidx:%02d-bs:%02d.png' % (h, idx_exp, imidx,num_bs))
					plt.close()
	print('=======================================================', file=filename1)

	# steal x2
	for idx_exp in range(num_exp):
		hs = int((batch_size - num_shadow) / num_recover)
		for h in range(hs):
			history = []
			g_in = 128
			b_victim = num_recover
			G_ran_in = torch.randn(b_victim, g_in).to(device)  # initialize GRNN input
			Gnet22 = Generator(channel=3, shape_img=32, batchsize=(b_victim), g_in=g_in, iters=0).to(device)
			Gnet22 = torch.load(save_G22)
			G_optimizer = torch.optim.RMSprop(Gnet22.parameters(), lr=0.0001, momentum=0.9)
			for iters in range(Iteration * 5):
				Gnet22.train()
				Gout = Gnet22(G_ran_in, b_victim, 0)  # produce recovered data
				Gout = Gout.to(device)
				yb1_fake = client_model2(Gout)
				loss1 = ((yb1_fake - yb1_victim[h * num_recover:(h + 1) * num_recover, :]) ** 2).sum()
				loss = loss1 + x_value2(Gout[:, :, :, :16]) + P_tv * tv_loss(Gout[:, :, :, :16])
				G_optimizer.zero_grad()
				loss.backward()
				clip_grad_norm_(Gnet22.parameters(), max_norm=5, norm_type=2)
				G_optimizer.step()

				if iters % int(Iteration * 5 / plot_num) == 0:
					history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(b_victim)])

			# visualization
			for imidx in range(b_victim):
				plt.figure(figsize=(12, 8))
				plt.subplot(plot_num // 10, 10, 1)
				plt.imshow(tp(x2_victim[h * num_recover + imidx].cpu()))
				for i in range(min(len(history), plot_num - 1)):
					plt.subplot(plot_num // 10, 10, i + 2)
					plt.imshow(history[i][imidx])
					plt.axis('off')
				if True:
					save_path = f'Results_attack/{dataset}_{attack_mode}_smooth{num_smooth_epoch}/{model}_G/c{num_cutlayer}_2/sb_{num_bs}/s_{num_shadow}/r_{num_recover}/{acti}/attack/{attack_time}'
					if not os.path.exists(save_path):
						os.makedirs(save_path)
					true_path = os.path.join(save_path, f'true_data/')
					fake_path = os.path.join(save_path, f'fake_data/')
					if not os.path.exists(true_path) or not os.path.exists(fake_path):
						os.makedirs(true_path)
						os.makedirs(fake_path)
					tp(x2_victim[h * num_recover + imidx].cpu()).save(
						os.path.join(true_path, f'{h}_{num_bs}_{imidx}.png'))
					history[-1][imidx].save(os.path.join(fake_path, f'{h}_{num_bs}_{imidx}.png'))
					plt.savefig(save_path + '/h:%03d-exp:%03d-imidx:%02d-bs:%02d.png' % (h, idx_exp, imidx, num_bs))
					plt.close()
	print('=======================================================', file=filename2)
