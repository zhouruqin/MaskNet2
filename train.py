import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from learning3d.ops import se3

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))
	
from learning3d.models import MaskNet
from learning3d.data_utils import RegistrationData, ModelNet40Data, AnyData

def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.exp_name):
		os.makedirs('checkpoints/' + args.exp_name)
	if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
	os.system('cp train.py checkpoints' + '/' + args.exp_name + '/' + 'train.py.backup')
	os.system('cp learning3d/models/masknet.py checkpoints' + '/' + args.exp_name + '/' + 'masknet.py.backup')
	os.system('cp learning3d/data_utils/dataloaders.py checkpoints' + '/' + args.exp_name + '/' + 'dataloaders.py.backup')


class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()

def eval_one_epoch(args, model, test_loader):
	model.eval()
	test_loss = 0.0
	test_loss_y = 0.0
	test_loss_x = 0.0
	percent_x_mean = 0.0
	percent_y_mean = 0.0
	pred  = 0.0
	count = 0

	predict_num_x= 0
	target_num_x= 0
	acc_num_x = 0
	predict_num_y= 0
	target_num_y= 0
	acc_num_y = 0

	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt, gt_mask_y, gt_mask_x = data

		template = template.to(args.device)
		source = source.to(args.device)
		gt_mask_y = gt_mask_y.to(args.device)
		gt_mask_x = gt_mask_x.to(args.device)

		masked_template,masked_source, predicted_mask_y, predicted_mask_x= model(template, source)

		if args.loss_fn == 'mse':
			loss_mask_y = torch.nn.functional.mse_loss(predicted_mask_y, gt_mask_y)
			loss_mask_x = torch.nn.functional.mse_loss(predicted_mask_x, gt_mask_x)
			
		elif args.loss_fn == 'bce':
			loss_mask_y = torch.nn.BCELoss()(predicted_mask_y, gt_mask_y)
			loss_mask_x = torch.nn.BCELoss()(predicted_mask_x, gt_mask_x)
		
		
		loss_mask =  loss_mask_y +loss_mask_x#
		
		mask_x_binary = torch.where(predicted_mask_x > 0.5, torch.ones(predicted_mask_x.size()).cuda(), torch.zeros(predicted_mask_x.size()).cuda())  
		mask_y_binary = torch.where(predicted_mask_y > 0.5, torch.ones(predicted_mask_y.size()).cuda(), torch.zeros(predicted_mask_y.size()).cuda())
		
		
		percent_x = torch.mean((mask_x_binary.size()[1] - torch.sum(torch.abs(mask_x_binary - gt_mask_x), dim =1))/mask_x_binary.size()[1])
		percent_y =  torch.mean((mask_y_binary.size()[1] - torch.sum(torch.abs(mask_y_binary - gt_mask_y), dim =1))/mask_y_binary.size()[1])

		percent_x_mean += percent_x
		percent_y_mean += percent_y
		test_loss += loss_mask.item()
		test_loss_y += loss_mask_y.item()
		test_loss_x += loss_mask_x.item()
		count += 1
		
	percent_x_mean = float(percent_x_mean)/count
	percent_y_mean = float(percent_y_mean)/count

	test_loss = float(test_loss)/count
	test_loss_y = float(test_loss_y)/count
	test_loss_x = float(test_loss_x)/count
	return test_loss, test_loss_y, test_loss_x, percent_y_mean, percent_x_mean

def test_one_epoch(args, model, test_loader):
	model.eval()
	test_loss = 0.0
	test_loss_y = 0.0
	test_loss_x = 0.0
	percent_x_mean = 0.0
	percent_y_mean = 0.0
	pred  = 0.0
	count = 0

	predict_num_x= 0
	target_num_x= 0
	acc_num_x = 0
	predict_num_y= 0
	target_num_y= 0
	acc_num_y = 0

	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt, gt_mask_y, gt_mask_x = data

		template = template.to(args.device)
		source = source.to(args.device)
		igt = igt.to(args.device)				# [source] = [igt]*[template]
		gt_mask_y = gt_mask_y.to(args.device)
		gt_mask_x = gt_mask_x.to(args.device)

		masked_template,masked_source, predicted_mask_y, predicted_mask_x= model(template, source)
		
		mask_x_binary = torch.where(predicted_mask_x > 0.5, torch.ones(predicted_mask_x.size()).cuda(), torch.zeros(predicted_mask_x.size()).cuda())  
		mask_y_binary = torch.where(predicted_mask_y > 0.5, torch.ones(predicted_mask_y.size()).cuda(), torch.zeros(predicted_mask_y.size()).cuda())
		
		if args.loss_fn == 'mse':
			loss_mask_y = torch.nn.functional.mse_loss(predicted_mask_y, gt_mask_y)
			loss_mask_x = torch.nn.functional.mse_loss(predicted_mask_x, gt_mask_x)
		elif args.loss_fn == 'bce':
			loss_mask_y = torch.nn.BCELoss()(predicted_mask_y, gt_mask_y)
			loss_mask_x = torch.nn.BCELoss()(predicted_mask_x, gt_mask_x)
		loss_mask =  loss_mask_y +loss_mask_x #	
		
		predict_num_x += mask_x_binary.sum(1)
		target_num_x += gt_mask_x.sum(1)
		acc_mask_x = mask_x_binary*gt_mask_x
		acc_num_x += acc_mask_x.sum(1)

		predict_num_y += mask_y_binary.sum(1)
		target_num_y += gt_mask_y.sum(1)
		acc_mask_y = mask_y_binary*gt_mask_y
		acc_num_y += acc_mask_y.sum(1)

		test_loss += loss_mask.item()
		test_loss_y += loss_mask_y.item()
		test_loss_x += loss_mask_x.item()
		count += 1

		mask_x_binary = mask_x_binary.unsqueeze(2).repeat(1, 1, 3)  #B,N1, 3
		mask_y_binary = mask_y_binary.unsqueeze(2).repeat(1, 1, 3)  #B, N2, 3
		
		transformed_source = se3.transform(igt, source.permute(0,2,1))#B, 3, N1
		
		non_masked_template = template.clone().detach()
		non_masked_source = transformed_source.permute(0,2,1).clone().detach()
		non_masked_template[torch.tensor(gt_mask_y, dtype = torch.bool)] = 0.0
		non_masked_source[torch.tensor(gt_mask_x, dtype = torch.bool)] =0.0
		np.savetxt(str(i)+'_template.txt', np.column_stack((non_masked_template.cpu().numpy()[0,:, 0],non_masked_template.cpu().numpy()[0,:, 1],non_masked_template.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n' ) #保存为整数
		np.savetxt(str(i)+'_source.txt', np.column_stack((non_masked_source.cpu().numpy()[0,:, 0],non_masked_source.cpu().numpy()[0,:, 1],non_masked_source.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n' ) #保存为整数
		

		masked_template = template.clone().detach()
		masked_source = transformed_source.permute(0,2,1).clone().detach()
		masked_template[~torch.tensor(mask_y_binary, dtype = torch.bool)] = 0.0
		masked_source[~torch.tensor(mask_x_binary, dtype = torch.bool)] =0.0

		gt_masked_template = template.clone().detach()
		gt_masked_source = transformed_source.permute(0,2,1).clone().detach()
		gt_masked_template[~torch.tensor(gt_mask_y, dtype = torch.bool)] = 0.0
		gt_masked_source[~torch.tensor(gt_mask_x, dtype = torch.bool)] =0.0

		
		np.savetxt(str(i)+'_masked_template.txt', np.column_stack((masked_template.cpu().numpy()[0,:, 0],masked_template.cpu().numpy()[0,:, 1],masked_template.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n' ) #保存为整数
		np.savetxt(str(i)+'_masked_source.txt',np.column_stack((masked_source.cpu().numpy()[0,:, 0],masked_source.cpu().numpy()[0,:,1],masked_source.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n') #保存为整数
		np.savetxt(str(i)+'_gt_masked_template.txt', np.column_stack((gt_masked_template.cpu().numpy()[0,:, 0],gt_masked_template.cpu().numpy()[0,:, 1],gt_masked_template.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n' ) #保存为整数
		np.savetxt(str(i)+'_gt_masked_source.txt',np.column_stack((gt_masked_source.cpu().numpy()[0,:, 0],gt_masked_source.cpu().numpy()[0,:,1],gt_masked_source.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n') #保存为整数


	recall_x = acc_num_x/target_num_x
	precision_x = acc_num_x/predict_num_x
	F1_x = 2*recall_x*precision_x/(recall_x+precision_x)
	print('recall_x: %f,precision_x: %f, F1_x:%f'%(recall_x,precision_x, F1_x) )

	recall_y = acc_num_y/target_num_y
	precision_y = acc_num_y/predict_num_y
	F1_y = 2*recall_y*precision_y/(recall_y+precision_y)
	print('recall_y: %f,precision_y: %f, F1_y:%f'%(recall_y,precision_y, F1_y) )

	test_loss = float(test_loss)/count
	test_loss_y = float(test_loss_y)/count
	test_loss_x = float(test_loss_x)/count
	return test_loss, test_loss_y, test_loss_x, precision_y, precision_x

def test(args, model, test_loader, textio):
	test_loss, test_loss_y, test_loss_x, percent_y_mean, percent_x_mean = test_one_epoch(args, model,  test_loader)
	textio.cprint('Test Loss: %f, Test Loss y: %f, Test Loss x: %f,Test y: %f,Test x: %f'%(test_loss, test_loss_y, test_loss_x, percent_y_mean, percent_x_mean))


def train_one_epoch(args, model, train_loader, optimizer):
	model.train()
	train_loss = 0.0
	train_loss_y = 0.0
	train_loss_x = 0.0
	percent_x_mean = 0.0
	percent_y_mean = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(train_loader)):
		template, source, igt, gt_mask_y, gt_mask_x = data

		template = template.to(args.device)
		source = source.to(args.device)
		gt_mask_y = gt_mask_y.to(args.device)
		gt_mask_x = gt_mask_x.to(args.device)

		masked_template, masked_source, predicted_mask_y, predicted_mask_x= model(template, source)

		if args.loss_fn == 'mse':
			loss_mask_y = torch.nn.functional.mse_loss(predicted_mask_y, gt_mask_y)
			loss_mask_x = torch.nn.functional.mse_loss(predicted_mask_x, gt_mask_x)
		
		elif args.loss_fn == 'bce':
			loss_mask_y = torch.nn.BCELoss()(predicted_mask_y, gt_mask_y)
			loss_mask_x = torch.nn.BCELoss()(predicted_mask_x, gt_mask_x)
		

		mask_x_binary = torch.where(predicted_mask_x > 0.5, torch.ones(predicted_mask_x.size()).cuda(), torch.zeros(predicted_mask_x.size()).cuda())  
		mask_y_binary = torch.where(predicted_mask_y > 0.5, torch.ones(predicted_mask_y.size()).cuda(), torch.zeros(predicted_mask_y.size()).cuda())
		
		percent_x = torch.mean((mask_x_binary.size()[1] - torch.sum(torch.abs(mask_x_binary - gt_mask_x), dim =1))/mask_x_binary.size()[1])
		percent_y =  torch.mean((mask_y_binary.size()[1] - torch.sum(torch.abs(mask_y_binary - gt_mask_y), dim =1))/mask_y_binary.size()[1])

		loss_mask =loss_mask_y +loss_mask_x #
		# forward + backward + optimize
		optimizer.zero_grad()
		loss_mask.backward()
		optimizer.step()

		percent_x_mean += percent_x
		percent_y_mean += percent_y
		train_loss += loss_mask.item()
		train_loss_y += loss_mask_y.item()
		train_loss_x += loss_mask_x.item()
		count += 1

	percent_x_mean = float(percent_x_mean)/count
	percent_y_mean = float(percent_y_mean)/count
	train_loss = float(train_loss)/count
	train_loss_y = float(train_loss_y)/count
	train_loss_x = float(train_loss_x)/count

	return train_loss, train_loss_y, train_loss_x, percent_y_mean, percent_x_mean

def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
	learnable_params = filter(lambda p: p.requires_grad, model.parameters())
	if args.optimizer == 'Adam':
		optimizer = torch.optim.Adam(learnable_params, lr=0.001)#0.001
	else:
		optimizer = torch.optim.SGD(learnable_params, lr=0.1)
	
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.000001)


	if checkpoint is not None:
		min_loss = checkpoint['min_loss']
		optimizer.load_state_dict(checkpoint['optimizer'])

	best_test_loss = np.inf

	for epoch in range(args.start_epoch, args.epochs):
		train_loss, train_loss_y, train_loss_x, train_percent_y, train_percent_x = train_one_epoch(args, model, train_loader, optimizer)
		test_loss, test_loss_y, test_loss_x, test_percent_y, test_percent_x = eval_one_epoch(args, model, test_loader)

		scheduler.step()

		if test_loss<best_test_loss:
			best_test_loss = test_loss
			
			snap = {'epoch': epoch + 1,
					'model': model.state_dict(),
					'min_loss': best_test_loss,
					'optimizer' : optimizer.state_dict(),}
			torch.save(snap, 'checkpoints/%s/models/best_model_snap.t7' % (args.exp_name))
			torch.save(model.state_dict(), 'checkpoints/%s/models/best_model.t7' % (args.exp_name))

		snap = {'epoch': epoch + 1,
				'model': model.state_dict(),
				'min_loss': best_test_loss,
				'optimizer' : optimizer.state_dict(),}
		torch.save(snap, 'checkpoints/%s/models/model_snap.t7' % (args.exp_name))
		torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % (args.exp_name))
		
		boardio.add_scalar('Train_Loss', train_loss, epoch+1)
		boardio.add_scalar('Test_Loss', test_loss, epoch+1)
		boardio.add_scalar('Best_Test_Loss', best_test_loss, epoch+1)

		textio.cprint('EPOCH:: %d, Train Loss: %f, Train Loss y: %f,Train Loss x: %f,Train y: %f,Train x: %f'%(epoch+1, train_loss, train_loss_y, train_loss_x, train_percent_y, train_percent_x))
		textio.cprint('Test Loss: %f, Test Loss y: %f, Test Loss x: %f,Test y: %f,Test x: %f, Best Loss: %f'%(test_loss, test_loss_y, test_loss_x, test_percent_y, test_percent_x, best_test_loss))

def options():
	parser = argparse.ArgumentParser(description='MaskNet: A Fully-Convolutional Network For Inlier Estimation (Training)')
	parser.add_argument('--exp_name', type=str, default='exp_masknet', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')
	
	# settings for input data
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')
	parser.add_argument('--partial', default= 1, type= int,
						help='Add partial to template point cloud.')
	parser.add_argument('--noise', default=0, type=int,
						help='Add noise in source point clouds.')
	parser.add_argument('--outliers', default=0 , type=int,
						help='Add outliers to template point cloud.')

	# settings for on training
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=32, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--test_batch_size', default=8, type=int,
						metavar='N', help='test-mini-batch size (default: 8)')
	parser.add_argument('--unseen', default=False, type=bool,
						help='Use first 20 categories for training and last 20 for testing')
	parser.add_argument('--epochs', default=500, type=int,
						metavar='N', help='number of total epochs to run')
	parser.add_argument('--start_epoch', default=0, type=int,
						metavar='N', help='manual epoch number (useful on restarts)')
	parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
						metavar='METHOD', help='name of an optimizer (default: Adam)')
	parser.add_argument('--resume', default='', type=str,
						metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
	parser.add_argument('--pretrained', default='', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')
	parser.add_argument('--loss_fn', default='mse', type=str, choices=['mse', 'bce'])
	parser.add_argument('--user_data', type=bool, default=False, help='Train or Evaluate the network with User Input Data.')
	parser.add_argument('--any_data', type=bool, default=False, help='Evaluate the network with Any Point Cloud.')
	parser.add_argument('--dataset_path_train', default='learning3d/match/train', type=str,
						help='Provide the path to .ply file in 3DMatch dataset.')
	parser.add_argument('--dataset_path_test', default='learning3d/match/test', type=str,
						help='Provide the path to .ply file in 3DMatch dataset.')
	
    
	args = parser.parse_args()
	return args


import os
import pandas as pd
from plyfile import PlyData, PlyElement


def readplyfile(filename):
    file_dir = filename  #文件的路径
    plydata = PlyData.read(file_dir)  # 读取文件
    data = plydata.elements[0].data  # 读取数据
    data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
    data_np = np.zeros(data_pd.shape, dtype=np.double)  # 初始化储存数据的array
    property_names = data[0].dtype.names  # 读取property的名字
    for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
        data_np[:, i] = data_pd[name]

    return data_np[:,0:3]

def normalize_pc(point_cloud):
	centroid = np.mean(point_cloud, axis=0)
	point_cloud -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(point_cloud)**2,axis=-1)))
	point_cloud /= furthest_distance
	return point_cloud

def read_mesh(path, sample_pc=True, num_points=10000):
	all_data = []
	files= os.listdir(path) 

	for file in files: #
		if not os.path.isdir(file): #
			pc = readplyfile(path+"/"+file)
			points = normalize_pc(np.array(pc))
	
			if sample_pc:
			# points_idx = farthest_point_sample(points, 10000)
				points_idx = np.arange(points.shape[0])
				np.random.shuffle(points_idx)
				points = points[points_idx[:num_points], :]#int(points.shape[0]/num_points) *
			
			all_data.append(points)
	all_data = np.concatenate(all_data, axis=0)
	all_data = all_data.reshape( -1 ,num_points, 3 )
	return all_data

def main():
	args = options()

	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
	_init_(args)

	textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
	textio.cprint(str(args))
	if args.eval == False:
		if args.any_data:
			points_train = read_mesh(path=args.dataset_path_train, sample_pc=True, num_points=args.num_points)
			trainset = AnyData(pc=points_train, mask=True)
			train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

			points_test = read_mesh(path=args.dataset_path_test, sample_pc=True, num_points=args.num_points)
			testset = AnyData(pc=points_test, mask=True)
			test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

		else:
			trainset = RegistrationData(ModelNet40Data(train=True, num_points=args.num_points, unseen=args.unseen),
								partial=args.partial, noise=args.noise, outliers=args.outliers)
			testset = RegistrationData(ModelNet40Data(train=False, num_points=args.num_points, unseen=args.unseen),
								partial =args.partial, noise=args.noise, outliers=args.outliers)
			train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
			test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	elif args.eval == True:
		if args.any_data:
			points_test = read_mesh(path=args.dataset_path_test, sample_pc=True, num_points=args.num_points)
			testset = AnyData(pc=points_test, mask=True)
			test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

		else:
			testset = RegistrationData(ModelNet40Data(train=False, num_points=args.num_points, unseen=args.unseen),
								partial=args.partial, noise=args.noise, outliers=args.outliers)
			test_loader = DataLoader(testset, batch_size= 1, shuffle=False, drop_last=False, num_workers=args.workers)


	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	model = MaskNet()
	model = model.to(args.device)

	checkpoint = None
	if args.resume:
		assert os.path.isfile(args.resume)
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model'])

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	model.to(args.device)

	if args.eval:
		test(args, model, test_loader, textio)
	else:
		train(args, model, train_loader, test_loader, boardio, textio, checkpoint)

if __name__ == '__main__':
	main()