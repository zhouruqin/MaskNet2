#import open3d as o3d
import argparse
import os
import sys
import copy
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))
	
from learning3d.models import MaskNet
from learning3d.data_utils import RegistrationData, ModelNet40Data, UserData, AnyData
from registration import Registration

def farthest_point_sample(xyz, npoint):
	"""
	Input:
		xyz: pointcloud data, [B, N, C]
		npoint: number of samples
	Return:
		centroids: sampled pointcloud index, [B, npoint]
	"""
	#import ipdb; ipdb.set_trace()
	if not torch.is_tensor(xyz): xyz = torch.tensor(xyz).float().view(1, -1, 3)
	device = xyz.device
	B, N, C = xyz.shape
	centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
	distance = torch.ones(B, N).to(device) * 1e10
	farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
	batch_indices = torch.arange(B, dtype=torch.long).to(device)
	for i in range(npoint):
		centroids[:, i] = farthest
		centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
		dist = torch.sum((xyz - centroid) ** 2, -1)
		mask = dist < distance
		distance[mask] = dist[mask]
		farthest = torch.max(distance, -1)[1]
	return centroids.cpu().numpy()[0]



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
	pc = readplyfile(path)
	points = normalize_pc(np.array(pc))
	
	if sample_pc:
		# points_idx = farthest_point_sample(points, 10000)
		points_idx = np.arange(points.shape[0])
		np.random.shuffle(points_idx)
		points = points[points_idx[:10000], :]
	return points

def pc2open3d(data):
	if torch.is_tensor(data): data = data.detach().cpu().numpy()
	if len(data.shape) == 2:
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(data)
		return pc
	else:
		print("Error in the shape of data given to Open3D!, Shape is ", data.shape)

# This function is taken from Deep Global Registration paper's github repository.
def create_pcd(xyz, color):
	# n x 3
	n = xyz.shape[0]
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(xyz)
	pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	return pcd

# This function is taken from Deep Global Registration paper's github repository.
def draw_geometries_flip(pcds):
	pcds_transform = []
	flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
	for pcd in pcds:
		pcd_temp = copy.deepcopy(pcd)
		pcd_temp.transform(flip_transform)
		pcds_transform.append(pcd_temp)
	o3d.visualization.draw_geometries(pcds_transform)

def display_results(template, source, est_T, mask_idx):
	non_mask_idx = np.array([i for i in range(mask_idx.shape[0]) if i not in mask_idx])
	unmasked_template = template[non_mask_idx]
	masked_template = template[mask_idx]

	transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]
	
	template = create_pcd(template, np.array([1, 0.706, 0]))
	source = create_pcd(source, np.array([0, 0.929, 0.651]))
	transformed_source = create_pcd(transformed_source, np.array([0, 0.651, 0.921]))
	masked_template = create_pcd(masked_template, np.array([0,0,1]))
	unmasked_template = create_pcd(unmasked_template, np.array([1,0,0]))

	draw_geometries_flip([template, source])
	draw_geometries_flip([template, transformed_source])
	draw_geometries_flip([masked_template, unmasked_template])

def store_results(args, template, source, est_T, igt, gt_mask, predicted_mask, mask_idx, est_T_series):
	est_T_series = est_T_series.detach().cpu().numpy().reshape(-1, 4, 4)
	est_T_series = est_T_series.reshape(-1, 4)
	mesh = read_mesh(path=args.dataset_path, sample_pc=False)

	np.savez('3dmatch_results', template=template.detach().cpu().numpy()[0], 
			  source = source.detach().cpu().numpy()[0], est_T = est_T.detach().cpu().numpy()[0], 
			  igt = igt.detach().cpu().numpy()[0], gt_mask=gt_mask.detach().cpu().numpy()[0], 
			  mask_idx=mask_idx.detach().cpu().numpy()[0], est_T_series=est_T_series,
			  predicted_mask=predicted_mask.detach().cpu().numpy()[0], mesh=mesh)

def evaluate_metrics(TP, FP, FN, TN, gt_mask):
	# TP, FP, FN, TN: 		True +ve, False +ve, False -ve, True -ve
	# gt_mask:				Ground Truth mask [Nt, 1]
	
	accuracy = (TP + TN)/gt_mask.shape[1]
	misclassification_rate = (FN + FP)/gt_mask.shape[1]
	# Precision: (What portion of positive identifications are actually correct?)
	precision = TP / (TP + FP)
	# Recall: (What portion of actual positives are identified correctly?)
	recall = TP / (TP + FN)

	fscore = (2*precision*recall) / (precision + recall)
	return accuracy, precision, recall, fscore

# Function used to evaluate the predicted mask with ground truth mask.
def evaluate_mask(gt_mask, predicted_mask, predicted_mask_idx):
	# gt_mask:					Ground Truth Mask [Nt, 1]
	# predicted_mask:			Mask predicted by network [Nt, 1]
	# predicted_mask_idx:		Point indices chosen by network [Ns, 1]

	if torch.is_tensor(gt_mask): gt_mask = gt_mask.detach().cpu().numpy()
	if torch.is_tensor(gt_mask): predicted_mask = predicted_mask.detach().cpu().numpy()
	if torch.is_tensor(predicted_mask_idx): predicted_mask_idx = predicted_mask_idx.detach().cpu().numpy()
	gt_mask, predicted_mask, predicted_mask_idx = gt_mask.reshape(1,-1), predicted_mask.reshape(1,-1), predicted_mask_idx.reshape(1,-1)
	
	gt_idx = np.where(gt_mask == 1)[1].reshape(1,-1) 				# Find indices of points which are actually in source.

	# TP + FP = number of source points.
	TP = np.intersect1d(predicted_mask_idx[0], gt_idx[0]).shape[0]			# is inliner and predicted as inlier (True Positive) 		(Find common indices in predicted_mask_idx, gt_idx)
	FP = len([x for x in predicted_mask_idx[0] if x not in gt_idx])			# isn't inlier but predicted as inlier (False Positive)
	FN = FP															# is inlier but predicted as outlier (False Negative) (due to binary classification)
	TN = gt_mask.shape[1] - gt_idx.shape[1] - FN 					# is outlier and predicted as outlier (True Negative)
	return evaluate_metrics(TP, FP, FN, TN, gt_mask)

def test_one_epoch(args, model, test_loader):
	model.eval()
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

		if args.loss_fn == 'mse':
			loss_mask_y = torch.nn.functional.mse_loss(predicted_mask_y, gt_mask_y)
			loss_mask_x = torch.nn.functional.mse_loss(predicted_mask_x, gt_mask_x)
		elif args.loss_fn == 'bce':
			loss_mask_y = torch.nn.BCELoss()(predicted_mask_y, gt_mask_y)
			loss_mask_x = torch.nn.BCELoss()(predicted_mask_x, gt_mask_x)
		
		loss_mask =  loss_mask_y +loss_mask_x #
		
		mask_x_binary = torch.where(predicted_mask_x > 0.5, torch.ones(predicted_mask_x.size()).cuda(), torch.zeros(predicted_mask_x.size()).cuda())  
		mask_y_binary = torch.where(predicted_mask_y > 0.5, torch.ones(predicted_mask_y.size()).cuda(), torch.zeros(predicted_mask_y.size()).cuda())
		
		
		#percent_x = torch.mean((mask_x_binary.size()[1] - torch.sum(torch.abs(mask_x_binary - gt_mask_x), dim =1))/mask_x_binary.size()[1])
		#percent_y =  torch.mean((mask_y_binary.size()[1] - torch.sum(torch.abs(mask_y_binary - gt_mask_y), dim =1))/mask_y_binary.size()[1])
		predict_num_x += mask_x_binary.sum(1)
		target_num_x += gt_mask_x.sum(1)
		acc_mask_x = mask_x_binary*gt_mask_x
		acc_num_x += acc_mask_x.sum(1)

		predict_num_y += mask_y_binary.sum(1)
		target_num_y += gt_mask_y.sum(1)
		acc_mask_y = mask_y_binary*gt_mask_y
		acc_num_y += acc_mask_y.sum(1)

		#percent_x_mean += percent_x
		#percent_y_mean += percent_y
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


	#percent_x_mean = float(percent_x_mean)/count
	#percent_y_mean = float(percent_y_mean)/count
	recall_x = acc_num_x/target_num_x
	precision_x = acc_num_x/predict_num_x
	F1_x = 2*recall_x*precision_x/(recall_x+precision_x)
	#accuracy_x = acc_num_x.sum(1)/target_num_x.sum(1)
	print('recall_x: %f,precision_x: %f, F1_x:%f'%(recall_x,precision_x, F1_x) )

	recall_y = acc_num_y/target_num_y
	precision_y = acc_num_y/predict_num_y
	F1_y = 2*recall_y*precision_y/(recall_y+precision_y)
	#accuracy_y = acc_num_y.sum(1)/target_num_y.sum(1)
	print('recall_y: %f,precision_y: %f, F1_y:%f'%(recall_y,precision_y, F1_y) )

	test_loss = float(test_loss)/count
	test_loss_y = float(test_loss_y)/count
	test_loss_x = float(test_loss_x)/count
	return test_loss, test_loss_y, test_loss_x, precision_y, precision_x

def test(args, model, test_loader):
	test_one_epoch(args, model, test_loader)

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
		igt = igt.to(args.device)					# [source] = [igt]*[template]
		gt_mask_y = gt_mask_y.to(args.device)
		gt_mask_x = gt_mask_x.to(args.device)

		masked_template, masked_source, predicted_mask_y, predicted_mask_x = model(template, source)

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
		
		mask_x_binary = mask_x_binary.unsqueeze(2).repeat(1, 1, 3)  #B,N1, 3
		mask_y_binary = mask_y_binary.unsqueeze(2).repeat(1, 1, 3)  #B, N2, 3

		template[~torch.tensor(mask_y_binary, dtype = torch.bool)] = 0.0
		source[~torch.tensor(mask_x_binary, dtype = torch.bool)] =0.0

		loss_overlap = torch.nn.BCELoss()
		loss_mask =loss_mask_y +loss_mask_x  #
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
		igt = igt.to(args.device)				# [source] = [igt]*[template]
		gt_mask_y = gt_mask_y.to(args.device)
		gt_mask_x = gt_mask_x.to(args.device)

		masked_template,masked_source, predicted_mask_y, predicted_mask_x= model(template, source)

		if args.loss_fn == 'mse':
			loss_mask_y = torch.nn.functional.mse_loss(predicted_mask_y, gt_mask_y)
			loss_mask_x = torch.nn.functional.mse_loss(predicted_mask_x, gt_mask_x)
		elif args.loss_fn == 'bce':
			loss_mask_y = torch.nn.BCELoss()(predicted_mask_y, gt_mask_y)
			loss_mask_x = torch.nn.BCELoss()(predicted_mask_x, gt_mask_x)
		
		loss_mask =  loss_mask_y +loss_mask_x #
		
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


	percent_x_mean = float(percent_x_mean)/count
	percent_y_mean = float(percent_y_mean)/count

	test_loss = float(test_loss)/count
	test_loss_y = float(test_loss_y)/count
	test_loss_x = float(test_loss_x)/count
	return test_loss, test_loss_y, test_loss_x, percent_y_mean, percent_x_mean

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
	parser = argparse.ArgumentParser(description='MaskNet: A Fully-Convolutional Network For Inlier Estimation (3DMatch Testing)')
	
	# settings for input data
	parser.add_argument('--dataset_path', default='/home/simple/zrq/masknet/3dmatch/sun3d-home_at-home_at_scan1_2013_jan_1/train/cloud_bin_29.ply', type=str,
						help='Provide the path to .ply file in 3DMatch dataset.')
	parser.add_argument('--num_points', default=10000, type=int, help='Number of points in sampled point cloud')
	parser.add_argument('--partial_source', default=True, type=bool,
						help='create partial source point cloud in dataset.')
	parser.add_argument('--noise', default=False, type=bool,
						help='Add noise in source point clouds.')
	parser.add_argument('--outliers', default=False, type=bool,
						help='Add outliers to template point cloud.')

	# settings for on testing
	parser.add_argument('-j', '--workers', default=1, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--test_batch_size', default=1, type=int,
						metavar='N', help='test-mini-batch size (default: 1)')
	parser.add_argument('--reg_algorithm', default='pointnetlk', type=str,
						help='Algorithm used for registration.', choices=['pointnetlk', 'icp', 'dcp', 'prnet', 'pcrnet', 'rpmnet'])
	parser.add_argument('--pretrained', default='../pretrained/model_masknet_3DMatch.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')
	parser.add_argument('--store_results', default=True, type=bool,
						help='Store results of 3DMatch Test')

	args = parser.parse_args()
	return args

def main():
	args = options()
	torch.backends.cudnn.deterministic = True

	points = read_mesh(path=args.dataset_path, sample_pc=True, num_points=args.num_points)
	testset = AnyData(pc=points, mask=True, repeat=1)
	test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Load Pretrained MaskNet.
	model = MaskNet()
	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	model = model.to(args.device)
	
	test(args, model, test_loader)

if __name__ == '__main__':
	main()