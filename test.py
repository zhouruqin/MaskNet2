#import open3d as o3d
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
from scipy.spatial.transform import Rotation
from sklearn.metrics import r2_score
import kornia.geometry.conversions as C

# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))
	
from learning3d.models import MaskNet
from learning3d.data_utils import RegistrationData, ModelNet40Data, UserData, AnyData
from registration import Registration
from learning3d.ops import se3
'''
def pc2open3d(data):
	if torch.is_tensor(data): data = data.detach().cpu().numpy()
	if len(data.shape) == 2:
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(data)
		return pc
	else:
		print("Error in the shape of data given to Open3D!, Shape is ", data.shape)

def display_results(template, source, est_T, masked_template):
	transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]

	template = pc2open3d(template)
	source = pc2open3d(source)
	transformed_source = pc2open3d(transformed_source)
	masked_template = pc2open3d(masked_template)
	
	template.paint_uniform_color([1, 0, 0])
	source.paint_uniform_color([0, 1, 0])
	transformed_source.paint_uniform_color([0, 0, 1])
	masked_template.paint_uniform_color([0, 0, 1])

	o3d.visualization.draw_geometries([template, source])
	o3d.visualization.draw_geometries([masked_template, source])
	o3d.visualization.draw_geometries([template, source, transformed_source])
'''
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

def dcm2euler(  mats: np.ndarray, seq: str = 'zyx', degrees: bool = True):
		"""Converts rotation matrix to euler angles

			Args:
				mats: (B, 3, 3) containing the B rotation matricecs
				seq: Sequence of euler rotations (default: 'zyx')
				degrees (bool): If true (default), will return in degrees instead of radians

			Returns:
		"""

		eulers = []
		for i in range(mats.shape[0]):
			r = Rotation.from_dcm(mats[i])
			eulers.append(r.as_euler(seq, degrees=degrees))
		return np.stack(eulers)

def valid_metric(rotations_ab, translations_ab, rotations_ab_pred, translations_ab_pred):
		rotations_ab = np.concatenate(rotations_ab, axis=0)
		translations_ab = np.concatenate(translations_ab, axis=0)
		rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
		translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
	   
		eulers_ab = dcm2euler(np.array(rotations_ab), seq='xyz')
		eulers_ab_pred = dcm2euler(np.array(rotations_ab_pred),seq='xyz')
		r_ab_mse = np.mean((eulers_ab - eulers_ab_pred) ** 2)
		r_ab_rmse = np.sqrt(r_ab_mse)
		r_ab_mae = np.mean(np.abs(eulers_ab - eulers_ab_pred))
		t_ab_mse = np.mean((translations_ab - translations_ab_pred) ** 2)
		t_ab_rmse = np.sqrt(t_ab_mse)
		t_ab_mae = np.mean(np.abs(translations_ab - translations_ab_pred))
		r_ab_r2_score = r2_score(eulers_ab, eulers_ab_pred)
		t_ab_r2_score = r2_score(translations_ab, translations_ab_pred)

		info = {
				'r_ab_mse': r_ab_mse,
				'r_ab_rmse': r_ab_rmse,
				'r_ab_mae': r_ab_mae,
				't_ab_mse': t_ab_mse,
				't_ab_rmse': t_ab_rmse,
				't_ab_mae': t_ab_mae,
				'r_ab_r2_score': r_ab_r2_score,
				't_ab_r2_score': t_ab_r2_score}
	   
		print(f'r_ab_mse= { r_ab_mse},'+ '\n' +
				f'r_ab_rmse= {r_ab_rmse},'+ '\n' +
				f'r_ab_mae={ r_ab_mae},'+ '\n' +
				f't_ab_mse= {t_ab_mse},'+ '\n' +
				f't_ab_rmse= {t_ab_rmse},'+ '\n' +
				f't_ab_mae= {t_ab_mae},'+ '\n' +
				f'r_ab_r2_score={ r_ab_r2_score},'+ '\n' +
				f't_ab_r2_score= {t_ab_r2_score}')
		
		return info

def dcm2euler(  mats: np.ndarray, seq: str = 'zyx', degrees: bool = True):
		"""Converts rotation matrix to euler angles

			Args:
				mats: (B, 3, 3) containing the B rotation matricecs
				seq: Sequence of euler rotations (default: 'zyx')
				degrees (bool): If true (default), will return in degrees instead of radians

			Returns:
		"""

		eulers = []
		for i in range(mats.shape[0]):
			r = Rotation.from_dcm(mats[i])
			eulers.append(r.as_euler(seq, degrees=degrees))
		return np.stack(eulers)

def compute_metrics( p1, p0,  gt_transforms_rotate, gt_transforms_trans
		, pred_transforms_rotate, pred_transforms_trans):
		"""Compute metrics required in the paper
		"""
		def square_distance(src, dst):
			return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

		with torch.no_grad():
		  
			# Euler angles, Individual translation errors (Deep Closest Point convention)
			# TODO Change rotation to torch operations

			r_gt_euler_deg = dcm2euler(np.array(gt_transforms_rotate), seq='xyz')
			r_pred_euler_deg = dcm2euler(np.array(pred_transforms_rotate), seq='xyz')
			
			t_gt = np.array(gt_transforms_trans)
			t_pred =np.array( pred_transforms_trans)
		
			r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2)
			r_rmse = np.sqrt(r_mse)
			r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg))
			t_mse = np.mean((t_gt - t_pred) ** 2)
			t_rmse = np.sqrt(t_mse)
			t_mae = np.mean(np.abs(t_gt - t_pred))
			r_ab_r2_score = r2_score(r_gt_euler_deg, r_pred_euler_deg)
			t_ab_r2_score = r2_score(t_gt, t_pred)

			# Rotation, translation errors (isotropic, i.e. doesn't depend on error
			# direction, which is more representative of the actual error)
			#concatenated = self.concatenate(self.inverse(gt_transforms), pred_transforms)
			#rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
			#residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
			#residual_transmag = concatenated[:, :, 3].norm(dim=-1)

			# Modified Chamfer distance
			#src_transformed = se3.transform(pred_transforms, points_src)
			#ref_clean = points_raw
			#src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)
			#dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
			#dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
			#chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

			metrics = {
				'r_mse': r_mse,
				'r_rmse': r_rmse,
				'r_mae': r_mae,
				'r_ab_r2_score': r_ab_r2_score,
				't_mse': t_mse,
				't_rmse': t_rmse,
				't_mae': t_mae,
				't_ab_r2_score':t_ab_r2_score
				#'err_r_deg': to_numpy(residual_rotdeg),
				#'err_t': to_numpy(residual_transmag),
			   # 'chamfer_dist': to_numpy(chamfer_dist)
			}

		return metrics

def test_one_epoch(args, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	AP_List = []
	GT_Size_List = []
	precision_list = []
	registration_model = Registration(args.reg_algorithm)

	rotations_ab = []
	translations_ab = []
	rotations_ab_pred = []
	translations_ab_pred = []

	predict_num_0= 0
	target_num_0= 0
	acc_num_0 = 0
	predict_num_1= 0
	target_num_1= 0
	acc_num_1 = 0


	for i, data in enumerate(tqdm(test_loader)):
		template, source, igt, gt_mask_1, gt_mask_0 = data

		template = template.to(args.device)
		source = source.to(args.device)
		igt = igt.to(args.device)			
		gt_mask_1 = gt_mask_1.to(args.device)
		gt_mask_0 = gt_mask_0.to(args.device)

		#----------------------------------------------------------------------------------------#
		masked_template, mask_source, predicted_mask_1, predicted_mask_0 = model(template, source)
		mask_0_binary = torch.where(predicted_mask_0 > 0.5, torch.ones(predicted_mask_0.size()).cuda(), torch.zeros(predicted_mask_0.size()).cuda())  
		mask_1_binary = torch.where(predicted_mask_1 > 0.5, torch.ones(predicted_mask_1.size()).cuda(), torch.zeros(predicted_mask_1.size()).cuda())

		predict_num_0 += mask_0_binary.sum(1)
		target_num_0 += gt_mask_0.sum(1)
		acc_mask_0 = mask_0_binary*gt_mask_0
		acc_num_0 += acc_mask_0.sum(1)

		predict_num_1 += mask_1_binary.sum(1)
		target_num_1 += gt_mask_1.sum(1)
		acc_mask_1 = mask_1_binary*gt_mask_1
		acc_num_1 += acc_mask_1.sum(1)
		
		#------------------------------------------------------------------------------------------#
		result = registration_model.register(template[:, torch.tensor(mask_1_binary, dtype = torch.bool).squeeze(0), 0:3]
		,source[:, torch.tensor(mask_0_binary, dtype = torch.bool).squeeze(0), 0:3] )  # rpmnet： template， source
		est_T = result['est_T']
		rotations_ab_pred.append(est_T[0,0:3,0:3].detach().cpu().numpy())
		translations_ab_pred.append(est_T[0,0:3,3].detach().cpu().numpy())

		rotations_ab.append(igt[0,0:3,0:3].detach().cpu().numpy())
		translations_ab.append(igt[0,0:3,3].detach().cpu().numpy())
	
		mask_x_binary = mask_0_binary.unsqueeze(2).repeat(1, 1, 3)  #B,N1, 3
		mask_y_binary = mask_1_binary.unsqueeze(2).repeat(1, 1, 3)  #B, N2, 3
		
		transformed_source = se3.transform(igt, source.permute(0,2,1))#B, 3, N1
		
		non_masked_template = template.clone().detach()
		non_masked_source = transformed_source.permute(0,2,1).clone().detach()
		non_masked_template[torch.tensor(gt_mask_1, dtype = torch.bool)] = 0.0
		non_masked_source[torch.tensor(gt_mask_0, dtype = torch.bool)] =0.0
		np.savetxt(str(i)+'_template.txt', np.column_stack((non_masked_template.cpu().numpy()[0,:, 0],non_masked_template.cpu().numpy()[0,:, 1],non_masked_template.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n' ) #保存为整数
		np.savetxt(str(i)+'_source.txt', np.column_stack((non_masked_source.cpu().numpy()[0,:, 0],non_masked_source.cpu().numpy()[0,:, 1],non_masked_source.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n' ) #保存为整数
		

		masked_template = template.clone().detach()
		masked_source = transformed_source.permute(0,2,1).clone().detach()
		masked_template[~torch.tensor(mask_y_binary, dtype = torch.bool)] = 0.0
		masked_source[~torch.tensor(mask_x_binary, dtype = torch.bool)] =0.0

		gt_masked_template = template.clone().detach()
		gt_masked_source = transformed_source.permute(0,2,1).clone().detach()
		gt_masked_template[~torch.tensor(gt_mask_1, dtype = torch.bool)] = 0.0
		gt_masked_source[~torch.tensor(gt_mask_0, dtype = torch.bool)] =0.0

		np.savetxt(str(i)+'_masked_template.txt', np.column_stack((masked_template.cpu().numpy()[0,:, 0],masked_template.cpu().numpy()[0,:, 1],masked_template.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n' ) #保存为整数
		np.savetxt(str(i)+'_masked_source.txt',np.column_stack((masked_source.cpu().numpy()[0,:, 0],masked_source.cpu().numpy()[0,:,1],masked_source.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n') #保存为整数
		np.savetxt(str(i)+'_gt_masked_template.txt', np.column_stack((gt_masked_template.cpu().numpy()[0,:, 0],gt_masked_template.cpu().numpy()[0,:, 1],gt_masked_template.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n' ) #保存为整数
		np.savetxt(str(i)+'_gt_masked_source.txt',np.column_stack((gt_masked_source.cpu().numpy()[0,:, 0],gt_masked_source.cpu().numpy()[0,:,1],gt_masked_source.cpu().numpy()[0,:, 2])),fmt='%f %f %f',newline='\n') #保存为整数
	

	recall_0 = acc_num_0/target_num_0
	precision_0 = acc_num_0/predict_num_0
	F1_0 = 2*recall_0*precision_0/(recall_0+precision_0)
	print('recall_0: %f,precision_0: %f, F1_0:%f'%(recall_0,precision_0, F1_0) )

	recall_1 = acc_num_1/target_num_1
	precision_1 = acc_num_1/predict_num_1
	F1_1 = 2*recall_1*precision_1/(recall_1+precision_1)
	print('recall_1: %f,precision_1: %f, F1_1:%f'%(recall_1,precision_1, F1_1) )
	
	metric =compute_metrics(source, template, rotations_ab,translations_ab, rotations_ab_pred,translations_ab_pred )
	print(f"r_mse = {metric['r_mse']}")
	print(f"r_rmse = {metric['r_rmse']}")
	print(f"r_mae = {metric['r_mae']}")
	print(f"t_mse = {metric['t_mse']}") 
	print(f"t_rmse = {metric['t_rmse']}")  
	print(f"t_mae = {metric['t_mae']}")
	print(f"r_ab_r2_score = {metric['r_ab_r2_score']}")
	print(f"t_ab_r2_score = {metric['t_ab_r2_score']}")

def test(args, model, test_loader):
	test_one_epoch(args, model, test_loader)


def options():
	parser = argparse.ArgumentParser(description='MaskNet: A Fully-Convolutional Network For Inlier Estimation (Testing)')
	parser.add_argument('--user_data', type=bool, default=False, help='Train or Evaluate the network with User Input Data.')
	parser.add_argument('--any_data', type=bool, default=False, help='Evaluate the network with Any Point Cloud.')

	# settings for input data
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')
	parser.add_argument('--partial', default=1, type=int,
						help='create partial source point cloud in dataset.')
	parser.add_argument('--noise', default=0, type=int,
						help='Add noise in source point clouds.')
	parser.add_argument('--outliers', default=0, type=int,
						help='Add outliers to template point cloud.')

	# settings for on testing
	parser.add_argument('-j', '--workers', default=1, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--test_batch_size', default=1, type=int,
						metavar='N', help='test-mini-batch size (default: 1)')
	parser.add_argument('--reg_algorithm', default='pointnetlk', type=str,
						help='Algorithm used for registration.', choices=['scanet', 'pointnetlk', 'icp', 'dcp', 'prnet', 'pcrnet', 'rpmnet'])
	parser.add_argument('--pretrained', default='./pretrained/model_masknet_ModelNet40.t7', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')
	parser.add_argument('--unseen', default=False, type=bool,
						help='Use first 20 categories for training and last 20 for testing')

	args = parser.parse_args()
	return args


import os
import pandas as pd
from plyfile import PlyData, PlyElement


def readplyfile(filename, num_sample):
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


def main():
	args = options()
	torch.backends.cudnn.deterministic = True

	if args.user_data:
		template = np.random.randn(1, 100, 3)					# Define/Read template point cloud. [Shape: BATCH x No. of Points x 3]
		source = np.random.randn(1, 75, 3)						# Define/Read source point cloud. [Shape: BATCH x No. of Points x 3]
		mask = np.zeros((1, 100, 1))							# Define/Read mask for point cloud. [Not mandatory in testing]
		igt = np.zeros((1, 4, 4))								# Define/Read igt transformation. [Not mandatory during testing]
		testset = UserData(template=template, source=source, mask=None, igt=None)	
	elif args.any_data:
		# Read Stanford bunny's point cloud.
		bunny_path = os.path.join('/home/simple/zrq/Bologna_dataset/room_scan1.ply')
		if not os.path.exists(bunny_path): 
			print("Please download bunny dataset from http://graphics.stanford.edu/data/3Dscanrep/")
			print("Add the extracted folder in learning3d/data/")
		data = readplyfile(bunny_path, args.num_points)
		points = normalize_pc(np.array(data))
	
		# points_idx = farthest_point_sample(points, 10000)
		points_idx = np.arange(points.shape[0])
		np.random.shuffle(points_idx)
		points = points[points_idx[:args.num_points], :]#int(points.shape[0]/num_points) *
			
		points = np.array(points)
		#print(points.shape)
		idx = np.arange(points.shape[0])
		np.random.shuffle(idx)
		points = points[idx[:]]

		testset = AnyData(pc=points, mask=True, partial=args.partial, noise=args.noise, outliers=args.outliers)
	else:
		testset = RegistrationData(ModelNet40Data(train=False, num_points=args.num_points, unseen=args.unseen),
									partial=args.partial, noise=args.noise, outliers=args.outliers)
	test_loader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False, num_workers=args.workers)

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
