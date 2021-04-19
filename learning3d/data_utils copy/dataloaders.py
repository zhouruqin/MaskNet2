import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import h5py
import subprocess
import shlex
import json
import glob
from .. ops import transform_functions, se3
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import transforms3d.quaternions as t3d
import h5py

def download_modelnet40():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		www += ' --no-check-certificate'
		os.system('wget %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))

def load_data(train, use_normals):
	if train: partition = 'train'
	else: partition = 'test'
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	all_data = []
	all_label = []
	for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
		f = h5py.File(h5_name, 'r')
		if use_normals: data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32')
		else: data = f['data'][:].astype('float32')
		label = f['label'][:].astype('int64')
		f.close()
		all_data.append(data)
		all_label.append(label)
	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
	# N, C = pointcloud.shape
	#sigma = 0.04*np.random.random_sample()
	pointcloud += torch.empty(pointcloud.shape).normal_(mean=0, std=sigma).clamp(-clip, clip)
	# pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
	return pointcloud

# Create Partial Point Cloud. [Code referred from PRNet paper.]
def farthest_subsample_points(pointcloud1, num_subsampled_points=768):
	pointcloud1 = pointcloud1
	num_points = pointcloud1.shape[0]
	nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
							 metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
	#nbrs2 = NearestNeighbors(n_neighbors=1024, algorithm='auto',
	#						 metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
	random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
	idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
	#idx2 = nbrs2.kneighbors(random_p1, return_distance=False).reshape((1024,))

	#intersect_mask, intersect_x, intersect_y  = np.intersect1d(idx1, idx2, return_indices=True)
	#print(intersect_mask)
	#print( idx1[intersect_x])
	#print(intersect_y) 

	#print(idx2)


	#print(intersect_mask)
	#print(idx2)
	#print(intersect_x)
	#print(intersect_y)
	#gt_mask_source = torch.zeros(pointcloud1.shape[0])	#tuple(p_source.shape[0])  
	#gt_mask_source [intersect_y]  = 1

	#gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)
	#print('gt_mask_source')
	#print(gt_mask_source)
	#print('gt_mask')
	#print(gt_mask)


	return pointcloud1[idx1, :], idx1

def add_outliers(pointcloud, gt_mask, count =100):
	# pointcloud: 			Point Cloud (ndarray) [NxC]
	# output: 				Corrupted Point Cloud (ndarray) [(N+300)xC]
	N, C = pointcloud.shape
	outliers = 2*torch.rand(count, C)-1 					# Sample points in a cube [-0.5, 0.5]
	pointcloud = torch.cat([pointcloud, outliers], dim=0)
	gt_mask = torch.cat([gt_mask, torch.zeros(count)])

	idx = torch.randperm(pointcloud.shape[0])
	pointcloud, gt_mask = pointcloud[idx], gt_mask[idx]

	return pointcloud, gt_mask

class UnknownDataTypeError(Exception):
	def __init__(self, *args):
		if args: self.message = args[0]
		else: self.message = 'Datatype not understood for dataset.'

	def __str__(self):
		return self.message


class ModelNet40Data(Dataset):
	def __init__(
		self,
		train=True,
		num_points=1024,
		download=True,
		randomize_data=False,
		unseen=False,
		use_normals=False
	):
		super(ModelNet40Data, self).__init__()
		if download: download_modelnet40()
		self.data, self.labels = load_data(train, use_normals)
		if not train: self.shapes = self.read_classes_ModelNet40()
		self.num_points = num_points
		self.randomize_data = randomize_data
		self.unseen = unseen
		if self.unseen:
			self.labels = self.labels.reshape(-1) 				# [N, 1] -> [N,] (Required to segregate data according to categories)
			if not train:
				self.data = self.data[self.labels>=20]
				self.labels = self.labels[self.labels>=20]
			if train:
				self.data = self.data[self.labels<20]
				self.labels = self.labels[self.labels<20]
				print("Successfully loaded first 20 categories for training and last 20 for testing!")
			self.labels = self.labels.reshape(-1, 1) 			# [N,]   -> [N, 1]

	def __getitem__(self, idx):
		if self.randomize_data: current_points = self.randomize(idx)
		else: current_points = self.data[idx].copy()

		current_points = torch.from_numpy(current_points[:self.num_points, :]).float()
		label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

		return current_points, label

	def __len__(self):
		return self.data.shape[0]

	def randomize(self, idx):
		pt_idxs = np.arange(0, self.num_points)
		np.random.shuffle(pt_idxs)
		return self.data[idx, pt_idxs].copy()

	def get_shape(self, label):
		return self.shapes[label]

	def read_classes_ModelNet40(self):
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
		file = open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r')
		shape_names = file.read()
		shape_names = np.array(shape_names.split('\n')[:-1])
		return shape_names


class ClassificationData(Dataset):
	def __init__(self, data_class=ModelNet40Data()):
		super(ClassificationData, self).__init__()
		self.set_class(data_class)

	def __len__(self):
		return len(self.data_class)

	def set_class(self, data_class):
		self.data_class = data_class

	def get_shape(self, label):
		try:
			return self.data_class.get_shape(label)
		except:
			return -1

	def __getitem__(self, index):
		return self.data_class[index]


def uniform_2_sphere(num: int = None):
	"""Uniform sampling on a 2-sphere

	Source: https://gist.github.com/andrewbolster/10274979

	Args:
		num: Number of vectors to sample (or None if single)

	Returns:
		Random Vector (np.ndarray) of size (num, 3) with norm 1.
		If num is None returned value will have size (3,)

	"""
	if num is not None:
		phi = np.random.uniform(0.0, 2 * np.pi, num)
		cos_theta = np.random.uniform(-1.0, 1.0, num)
	else:
		phi = np.random.uniform(0.0, 2 * np.pi)
		cos_theta = np.random.uniform(-1.0, 1.0)

	theta = np.arccos(cos_theta)
	x = np.sin(theta) * np.cos(phi)
	y = np.sin(theta) * np.sin(phi)
	z = np.cos(theta)
	
	#xyz =  torch.from_numpy( np.stack((x, y, z), axis=-1))
	return np.stack((x, y, z), axis=-1)



def PointcloudCrop(points, p_keep= 0.7):
	p_keep = np.array(p_keep, dtype=np.float32)

	rand_xyz = uniform_2_sphere()
	pts = points.numpy()
	centroid = np.mean(pts[:, :3], axis=0)
	points_centered = pts[:, :3] - centroid

	dist_from_plane = np.dot(points_centered, rand_xyz)
	#if p_keep ==  0.5:
	#	mask = dist_from_plane > 0
	#else:
	mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 -p_keep) * 100)

	#mask_y = dist_from_plane >= np.percentile(dist_from_plane, (1.0 -p_keep) * 100)

	idx_x = torch.Tensor(np.nonzero(mask))
	#idx_y = torch.Tensor(np.nonzero(mask_y))

	#intersect_mask, intersect_x, intersect_y  = np.intersect1d(idx_x, idx_y, return_indices=True)
	#print(intersect_mask)
	#print(intersect_x)
	#print(intersect_y)
	#gt_mask_source = torch.zeros(points.shape[0])	#tuple(p_source.shape[0])  
	#gt_mask_source [intersect_y]  = 1

	return torch.Tensor(pts[mask, :3]), idx_x#gt_mask_source


class RegistrationData(Dataset):#
	def __init__(self, data_class=ModelNet40Data(), partial= 1, noise= 0, outliers=0):
		super(RegistrationData, self).__init__()
		#print(partial)
		
		self.set_class(data_class)
		self.partial = partial
		self.noise = noise
		self.outliers = outliers

		from .. ops.transform_functions import PNLKTransform
		
	
		self.transforms = PNLKTransform(0.8, True)
		#self.transforms = create_random_transform(torch.float32, 45, 0.5) 
	

	def __len__(self):
		return len(self.data_class)

	def set_class(self, data_class):
		self.data_class = data_class

	def __getitem__(self, index):
		template, label = self.data_class[index]
		source = template.detach()

		gt_mask_y = torch.ones(template.shape[0])
		gt_mask_x = torch.ones(source.shape[0])
		
		if self.partial: 
			source, gt_idx_source = PointcloudCrop(source)
			template, gt_idx_template = PointcloudCrop(template)
			intersect_mask, intersect_x, intersect_y  = np.intersect1d(gt_idx_source, gt_idx_template, return_indices=True)
			gt_mask_y = torch.zeros(template.shape[0])
			gt_mask_x = torch.zeros(source.shape[0])
			gt_mask_y[intersect_y]  = 1
			gt_mask_x[intersect_x]  = 1

		
		if self.noise: 
			source = jitter_pointcloud(source)	
			template = jitter_pointcloud(template)				# Add noise in source point cloud.


		if self.outliers: 
			template, gt_mask_y = add_outliers(template, gt_mask_y)
			source, gt_mask_x = add_outliers(source, gt_mask_x)

		source = self.transforms(source)
		igt = self.transforms.gt
		return  template,source, igt, gt_mask_y ,gt_mask_x


class SegmentationData(Dataset):
	def __init__(self):
		super(SegmentationData, self).__init__()

	def __len__(self):
		pass

	def __getitem__(self, index):
		pass


class FlowData(Dataset):
	def __init__(self):
		super(FlowData, self).__init__()
		self.pc1, self.pc2, self.flow = self.read_data()

	def __len__(self):
		if isinstance(self.pc1, np.ndarray):
			return self.pc1.shape[0]
		elif isinstance(self.pc1, list):
			return len(self.pc1)
		else:
			raise UnknownDataTypeError

	def read_data(self):
		pass

	def __getitem__(self, index):
		return self.pc1[index], self.pc2[index], self.flow[index]


class SceneflowDataset(Dataset):
	def __init__(self, npoints=1024, root='', partition='train'):
		if root == '':
			BASE_DIR = os.path.dirname(os.path.abspath(__file__))
			DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
			root = os.path.join(DATA_DIR, 'data_processed_maxcut_35_20k_2k_8192')
			if not os.path.exists(root): 
				print("To download dataset, click here: https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view")
				exit()
			else:
				print("SceneflowDataset Found Successfully!")

		self.npoints = npoints
		self.partition = partition
		self.root = root
		if self.partition=='train':
			self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
		else:
			self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
		self.cache = {}
		self.cache_size = 30000

		###### deal with one bad datapoint with nan value
		self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
		######
		print(self.partition, ': ',len(self.datapath))

	def __getitem__(self, index):
		if index in self.cache:
			pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
		else:
			fn = self.datapath[index]
			with open(fn, 'rb') as fp:
				data = np.load(fp)
				pos1 = data['points1'].astype('float32')
				pos2 = data['points2'].astype('float32')
				color1 = data['color1'].astype('float32')
				color2 = data['color2'].astype('float32')
				flow = data['flow'].astype('float32')
				mask1 = data['valid_mask1']

			if len(self.cache) < self.cache_size:
				self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

		if self.partition == 'train':
			n1 = pos1.shape[0]
			sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
			n2 = pos2.shape[0]
			sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

			pos1 = pos1[sample_idx1, :]
			pos2 = pos2[sample_idx2, :]
			color1 = color1[sample_idx1, :]
			color2 = color2[sample_idx2, :]
			flow = flow[sample_idx1, :]
			mask1 = mask1[sample_idx1]
		else:
			pos1 = pos1[:self.npoints, :]
			pos2 = pos2[:self.npoints, :]
			color1 = color1[:self.npoints, :]
			color2 = color2[:self.npoints, :]
			flow = flow[:self.npoints, :]
			mask1 = mask1[:self.npoints]

		pos1_center = np.mean(pos1, 0)
		pos1 -= pos1_center
		pos2 -= pos1_center

		return pos1, pos2, color1, color2, flow, mask1

	def __len__(self):
		return len(self.datapath)

class AnyData:
	def __init__(self, pc, mask=False, repeat=1000):
		# pc:			Give any point cloud [N, 3] (ndarray)
		# mask:			False means full source and True mean partial source.

		self.template = torch.tensor(pc, dtype=torch.float32).unsqueeze(0)
		self.template = self.template.repeat(repeat, 1, 1)
		from .. ops.transform_functions import PNLKTransform
		self.transforms = PNLKTransform(mag=0.5, mag_randomly=True)
		self.mask = mask

	def __len__(self):
		return self.template.shape[0]

	def __getitem__(self, index):
		template = self.template[index]
		source = self.transforms(template)
		if self.mask:
			source, gt_mask = farthest_subsample_points(source, num_subsampled_points=int(template.shape[0]*0.7))
		igt = self.transforms.igt
		if self.mask:
			return template, source, igt, gt_mask
		else:
			return template, source, igt

class UserData:
	def __init__(self, template, source, mask=None, igt=None):
		self.template = template
		self.source = source
		self.mask = mask
		self.igt = igt
		self.check_dataset()

	def check_dataset(self):
		if len(self.template)>2:
			assert self.template.shape[0] == self.source.shape[0], "Number of templates are not equal to number of sources."
			if self.mask is None: self.mask = np.zeros((self.template.shape[0], self.template.shape[1], 1))
			if self.igt is None: self.igt = np.eye(4).reshape(1, 4, 4).repeat(self.template.shape[0], 0)
		else:
			self.template = self.template.reshape(1, -1, 3)
			self.source = self.source.reshape(1, -1, 3)
			if self.mask is None: self.mask = np.zeros((1, self.template.shape[0], 1))
			if self.igt is None: self.igt = np.eye(4).reshape(1, 4, 4)
		assert self.template.shape[-1] == 3, "Template point cloud array should have 3 co-ordinates."
		assert self.source.shape[-1] == 3, "Source point cloud array should have 3 co-ordinates."

	def __len__(self):
		if len(self.template.shape) == 2: return 1
		elif len(self.template.shape) == 3: return self.template.shape[0]
		else: print("Error in the data given by user!")

	@staticmethod
	def pc2torch(data):
		return torch.tensor(data).float()

	def __getitem__(self, index):
		template = self.pc2torch(self.template[index])
		source = self.pc2torch(self.source[index])
		mask = self.pc2torch(self.mask[index])
		igt = self.pc2torch(self.igt[index])
		return template, source, mask, igt
		

if __name__ == '__main__':
	class Data():
		def __init__(self):
			super(Data, self).__init__()
			self.data, self.label = self.read_data()

		def read_data(self):
			return [4,5,6], [4,5,6]

		def __len__(self):
			return len(self.data)

		def __getitem__(self, idx):
			return self.data[idx], self.label[idx]

	cd = RegistrationData('abc')
	import ipdb; ipdb.set_trace()