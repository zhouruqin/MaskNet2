import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#Mish激活函数
class Mish(nn.Module):
	def __init__(self):
		super(Mish, self).__init__()

	def forward(self, x):
		return x * torch.tanh(F.softplus(x))

		
def knn(x, k):
	inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
	xx = torch.sum(x ** 2, dim=1, keepdim=True)
	pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

	idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
	return idx


def get_graph_feature(x, k=5):
	# x = x.squeeze()
	idx = knn(x, k=k)  # (batch_size, num_points, k)
	batch_size, num_points, _ = idx.size()
	
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

	idx = idx.cuda() + idx_base

	idx = idx.view(-1)

	_, num_dims, _ = x.size()

	# (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
	x = x.transpose(2, 1).contiguous()  

	feature = x.view(batch_size * num_points, -1)[idx, :]
	feature = feature.view(batch_size, num_points, k, num_dims)
	x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

	feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

	return feature


class BasicConv2D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, active = True):
		super(BasicConv2D, self).__init__()
		self. active = active
		#self.bn = nn.BatchNorm1d(in_channels)
		if self.active == True:
			self.activation = Mish()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False)
		#self.dropout = nn.Dropout(0.5) 
	   

	def forward(self, x):
		#x = self.bn(x)
		if self.active == True:
			x = self.activation(x)
		x = self.conv(x)
	   
		return x

#基本残差块
class Resblock2D(nn.Module):
	def __init__(self, channels, out_channels, residual_activation=nn.Identity()):
		super(Resblock2D, self).__init__()

		self.channels = channels
		self.out_channels = out_channels
		if self.channels!= self.out_channels:
			self.res_conv = BasicConv2D(channels, out_channels, 1)
	   
		self.activation = Mish()
		self.block = nn.Sequential(
			BasicConv2D(channels, out_channels//2, 1)   ,
			BasicConv2D(out_channels//2, out_channels, 1 , active = False)       
		)

	def forward(self, x):
		residual =x 
		if self.channels!= self.out_channels:
			residual = self.res_conv(x)
		return self.activation(residual+self.block(x)) 

#基本自注意力块
class Self_Attn(nn.Module):
	""" Self attention Layer"""
	def __init__(self, in_dim, out_dim):
		super(Self_Attn,self).__init__()

		self.in_dim = in_dim
		self.out_dim = out_dim
		
		#查询卷积
		self.query_conv =BasicConv2D(in_dim, out_dim)  
	   
		self.value_conv = nn.Sequential(
			 Resblock2D(in_dim, out_dim),
			 Resblock2D(out_dim, out_dim)
			 )

		self.beta = nn.Parameter(torch.zeros(1))

		self.softmax  = nn.Softmax(dim=-1) #

	def forward(self,x):
		"""
			inputs :
				x : input feature maps( B X C X N)  32, 1024, 64
			returns :
				out : self attention value + input feature 
				attention: B X N X N (N is Width*Height)
		"""

		proj_query  = self.query_conv(x).transpose(2, 3)      # B, in_dim, N   ---> B, in_dim // 8, N   ---->  B, N, in_dim // 8
		proj_key =   proj_query.transpose(2, 3) #B, in_dim, N   ---> B, in_dim // 8, N
		proj_value = self.value_conv(x)  #proj_key# #B, in_dim, N ----> B, out_dim, N
		

		energy =  torch.matmul(proj_query,proj_key) # transpose check    B, N, N

		attention = self.softmax(energy) # B , N,  N  
		
		
	   
		out_x = torch.matmul(proj_value, attention.transpose(2, 3) )   #B, out_dim, N
		
		out =  self.beta * out_x + proj_value #+ proj_value #self.short_conv(x)self.alpha* 
		
		return out#, out1

class DGCNN(torch.nn.Module):
	def __init__(self, emb_dims=512, input_shape="bnc", use_bn=True):
		super(DGCNN, self).__init__()
		if input_shape not in ["bcn", "bnc"]:
			raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")
		self.input_shape = input_shape
		self.emb_dims = emb_dims

		self.conv1 = torch.nn.Conv2d(6, 64, kernel_size=1, bias=False)
		self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=1, bias=False)
		self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=1, bias=False)
		self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=1, bias=False)
		#self.conv5 = torch.nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
		self.bn1 = torch.nn.BatchNorm2d(64)
		self.bn2 = torch.nn.BatchNorm2d(64)
		self.bn3 = torch.nn.BatchNorm2d(128)
		self.bn4 = torch.nn.BatchNorm2d(256)
		#self.bn5 = torch.nn.BatchNorm2d(emb_dims)

	def forward(self, input_data):
		if self.input_shape == "bnc":
			input_data = input_data.permute(0, 2, 1)
		if input_data.shape[1] != 3:
			raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

		batch_size, num_dims, num_points = input_data.size()
		output = get_graph_feature(input_data)

		output = F.relu(self.bn1(self.conv1(output)))#
		output1 = output.max(dim=-1, keepdim=True)[0]

		output = F.relu(self.bn2(self.conv2(output)))#
		output2 = output.max(dim=-1, keepdim=True)[0]

		output = F.relu(self.bn3(self.conv3(output)))#
		output3 = output.max(dim=-1, keepdim=True)[0]

		output = F.relu(self.bn4(self.conv4(output)))#
		output4 = output.max(dim=-1, keepdim=True)[0]

		output = torch.cat((output1, output2, output3, output4), dim=1).view(batch_size, -1, num_points)

		#output = F.relu(self.bn5(self.conv5(output))).view(batch_size, -1, num_points)#
		
		return output


def clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim=-1)
	return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = None

	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)

		# 1) Do all the linear projections in batch from d_model => h x d_k
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
			 for l, x in zip(self.linears, (query, key, value))]

		# 2) Apply attention on all the projected vectors in batch.
		x, self.attn = attention(query, key, value, mask=mask,
								 dropout=self.dropout)

		# 3) "Concat" using a view and apply a final linear.
		x = x.transpose(1, 2).contiguous() \
			.view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."

	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = None

	def forward(self, x):
		return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class EncoderDecoder(nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many
	other models.
	"""

	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator

	def forward(self, src, tgt, src_mask, tgt_mask):
		"Take in and process masked src and target sequences."
		return self.decode(self.encode(src, src_mask), src_mask,
						   tgt, tgt_mask)

	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask)

	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

class Encoder(nn.Module):
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		#self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		for layer in self.layers:
			x = layer(x, mask)
		return x#self.norm(x)


class Decoder(nn.Module):
	"Generic N layer decoder with masking."

	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		#self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, tgt_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return x#self.norm(x)


class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
	def __init__(self, size, dropout=None):
		super(SublayerConnection, self).__init__()
		#self.norm = LayerNorm(size)

	def forward(self, x, sublayer):
		return x + sublayer(x)#self.norm(x)

class EncoderLayer(nn.Module):
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	def forward(self, x, mask):
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
	"Decoder is made of self-attn, src-attn, and feed forward (defined below)"

	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
		"Follow Figure 1 (right) for connections."
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)

class Transformer(nn.Module):
	def __init__(self, emb_dims = 1024, n_blocks = 1, dropout = 0.0, ff_dims = 1024, n_heads = 4):
		super(Transformer, self).__init__()
		self.emb_dims = emb_dims 
		self.N = n_blocks
		self.dropout = dropout
		self.ff_dims = ff_dims
		self.n_heads = n_heads
		c = copy.deepcopy
		attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
		ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
		self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
									Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
									nn.Sequential(),
									nn.Sequential(),
									nn.Sequential())

	def forward(self, *input):
		src = input[0]
		tgt = input[1]
		src = src.transpose(2, 1).contiguous()
		tgt = tgt.transpose(2, 1).contiguous()
		tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
		src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
		return src_embedding, tgt_embedding



class DTFeature(nn.Module):
	def __init__(self, emb_dims=1024, input_shape="bcn"):
		super(DTFeature, self).__init__()
		if input_shape not in ["bcn", "bnc"]:
			raise ValueError(
				"allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
			)
		self.input_shape = input_shape
		self.emb_dims = emb_dims
		
		self.emb_nn = DGCNN(emb_dims=self.emb_dims)
		self.pointer = Transformer()
	


	def forward(self, src, tgt):
	   
		src_embedding = self.emb_nn(src)
		tgt_embedding = self.emb_nn(tgt)

		src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

		src_embedding = src_embedding + src_embedding_p
		tgt_embedding = tgt_embedding + tgt_embedding_p

		return src_embedding, tgt_embedding

if __name__ == '__main__':
	# Test the code.
	x = torch.rand((10,1024,3))

	dgcnn = DGCNN()
	y = dgcnn(x)
	print("\nInput Shape of DGCNN: ", x.shape, "\nOutput Shape of DGCNN: ", y.shape)