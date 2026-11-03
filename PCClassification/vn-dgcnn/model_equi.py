#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Congyue Deng
@Contact: congyue@stanford.edu
@File: model_equi.py
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

EPS = 1e-6

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    """
    Constructs a local geometric graph and computes edge features for Vector Neurons.
    
    This function performs a k-Nearest Neighbors (k-NN) search to find the local neighborhood 
    for each point. It then constructs edge features by concatenating the relative features 
    (neighbor - center) with the center point features.

    Args:
        x (torch.Tensor): Input point features of shape (batch_size, num_channels, 3, num_points).
        k (int): Number of nearest neighbors to consider. Default is 20.
        idx (torch.Tensor, optional): Precomputed k-NN indices. If None, they are computed dynamically.
        x_coord (torch.Tensor, optional): Coordinates to use for k-NN search. If None, 'x' is used.

    Returns:
        torch.Tensor: Edge features of shape (batch_size, 2*num_channels, 3, num_points, k).
    """
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:          # fixed knn graph with input point coordinates
            x_coord = x_coord.view(batch_size, -1, num_points)
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature

    
class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.dense_lin=nn.Linear(in_channels*3,out_channels*3)
        self.unfl=nn.Unflatten(1,(-1,3))
    def forward(self, x,equiv,proj):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        print("I am using VNLinear")
        if equiv:
            x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        else:
            x_out=self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)+proj*self.unfl(self.dense_lin(torch.flatten(x,start_dim=1,end_dim=2).transpose(1,-1)).transpose(1,-1))
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out




class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        
        #self.scale_lin=nn.Parameter(torch.zeros(1)) 
        self.dense_lin=nn.Linear(in_channels*3,out_channels*3)
        self.unfl=nn.Unflatten(1,(-1,3))

        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            self.dense_map_to_dir=nn.Linear(in_channels*3,3,bias=False)
            #self.scale_mtd=nn.Parameter(torch.zeros(1))
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.dense_map_to_dir=nn.Linear(in_channels*3,out_channels*3,bias=False)
            #self.scale_mtd=nn.Parameter(torch.zeros(1))
 
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    
    def forward(self, x,equiv,mix):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        if equiv:
            p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        else:
            p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)+mix*self.unfl(self.dense_lin(torch.flatten(x,start_dim=1,end_dim=2).transpose(1,-1)).transpose(1,-1))
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        if equiv:
            d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        else:
            d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)+mix*self.unfl(self.dense_map_to_dir(torch.flatten(x,start_dim=1,end_dim=2).transpose(1,-1)).transpose(1,-1))
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out

def create_gen(channels):
        genX=torch.tensor([[0,0,0],[0,0,-1],[0,1,0]])
        genY=torch.tensor([[0,0,1],[0,0,0],[-1,0,0]])
        genZ=torch.tensor([[0,-1,0],[1,0,0],[0,0,0]])

        X_in_list=channels*[genX]
        Y_in_list=channels*[genY]
        Z_in_list=channels*[genZ]
        X_in_bl=torch.block_diag(*X_in_list)
        Y_in_bl=torch.block_diag(*Y_in_list)
        Z_in_bl=torch.block_diag(*Z_in_list)
    
        in_bl=torch.cat([X_in_bl,Y_in_bl,Z_in_bl],dim=0)       
        #layer=nn.Linear(3*channels,channels*9,bias=False)
        #with torch.no_grad():

        #    layer.weight.data=in_bl
        return in_bl.T #layer

def LieBracketNorm(in_channels,out_channels):

    genIn=create_gen(in_channels).float().to('cuda')
    genOut=create_gen(out_channels).float().to('cuda')
    return genIn.unsqueeze(0).unsqueeze(0),genOut.unsqueeze(0).unsqueeze(0)


class VNLinearLeakyReLU_Dual(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2, strict_equiv: bool = False):
        super(VNLinearLeakyReLU_Dual, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        self.strict_equiv = strict_equiv
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        
        # non-equivariant-only components
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not self.strict_equiv:
            self.dense_lin = nn.Linear(in_channels*3, out_channels*3)
            self.dense_lin_norm = LieBracketNorm(in_channels, out_channels)
            self.unfl = nn.Unflatten(1, (-1, 3))

        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            if not self.strict_equiv:
                self.dense_map_to_dir = nn.Linear(in_channels*3, 3, bias=False)
                self.dense_map_to_dir_norm = LieBracketNorm(in_channels, 1)
            self.map_out = 1
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            if not self.strict_equiv:
                self.dense_map_to_dir = nn.Linear(in_channels*3, out_channels*3, bias=False)
                self.dense_map_to_dir_norm = LieBracketNorm(in_channels, out_channels)
            self.map_out = out_channels
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)

    
    def forward(self, x,equiv,proj):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        
        if equiv or self.strict_equiv:
            p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
            # return tensor zeros to keep type consistency with relaxed mode
            norm = x.new_zeros(())
            ld_norm = x.new_zeros(())
        else:
            p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)+proj*self.unfl(self.dense_lin(torch.flatten(x,start_dim=1,end_dim=2).transpose(1,-1)).transpose(1,-1))
            addW=torch.flatten(x.detach(),start_dim=1,end_dim=2).transpose(1,-1)
            WA=self.dense_lin((addW@self.dense_lin_norm[0]).reshape(-1,self.in_channels*3,3).permute(0,2,1))
            Wx=self.dense_lin(addW)
            AW=(Wx@self.dense_lin_norm[1]).reshape(-1,self.out_channels*3,3).permute(0,2,1)
            ld_norm=torch.norm((WA-AW).flatten())**2
            norm=torch.norm(Wx.flatten())**2

        p = self.batchnorm(p)
        # LeakyReLU
        if equiv or self.strict_equiv:
            d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        else:
            d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)+proj*self.unfl(self.dense_map_to_dir(torch.flatten(x,start_dim=1,end_dim=2).transpose(1,-1)).transpose(1,-1))
            addW=torch.flatten(x.detach(),start_dim=1,end_dim=2).transpose(1,-1)
            WA=self.dense_map_to_dir((addW@self.dense_map_to_dir_norm[0]).reshape(-1,self.in_channels*3,3).permute(0,2,1))
            Wx=self.dense_map_to_dir(addW)
            AW=(Wx@self.dense_map_to_dir_norm[1]).reshape(-1,self.map_out*3,3).permute(0,2,1)
            ld_norm+=torch.norm((WA-AW).flatten())**2
            norm+=torch.norm(Wx.flatten())**2



        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out,ld_norm,norm


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)

class VNStdFeature_Dual(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2, strict_equiv: bool = False):
        super(VNStdFeature_Dual, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        self.strict_equiv = strict_equiv
        
        self.vn1 = VNLinearLeakyReLU_Dual(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, strict_equiv=strict_equiv)
        self.vn2 = VNLinearLeakyReLU_Dual(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, strict_equiv=strict_equiv)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x,equiv=False,mix=1.0):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0,n1_ld,n1 = self.vn1(z0,equiv,mix)
        z0,n_ld,n = self.vn2(z0,equiv,mix)
        n1_ld+=n_ld
        n1+=n
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0,n1_ld,n1


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x,equiv,mix):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0,equiv,mix)
        z0 = self.vn2(z0,equiv,mix)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0


class EQCNN_cls_normDual(nn.Module):
    def __init__(self, args, output_channels=40, in_channels=3):
        super(EQCNN_cls_normDual, self).__init__()
        self.args = args
        self.k = args.k
        strict_equiv = getattr(args, 'strict_equiv', False)
        
        self.conv1 = VNLinearLeakyReLU_Dual(in_channels // 3 * 2, 64//3, strict_equiv=strict_equiv)
        self.conv2 = VNLinearLeakyReLU_Dual(64//3*2, 64//3, strict_equiv=strict_equiv)
        self.conv3 = VNLinearLeakyReLU_Dual(64//3*2, 128//3, strict_equiv=strict_equiv)
        self.conv4 = VNLinearLeakyReLU_Dual(128//3*2, 256//3, strict_equiv=strict_equiv)

        self.conv5 = VNLinearLeakyReLU_Dual(256//3+128//3+64//3*2, 1024//3, dim=4, share_nonlinearity=True, strict_equiv=strict_equiv)
        
        self.std_feature = VNStdFeature_Dual(1024//3*2, dim=4, normalize_frame=False, strict_equiv=strict_equiv)
        self.linear1 = nn.Linear((1024//3)*12, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)
        
        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x,equiv,mix):
        batch_size = x.size(0)
        num_points = x.size(-1)
        x = x.view(batch_size, -1, 3, num_points)
        x = get_graph_feature(x, k=self.k)
        x,n1_ld,n1 = self.conv1(x,equiv,mix)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.k)
        x,n_ld,n = self.conv2(x,equiv,mix)
        x2 = self.pool2(x)
        n1_ld+=n_ld
        n1+=n
        x = get_graph_feature(x2, k=self.k)
        x,n_ld,n = self.conv3(x,equiv,mix)
        x3 = self.pool3(x)
        n1_ld+=n_ld
        n1+=n
        x = get_graph_feature(x3, k=self.k)
        x,n_ld,n = self.conv4(x,equiv,mix)
        x4 = self.pool4(x)
        n1_ld+=n_ld
        n1+=n
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x,n_ld,n = self.conv5(x,equiv,mix)
        n1_ld+=n_ld
        n1+=n
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans,n_ld,n = self.std_feature(x,equiv,mix)
        n1_ld+=n_ld
        n1+=n

        x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x,(n1_ld,n1)




class EQCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(EQCNN_cls, self).__init__()
        self.args = args
        self.k = args.k
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2, 1024//3, dim=4, share_nonlinearity=True)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024//3)*12, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)
        
        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x,equiv,mix):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x,equiv,mix)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x,equiv,mix)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x,equiv,mix)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x,equiv,mix)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x,equiv,mix)
        
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x,equiv,mix)
        x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x

