import numpy as np
import torch
import torch.nn as nn
import torch_scatter

from .. import backbones_2d 
from ..model_utils.misc import bilinear_interpolate_torch
from ..model_utils.weight_init import *


class HH3DVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size=None, point_cloud_range=None, **kwargs):
        super().__init__()
        self.point_cloud_range = torch.from_numpy(point_cloud_range).float().cuda()
        self.model_cfg = model_cfg
        self.feature_dim = model_cfg.FEATURE_DIM
        
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.bev_voxel_size = torch.from_numpy(np.array(model_cfg.VOXEL_SIZE_BEV)).float().cuda()

        self.cyl_grid_shape = torch.from_numpy(np.array(model_cfg.GRID_SHAPE_CYL)).int().cuda()
        self.cyl_range = torch.from_numpy(np.array(model_cfg.RV_RANGE)).float().cuda()
        
        # calculate cylindircal view voxel size
        crop_range = torch.tensor(self.cyl_range[[2,3]] - self.cyl_range[[0,1]]).float().cuda()
        self.cyl_voxel_size = crop_range / self.cyl_grid_shape
        self.cyl_scale_xy = self.cyl_grid_shape[0] * self.cyl_grid_shape[1]
        self.cyl_scale_y = self.cyl_grid_shape[1]

        # calculate bird-eye view grid shape
        self.bev_range = torch.from_numpy(np.array(point_cloud_range[[0,1,3,4]])).float().cuda()
        crop_range = self.bev_range[[2,3]] - self.bev_range[[0,1]]
        self.bev_grid_shape = (crop_range / self.bev_voxel_size).round().int()
        self.bev_scale_xy = self.bev_grid_shape[0] * self.bev_grid_shape[1]
        self.bev_scale_y = self.bev_grid_shape[1]
        
        self.input_transform = nn.Sequential(
            nn.Conv1d(model_cfg.INPUT_DIM, self.feature_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.feature_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.feature_dim, self.feature_dim*2, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.feature_dim*2, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.mvf_pointnet = nn.Sequential(
            nn.Conv1d(self.feature_dim*2, self.feature_dim, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(self.feature_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        cyl_net_init_type = model_cfg.get('CYL_NET_INIT_TYPE', 'kaiming_uniform')
        self.cyl_net = backbones_2d.PV_NET[model_cfg.CYL_NET_NAME](self.feature_dim, cyl_net_init_type)

        pointnet_init_type = model_cfg.get('POINTNET_INIT_TYPE', 'kaiming_uniform')
        self.init_weights(pointnet_init_type)

    def init_weights(self, init_type):
        for module_list in [self.input_transform, self.mvf_pointnet]:
            for m in module_list.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                    if init_type == 'kaiming_uniform':
                        kaiming_init(m, distribution='uniform')
                    elif init_type == 'kaiming_normal':
                        kaiming_init(m, distribution='normal')
                    elif init_type == 'xavier':
                        xavier_init(m)
                    elif init_type =='caffe2_xavier':
                        caffe2_xavier_init(m)
            
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)

    def get_output_feature_dim(self):
        return self.feature_dim

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points'] # (batch_idx, x, y, z, r)
        points_cyl = batch_dict['points_cyl'] # (phi, z, rho)
        cyl_idxs = batch_dict['points_cyl_idxs']
        bev_idxs = batch_dict['points_bev_idxs']
        
        bev_coords = torch.floor(bev_idxs).int()
        cyl_coords = torch.floor(cyl_idxs).int()
        points_xyz = points[:, [1,2,3]]
        points_feature = points[:, 4:]
        
        # # get unique bird-eye view and cylindrical view coordinates
        bev_merge_coords = points[:, 0].int() * self.bev_scale_xy + bev_coords[:, 0] * \
                            self.bev_scale_y + bev_coords[:, 1]
        bev_unq_coords, bev_unq_inv, bev_unq_cnt = torch.unique(bev_merge_coords, \
                            return_inverse=True, return_counts=True, dim=0)

        cyl_merge_coords = points[:, 0].int() * self.cyl_scale_xy + cyl_coords[:, 0] * \
                            self.cyl_scale_y + cyl_coords[:, 1]
        cyl_unq_coords, cyl_unq_inv, cyl_unq_cnt = torch.unique(cyl_merge_coords, \
                            return_inverse=True, return_counts=True, dim=0)

        bev_f_center = points_xyz[:, [0,1]] - ((bev_coords.to(points_xyz.dtype) + 0.5) \
                        * self.bev_voxel_size + self.bev_range[[0,1]])
        bev_f_mean = torch_scatter.scatter_mean(points_xyz, bev_unq_inv, dim=0)
        bev_f_cluster = points_xyz - bev_f_mean[bev_unq_inv, :]
        bev_f_cluster = bev_f_cluster[:, [0, 1]]

        cyl_f_center = points_cyl[:, [0,1]] - ((cyl_coords.to(points_cyl.dtype) + 0.5) \
                        * self.cyl_voxel_size + self.cyl_range[[0,1]])
        cyl_f_mean = torch_scatter.scatter_mean(points_cyl, cyl_unq_inv, dim=0)
        cyl_f_cluster = points_cyl - cyl_f_mean[cyl_unq_inv, :]
        cyl_f_cluster = cyl_f_cluster[:, [0, 1]]

        distance = torch.sqrt(torch.sum(points_xyz**2, dim=1, keepdim=True))

        mvf_input = torch.cat([points_xyz,
                               points_cyl,
                               bev_f_center,
                               cyl_f_center,
                               bev_f_cluster,
                               cyl_f_cluster,
                               distance,
                               points_feature
                              ], dim=1).contiguous()
        mvf_input = mvf_input.transpose(0, 1).unsqueeze(0)
        
        pt_fea_in = self.input_transform(mvf_input)
        pt_fea_cyl, pointwise_features = torch.chunk(pt_fea_in, 2, dim=1)

        pt_fea_cyl = pt_fea_cyl.squeeze(0).transpose(0, 1)

        cyl_fea_in = torch_scatter.scatter_max(pt_fea_cyl, cyl_unq_inv, dim=0)[0]
        
        voxel_coords = torch.stack(
            (cyl_unq_coords // self.cyl_scale_xy, 
            (cyl_unq_coords % self.cyl_scale_xy) // self.cyl_scale_y,
            (cyl_unq_coords % self.cyl_scale_y) // 1, cyl_unq_coords % 1), dim=1
        )
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_cyl_features = []
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.feature_dim,
                self.cyl_scale_xy,
                dtype=cyl_fea_in.dtype,
                device=cyl_fea_in.device)

            batch_mask = voxel_coords[:, 0] == batch_idx
            this_coords = voxel_coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.cyl_grid_shape[0] + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = cyl_fea_in[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_cyl_features.append(spatial_feature)
        batch_cyl_features = torch.stack(batch_cyl_features, 0)
        batch_cyl_features = batch_cyl_features.view(batch_size, self.feature_dim, 
                                        self.cyl_grid_shape[1], self.cyl_grid_shape[0])
        batch_cyl_features = batch_cyl_features.permute(0,1,3,2)
        batch_cyl_features = self.cyl_net(batch_cyl_features)

        # bilinear interpolate to get pointwise features
        cyl_point_idxs = cyl_idxs
        point_cyl_feature_list = []
        for k in range(batch_size):
            bs_mask = points[:, 0] == k
            cur_cyl_point_idxs = cyl_point_idxs[bs_mask, :]
            cur_cyl_features = batch_cyl_features[k].permute(1, 2, 0)
            point_cyl_features = bilinear_interpolate_torch(cur_cyl_features, cur_cyl_point_idxs[:, 1], cur_cyl_point_idxs[:, 0])
            point_cyl_feature_list.append(point_cyl_features)
        point_cyl_features = torch.cat(point_cyl_feature_list, dim=0)
        point_cyl_features = point_cyl_features.transpose(0, 1).unsqueeze(0)

        mvf_pt_fea = torch.cat([pointwise_features, point_cyl_features], dim=1)
        mvf_pt_fea = self.mvf_pointnet(mvf_pt_fea)    
        mvf_pt_fea = mvf_pt_fea.squeeze(0).transpose(0, 1)
        bev_max_fea = torch_scatter.scatter_max(mvf_pt_fea, bev_unq_inv, dim=0)[0]

        voxel_coords = torch.stack((bev_unq_coords // self.bev_scale_xy, 
                        (bev_unq_coords % self.bev_scale_xy) // self.bev_scale_y, 
                        bev_unq_coords % self.bev_scale_y, 
                        torch.zeros(bev_unq_coords.shape[0]).to(bev_unq_coords.device).int()
                        ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        rv_proj_dict = {}
        rv_proj_dict.update({
            'rv_point_idxs': cyl_idxs,
            'rv_unq_coords': cyl_unq_coords,
            'rv_unq_inv': cyl_unq_inv,
            'rv_unq_cnt': cyl_unq_cnt,
        })

        bev_proj_dict = {}
        bev_proj_dict.update({
            'bev_point_idxs': bev_idxs,
            'bev_unq_coords': bev_unq_coords,
            'bev_unq_inv': bev_unq_inv,
            'bev_unq_cnt': bev_unq_cnt,
        })

        batch_dict.update({
            'point_xyz': points_xyz,
            'point_features': mvf_pt_fea,
            'point_raw_features': pointwise_features, # (1, C, N1+N2+...)
            'point_cyl_features': point_cyl_features, # (1, C, N1+N2+...)
            'point_batch_inds': points[:, 0].int(),
            'rv_proj_dict': rv_proj_dict,
            'bev_proj_dict': bev_proj_dict,
            'pillar_features': bev_max_fea.contiguous(),
            'voxel_coords': voxel_coords.contiguous()
        })
        

        return batch_dict
