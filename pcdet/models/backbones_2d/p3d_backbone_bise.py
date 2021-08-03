import numpy as np
import torch
import torch_scatter
import torch.nn as nn

from .base_bev_backbone import BaseBEVBackbone
from ..model_utils.misc import bilinear_interpolate_torch
from ..model_utils.weight_init import *


class P3DBackbone_BiSE(nn.Module):
    def __init__(self, model_cfg, input_channels, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        num_filters = model_cfg.NUM_FILTERS

        self.bev_net = BaseBEVBackbone(model_cfg.BEV_BACKBONE_2D, input_channels)
        self.bev_stride = model_cfg.BEV_STRIDE
        self.num_bev_features = self.bev_net.num_bev_features

        self.point_cloud_range = torch.tensor(point_cloud_range).float().cuda()
        self.aux_voxel_size = torch.from_numpy(np.array(model_cfg.AUX_VOXEL_SIZE)).float().cuda()
        crop_range = self.point_cloud_range[[3,4,5]] - self.point_cloud_range[[0,1,2]]
        self.aux_grid_size = (crop_range / self.aux_voxel_size).round().int()
        self.aux_scale_xyz = self.aux_grid_size[0] * self.aux_grid_size[1] * self.aux_grid_size[2]
        self.aux_scale_yz  = self.aux_grid_size[1] * self.aux_grid_size[2]
        self.aux_scale_z   = self.aux_grid_size[2]

        self.bev_transform = nn.Sequential(
            nn.Conv2d(self.num_bev_features, num_filters, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters, eps=1e-3, momentum=0.01)
        )

        self.point_features_with_xyz = True

        self.bev_fc1 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True)
        )

        self.cyl_fc1 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True)
        )

        self.bev_fc2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(num_filters),
        )

        self.cyl_fc2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(num_filters),
        )

        self.bev_relu = nn.ReLU()
        self.cyl_relu = nn.ReLU()

        self.bev_att_path = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_filters, num_filters, 1, stride=1, padding=0),
        )

        self.cyl_att_path = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_filters, num_filters, 1, stride=1, padding=0),
        )

        ch_in = num_filters * 2
        if self.point_features_with_xyz:
            ch_in += 3
        self.fusion_transform = nn.Sequential(
            nn.Conv1d(ch_in, num_filters, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True),
        )

        self.voxel_feature_transform = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True),
        )
        
        init_type = model_cfg.get('INIT_TYPE', 'kaiming_uniform')
        self.init_weights(init_type)
        
    def init_weights(self, init_type):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d):
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


    def get_bev_features(self, batch_dict, batch_bev_features):
        batch_size = batch_dict['batch_size']
        point_batch_inds = batch_dict['point_batch_inds']
        bev_proj_dict = batch_dict['bev_proj_dict']

        bev_point_idxs = bev_proj_dict['bev_point_idxs']
        bev_point_idxs = bev_point_idxs / self.bev_stride

        point_bev_feature_list = []
        for k in range(batch_size):
            bs_mask = (point_batch_inds == k)
            cur_bev_point_idxs = bev_point_idxs[bs_mask]
            cur_bev_features = batch_bev_features[k].permute(1, 2, 0).contiguous()
            cur_point_bev_features = bilinear_interpolate_torch(cur_bev_features, \
                                    cur_bev_point_idxs[:, 0], cur_bev_point_idxs[:, 1])
            point_bev_feature_list.append(cur_point_bev_features)
        point_bev_features = torch.cat(point_bev_feature_list, dim=0)

        return point_bev_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                spatial_features
        Returns:
        """
        batch_dict = self.bev_net(batch_dict)
        batch_size = batch_dict['batch_size']
        bev_out = batch_dict['spatial_features_2d']
        
        bev_out = self.bev_transform(bev_out)

        # get bev features by bilinear interpolation
        point_bev_features = self.get_bev_features(batch_dict, bev_out)    
        point_bev_features = point_bev_features.transpose(0, 1).unsqueeze(0)

        point_xyz = batch_dict['point_xyz']
        point_raw_features = batch_dict['point_raw_features']
        point_cyl_features = batch_dict['point_cyl_features']
        point_batch_inds = batch_dict['point_batch_inds']
        
        cyl_x1 = self.cyl_fc1(point_cyl_features)
        bev_x1 = self.bev_fc1(point_bev_features)

        att_cyl_to_bev = torch.sigmoid(self.cyl_att_path(cyl_x1))
        att_bev_to_cyl = torch.sigmoid(self.bev_att_path(bev_x1))

        cyl_x2 = self.cyl_fc2(cyl_x1)
        bev_x2 = self.bev_fc2(bev_x1)

        pt_cyl_pre_fusion = self.cyl_relu(cyl_x1 + att_bev_to_cyl * cyl_x2)
        pt_bev_pre_fusion = self.bev_relu(bev_x1 + att_cyl_to_bev * bev_x2)

        if self.point_features_with_xyz:
            point_xyz_ = point_xyz.transpose(0, 1).unsqueeze(0)
            point_features = torch.cat([pt_cyl_pre_fusion, pt_bev_pre_fusion, point_xyz_], dim=1)
        else:
            point_features = torch.cat([pt_cyl_pre_fusion, pt_bev_pre_fusion], dim=1)
        
        point_features = self.fusion_transform(point_features)
        point_features = point_features.squeeze(0).transpose(0, 1)

        aux_point_idxs = (point_xyz - self.point_cloud_range[[0,1,2]]) / self.aux_voxel_size
        aux_point_coords = torch.floor(aux_point_idxs).int()

        aux_merge_coords = point_batch_inds * self.aux_scale_xyz + \
                           aux_point_coords[:, 0] * self.aux_scale_yz + \
                           aux_point_coords[:, 1] * self.aux_scale_z + \
                           aux_point_coords[:, 2]

        aux_unq_coords, aux_unq_inv, aux_unq_cnt = torch.unique(aux_merge_coords, \
                            return_inverse=True, return_counts=True, dim=0)

        aux_unq_features = torch_scatter.scatter_mean(point_features, aux_unq_inv, dim=0)
        aux_unq_features = aux_unq_features.transpose(0, 1).unsqueeze(0)
        aux_unq_features = self.voxel_feature_transform(aux_unq_features)
        aux_unq_features = aux_unq_features.squeeze(0).transpose(0, 1)

        aux_unq_xyz = torch_scatter.scatter_mean(point_xyz, aux_unq_inv, dim=0)

        aux_unq_coords = torch.stack(
            (aux_unq_coords // self.aux_scale_xyz, (aux_unq_coords % self.aux_scale_xyz) // self.aux_scale_yz,
             (aux_unq_coords % self.aux_scale_yz) // self.aux_scale_z, aux_unq_coords % self.aux_scale_z), dim=1
        )

        aux_sp_tensor_info = {}
        aux_sp_tensor_info.update({
            'features': aux_unq_features.contiguous(),
            'indices': aux_unq_coords[:, [0, 3, 2, 1]].contiguous(),
            'spatial_shape': self.aux_grid_size[[2, 1, 0]].clone(),
            'batch_size': batch_size
        })

        batch_dict.update({
            'aux_sp_tensor': aux_sp_tensor_info,
            'aux_voxel_stride': 1,
            'aux_voxel_size': self.aux_voxel_size,
            'aux_voxel_xyz': aux_unq_xyz
        })

        return batch_dict