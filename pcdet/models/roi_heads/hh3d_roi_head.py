import torch
import torch.nn as nn
import torch.nn.functional as F
from .roi_head_template import RoIHeadTemplate
from ..model_utils.misc import generate_voxel2pinds
from ..model_utils.weight_init import *
from ...utils import common_utils
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules_v2 as pointnet2_stack_modules


class HH3DRoIHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        self.point_cloud_range = torch.from_numpy(point_cloud_range).float().cuda()

        COARSE_LAYER_cfg = self.pool_cfg.COARSE_POOL_LAYERS
        coarse_mlps = COARSE_LAYER_cfg.MLPS
        for k in range(len(coarse_mlps)):
            coarse_mlps[k] = [input_channels] + coarse_mlps[k]
        self.coarse_pool_layer = pointnet2_stack_modules.NeighborVoxelSAModuleMSG_v2(
            query_ranges=COARSE_LAYER_cfg.QUERY_RANGES,
            nsamples=COARSE_LAYER_cfg.NSAMPLE,
            radii=COARSE_LAYER_cfg.POOL_RADIUS,
            mlps=coarse_mlps,
            pool_method=COARSE_LAYER_cfg.POOL_METHOD,
        )
        coarse_c_out = sum([x[-1] for x in coarse_mlps])

        FINE_LAYER_cfg = self.pool_cfg.FINE_POOL_LAYERS
        fine_mlps = FINE_LAYER_cfg.MLPS
        for k in range(len(fine_mlps)):
            fine_mlps[k] = [input_channels] + fine_mlps[k]
        self.fine_pool_layer = pointnet2_stack_modules.NeighborVoxelSAModuleMSG_v2(
            query_ranges=FINE_LAYER_cfg.QUERY_RANGES,
            nsamples=FINE_LAYER_cfg.NSAMPLE,
            radii=FINE_LAYER_cfg.POOL_RADIUS,
            mlps=fine_mlps,
            pool_method=FINE_LAYER_cfg.POOL_METHOD,
        )
        fine_c_out = sum([x[-1] for x in fine_mlps])

        FINE_GRID_SIZE_LIST = self.model_cfg.ROI_GRID_POOL.FINE_GRID_SIZE_LIST
        pre_channel = FINE_GRID_SIZE_LIST[0] * FINE_GRID_SIZE_LIST[1] * FINE_GRID_SIZE_LIST[2] * (coarse_c_out + fine_c_out)
        # pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)

        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)

        # self.init_weights(weight_init='xavier')
        init_type = model_cfg.get('INIT_TYPE', 'xavier')
        self.init_weights(init_type)

    def init_weights(self, init_type):
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    if init_type == 'kaiming_uniform':
                        kaiming_init(m, distribution='uniform')
                    elif init_type =='caffe2_xavier':
                        caffe2_xavier_init(m)
                    elif init_type == 'xavier':
                        xavier_init(m)
                    
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)
    
    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:
        """
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        voxel_size = batch_dict['aux_voxel_size']
        COARSE_GRID_SIZE_LIST = self.pool_cfg.COARSE_GRID_SIZE_LIST
        FINE_GRID_SIZE_LIST = self.pool_cfg.FINE_GRID_SIZE_LIST

        # coarse grid RoI pooling
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size_3d=(COARSE_GRID_SIZE_LIST[0], COARSE_GRID_SIZE_LIST[1], COARSE_GRID_SIZE_LIST[2])
        )  # (BxN, 3x3x3, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3) # (B, Nx3x3x3, 3)
        # compute the voxel coordinates of grid points
        roi_grid_coords = torch.floor((roi_grid_xyz - self.point_cloud_range[0:3]) / voxel_size).int() # (B, Nx3x3x3, 3)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        batch_idx = batch_idx.int()

        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])
        
        cur_stride = batch_dict['aux_voxel_stride']
        cur_voxel_xyz = batch_dict['aux_voxel_xyz']
        cur_sp_tensors = batch_dict['aux_sp_tensor']
        cur_coords = cur_sp_tensors['indices']
        cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
        # get voxel2point tensor
        v2p_ind_tensor = generate_voxel2pinds(cur_sp_tensors)

        # compute the grid coordinates in this cale, in [batch_ind, x, y, z] order
        cur_roi_grid_coords = roi_grid_coords // cur_stride
        cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
        cur_roi_grid_coords = cur_roi_grid_coords.int()
        
        coarse_pooled_features = self.coarse_pool_layer(
            xyz=cur_voxel_xyz.contiguous(),
            xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
            new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
            new_xyz_batch_cnt=roi_grid_batch_cnt,
            new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
            features=cur_sp_tensors['features'].contiguous(),
            voxel2point_indices=v2p_ind_tensor
        )

        coarse_pooled_features = coarse_pooled_features.view(
            -1, COARSE_GRID_SIZE_LIST[0], COARSE_GRID_SIZE_LIST[1], COARSE_GRID_SIZE_LIST[2],
            coarse_pooled_features.shape[-1]
        ) # (BxN, 3, 3, 3, C)

        coarse_pooled_features = coarse_pooled_features.permute(0, 4, 1, 2, 3) # (BxN, C, 3, 3, 3)
        coarse_pooled_features = F.upsample(coarse_pooled_features, (FINE_GRID_SIZE_LIST[0], FINE_GRID_SIZE_LIST[1],  FINE_GRID_SIZE_LIST[2])) # (BxN, C, 6, 6, 6)
        coarse_pooled_features = coarse_pooled_features.permute(0, 2, 3, 4, 1) # (BxN, 6, 6, 6, C)
        # coarse_pooled_features = coarse_pooled_features.reshape(-1, coarse_pooled_features.shape[-1])

        # fine grid RoI pooling
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size_3d=(FINE_GRID_SIZE_LIST[0], FINE_GRID_SIZE_LIST[1], FINE_GRID_SIZE_LIST[2])
        )  # (BxN, 6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3) # (B, Nx3x3x3, 3)
        # compute the voxel coordinates of grid points
        roi_grid_coords = torch.floor((roi_grid_xyz - self.point_cloud_range[0:3]) / voxel_size).int() # (B, Nx3x3x3, 3)

        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        batch_idx = batch_idx.int()

        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])
        
        cur_stride = batch_dict['aux_voxel_stride']
        cur_voxel_xyz = batch_dict['aux_voxel_xyz']
        cur_sp_tensors = batch_dict['aux_sp_tensor']
        cur_coords = cur_sp_tensors['indices']
        cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
        # get voxel2point tensor
        v2p_ind_tensor = generate_voxel2pinds(cur_sp_tensors)

        # compute the grid coordinates in this cale, in [batch_ind, x, y, z] order
        cur_roi_grid_coords = roi_grid_coords // cur_stride
        cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
        cur_roi_grid_coords = cur_roi_grid_coords.int()
        
        fine_pooled_features = self.fine_pool_layer(
            xyz=cur_voxel_xyz.contiguous(),
            xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
            new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
            new_xyz_batch_cnt=roi_grid_batch_cnt,
            new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
            features=cur_sp_tensors['features'].contiguous(),
            voxel2point_indices=v2p_ind_tensor
        )

        fine_pooled_features = fine_pooled_features.view(
            -1, FINE_GRID_SIZE_LIST[0], FINE_GRID_SIZE_LIST[1], FINE_GRID_SIZE_LIST[2],
            coarse_pooled_features.shape[-1]
        ) # (BxN, 6, 6, 6, C)

        pooled_features = torch.cat([coarse_pooled_features, fine_pooled_features], dim=-1)

        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size_3d=(5, 5, 5)):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size_3d)  # (B, 7x7, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)

        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size_3d=(5, 5, 5)):
        faked_features = rois.new_ones((grid_size_3d))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        # import pdb
        # pdb.set_trace()

        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 7x7, 3)
        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]

        grid_size_3d_tensor = torch.tensor(grid_size_3d).to(rois.device)
        grid_size_3d_tensor = grid_size_3d_tensor.unsqueeze(0).unsqueeze(1)
        roi_grid_points = (dense_idx + 0.5) / grid_size_3d_tensor * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 7x7, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        # Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict