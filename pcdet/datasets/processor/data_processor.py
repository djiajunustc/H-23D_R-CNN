from functools import partial

import numpy as np

from ...utils import box_utils, common_utils


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict
    
    def compute_indices(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.compute_indices, config=config)
        
        points = data_dict['points']
        rho = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        phi = np.arctan2(points[:, 1], points[:, 0])
        z = points[:, 2]
        points_cyl = np.stack([phi, z, rho], axis=1)

        # compute perspective view index
        cyl_grid_shape = np.array(config.CYL_GRID_SHAPE)
        if config.get('CYL_RANGE', False):
            cyl_range = np.array(config.CYL_RANGE)
            cyl_voxel_size = (cyl_range[2:] - cyl_range[:2]) / cyl_grid_shape
            cyl_idxs = (points_cyl[:, :2] - cyl_range[:2]) / cyl_voxel_size
        else:
            cyl_voxel_size = 2 * np.pi / cyl_grid_shape
            cyl_idxs = (points_cyl[:, :2] + np.pi) / cyl_voxel_size
        
        # compute bird-eye view index
        bev_range = np.array(config.BEV_RANGE)
        bev_voxel_size = np.array(config.BEV_VOXEL_SIZE)
        bev_grid_shape = (bev_range[2:] - bev_range[:2]) / bev_voxel_size
        bev_idxs = (points[:, 0:2] - bev_range[0:2]) / bev_voxel_size
        
        # filter points out of range
        bev_mask = ((bev_idxs[:, 0] >= 0) & (bev_idxs[:, 0] < bev_grid_shape[0]) \
                    & (bev_idxs[:, 1] >=0) & (bev_idxs[:, 1] < bev_grid_shape[1]))
        
        cyl_mask = ((cyl_idxs[:, 0] >= 0) & (cyl_idxs[:, 0] < cyl_grid_shape[0]) \
                    & (cyl_idxs[:, 1] >=0) & (cyl_idxs[:, 1] < cyl_grid_shape[1]))

        mask = cyl_mask & bev_mask

        points = points[mask]
        points_cyl = points_cyl[mask]
        cyl_idxs = cyl_idxs[mask]
        bev_idxs = bev_idxs[mask]
        
        data_dict['points'] = points
        data_dict['points_cyl'] = points_cyl
        data_dict['points_cyl_idxs'] = cyl_idxs
        data_dict['points_bev_idxs'] = bev_idxs
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
