import torch
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils


def points_xyz_to_cylinder(input_xyz):
    rho = torch.sqrt(input_xyz[:, 0]**2 + input_xyz[:, 1]**2)
    phi = torch.atan2(input_xyz[:, 1], input_xyz[:, 0])
    return torch.stack((phi, input_xyz[:, 2], rho), dim=1)


def get_points_idxs(x_in, range_low, voxel_size):
    idxs = (x_in - range_low) / voxel_size
    return idxs


def bilinear_interpolate_torch(im, x, y, align=False):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]
    
    if align:
        x0 = x0.float() + 0.5
        x1 = x1.float() + 0.5
        y0 = y0.float() + 0.5
        y1 = y1.float() + 0.5
    
    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret


def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor['indices'].device
    batch_size = sparse_tensor['batch_size']
    spatial_shape = sparse_tensor['spatial_shape']
    indices = sparse_tensor['indices'].long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor


def three_nearest_neighbor_interpolate(unknown_xyz, unknown_batch_cnt, \
                                        known_xyz, known_batch_cnt, known_features):

    dist, idx = pointnet2_utils.three_nn(unknown_xyz, unknown_batch_cnt, known_xyz, known_batch_cnt)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm
    interpolated_features = pointnet2_utils.three_interpolate(known_features, idx, weight)

    return interpolated_features