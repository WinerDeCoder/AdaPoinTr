import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
# from knn_cuda import KNN
# knn = KNN(k=16, transpose_mode=False)


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region     k =  16
        xyz: all points, [B, N, C]      B, N, 3
        new_xyz: query points, [B, S, C]         B, N, 3
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx  # B, 2048, 16

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist  


class DGCNN_Grouper(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(3, 8, 1)  # C =3 --> C = 8

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False), # B,8,N,16 --> B,32,N,16
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)  # B, 512

        combined_x = torch.cat([coor, x], dim=1)  # B,3,N + B,16,N ---> B,19,N . Ini points + features

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x  , fps_idx 
            )
        )
        # gộp thêm Center point ---> B, C(19) , npoint

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x # B,3,npoint    ------ B,16, npoint

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = 16
        batch_size = x_k.size(0)  # B
        num_points_k = x_k.size(2) # N (2048)
        num_points_q = x_q.size(2) # N  (2048)

        with torch.no_grad():
#             _, idx = knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            # coor_k = B, N, 8  -----coor_q = B, N, 3 ----
            # idx shape = B,N (2048),16
            idx = idx.transpose(-1, -2).contiguous() # idx.shape = B,16,N (2048)
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            # torch.arange() -> [0,1,....,batch_size-1] --> view = [batch_size,1,1] -> each * with N ( number of point)
            idx = idx + idx_base # B,N,16 + batch_size,1,1 = B,N (2048),16 
            idx = idx.view(-1)  # B*N*C(16)
        num_dims = x_k.size(1)  # = numdims = C =8
        x_k = x_k.transpose(2, 1).contiguous() # B, N (2048), C(8)
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :] 
        # x_k.view(bs*N,-1) -> B,N (2048),C(8) --> B*N (2048),C (8)
        # [idx, :] -- > take first 
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature  # B, 8+8, N, k (16)

    def forward(self, x):

        # x: bs, 3, np (2048)

        # bs 3 N(128)   bs C(224)128 N(128)
        coor = x  # B,3,N
        f = self.input_trans(x)  # B, C (8),N (2048)

        f = self.get_graph_feature(coor, f, coor, f)   # feature = B, 8 + 8, N, k (16)
        f = self.layer1(f)  # B, 32, N, 16
        f = f.max(dim=-1, keepdim=False)[0]  # B,32,N ---> Max value of dim=-1 ;là k. Điểm gần nhất hay chính nó

        coor_q, f_q = self.fps_downsample(coor, f, 512)  
        # new_coor_q = B,3,512 ---- new_f_q = B,16,512
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, 128)
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q
        # coor la sau khi qua fps , tu coor se qua la pos embedding, f la tu LDGCNN
        return coor, f  # coor: B, 3, 128 ----- f: B, ? ,128
