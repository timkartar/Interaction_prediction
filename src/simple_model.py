import torch
from mlp import MLP
import sys
from torch_geometric.nn import  PPFConv, PointNetConv, fps, radius, global_max_pool

class SAModule(torch.nn.Module):
    def __init__(self, dim_in, dim_out, ratio, radius,
            aggr='mean',
            max_neighbors=32,
            batch_norm=False,
        ):
        super(SAModule, self).__init__()
        """This module acts as a pooling/conv layer"""
        self.ratio = ratio
        self.r = radius
        self.K = max_neighbors

        # set up convolution
        dim = dim_in + 4
        nn1 = MLP([dim, dim, dim], batch_norm=batch_norm)
        nn2 = MLP([dim, dim_out], batch_norm=batch_norm)
        self.conv = PPFConv(nn1, nn2, add_self_loops=False)
        self.conv.aggr = aggr
    
    def forward(self, x, pos, batch, norm=None):
        # pool points based on FPS algorithm, returning Npt*ratio centroids
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
        
        # finds points within radius `self.r` of the centroids, up to `self.K` pts per centroid
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=self.K)
        
        # perform convolution over edges joining centroids to their neighbors within ball of radius `self.r`
        row = idx[row]
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, pos, norm, edge_index)[idx]
        pos, batch = pos[idx], batch[idx]
        
        return x, pos, batch, idx

class BindingSiteEncoder(torch.nn.Module):
    def __init__(self, dim_in, dim_z,
            depth=4,
            nhidden=32,
            ratios=None,
            radii=None,
            max_neighbors=32,
            batch_norm=False,
            edges_to_count=None
        ):
        super(BindingSiteEncoder, self).__init__()
        self.name = "Binding site encoder"
        self.depth = depth
        if ratios is None:
            ratios = [0.4]*depth
        assert len(ratios) == depth
        
        if radii is None:
            #radii = [2.0]*depth
            radii = [2.4, 5.4, 10.2, 20.2]
            #radii = [2.4, 3.3, 5.4, 7.2]
        assert len(radii) == depth
        
        # linear input
        self.lin_in = MLP([dim_in, nhidden, nhidden])
        
        # conv/pooling layers
        self.SA_modules = torch.nn.ModuleList()
        for i in range(depth):
            self.SA_modules.append(
                SAModule(
                    nhidden,
                    nhidden,
                    ratios[i],
                    radii[i], 
                    max_neighbors=max_neighbors,
                    batch_norm=batch_norm
                )
            )
        
        # latent distribution parameters
        self.lin_out = MLP(
            [nhidden, nhidden, dim_z],
            batch_norm=[batch_norm, False],
            act=['relu', None]
        )
    
    def forward(self, data):
        x = self.lin_in(data.x)
        
        # conv/pooling
        norm = data.norm
        sa_out = (x, data.pos, data.batch)
        for i in range(self.depth):
            args = (*sa_out, norm)
            x, pos, batch, idx = self.SA_modules[i].forward(*args)
            sa_out = (x, pos, batch)
            norm = norm[idx]
        
        # global pool
        x = global_max_pool(x, batch)
        
        # latent distribution parameters
        x = self.lin_out(x)
        
        left = x[edges_to_count[:,0],:]
        right = x[edges_to_count[:,1],:]

        edge_weights = torch.sum(left * right, dim =-1)
        
        return torch.sum(edge_weights)

