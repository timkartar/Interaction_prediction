import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace


class EQConv(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, num_basis=10):
        super().__init__()

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        #irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
        irreps_out = o3.Irreps(irreps_out)
        irreps_in = o3.Irreps(irreps_in)

        self.irreps_in = irreps_in

        self.tp = o3.FullyConnectedTensorProduct(
                irreps_in1=self.irreps_in,
                irreps_in2=self.irreps_sh,
                irreps_out=irreps_out, shared_weights=False
            )

        self.irreps_out = self.tp.irreps_out
        self.num_basis = num_basis
        self.fc = nn.FullyConnectedNet([self.num_basis, 16, self.tp.weight_numel], torch.relu)
    
    def forward(self, f_in, pos, edge_src, edge_dst, max_radius):
        #edge_src, edge_dst = radius_graph(pos, max_radius, max_num_neighbors=len(pos) - 1)
        avg_num_neighbors = len(edge_src)/len(pos)
        edge_vec = pos[edge_dst] - pos[edge_src]
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, max_radius, self.num_basis, basis='smooth_finite', cutoff=True).mul(self.num_basis**0.5)
        return scatter(self.tp(f_in[edge_src], sh, self.fc(emb)), edge_dst, dim=0, dim_size=len(pos)).div(avg_num_neighbors**0.5)
