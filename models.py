import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv, SAGEConv, SimpleConv
from torch_geometric.utils import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

def topk(
    x,
    batch,
    min_score,  
    tol= 1e-7,
):
    # Make sure that we do not drop all nodes in a graph.
    scores_max = scatter(x, batch, reduce='max')[batch] - tol
    scores_min = scores_max.clamp(max=min_score)
    perm = (x > scores_min)
    return perm

def filter_adj(
    edge_index,
    perm,
    num_nodes,
):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(num_nodes, dtype=torch.long, device=perm.device)
    mask[perm.squeeze()] = i[perm.squeeze()]
    

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]


    return torch.stack([row, col], dim=0)

# MIA Pruner
class MIAPruner(MessagePassing):
    def __init__(self):
        super(MIAPruner, self).__init__(aggr='min')
        self.interface_aggr = SimpleConv(aggr='max')

    def forward(self, d, interface, edge_index, large_val=1e4):
        d_interface = d * interface
        d_interface[d_interface==0] = large_val
        d_interface_out = self.propagate(edge_index, x=d_interface)
        interface_out = self.interface_aggr(edge_index=edge_index, x=interface)
        d_mask = d * interface_out
        
        user_mask = ((d_mask-d_interface_out)<=0).long()
        return user_mask


# Learning Which To Preserve (LWP)
class LWP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(LWP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(input_size, hidden_size))
        for _ in range(num_layers-1):
            self.layers.append(SAGEConv(hidden_size, hidden_size))
        self.layers.append(SAGEConv(hidden_size, 1))

    def forward(self, x, diff, rec_prev, hidden_prev, edge_index):
        out = torch.cat((x, diff, rec_prev, hidden_prev), 1)
        out = self.layers[0](out, edge_index)
        for layer in self.layers[1:]:
            out = F.relu(out)
            out = layer(out, edge_index)
        out = F.sigmoid(out)
        lwp_util = torch.mean(out)
        return out, lwp_util


def _mia_prune(d, interface, edge_index):
    mia_pruner = MIAPruner()
    mask = mia_pruner(d, interface, edge_index)
    return mask

def _mia_diff(x, edge_index, edge_index_prev):
    conv = SimpleConv()
    x_ones_0 = torch.ones(x.size(0), 1, device=x.device)
    x_ones_1 = conv(x=x_ones_0, edge_index=edge_index)
    x_ones_1_prev = conv(x=x_ones_0, edge_index=edge_index_prev)
    x_ones_2 = conv(x=x_ones_1, edge_index=edge_index)
    x_ones_2_prev = conv(x=x_ones_1_prev, edge_index=edge_index_prev)
    return x_ones_0, (x_ones_1-x_ones_1_prev), (x_ones_2-x_ones_2_prev)

def _mia_norm(x, d): #x=[p,s]
    return x/(d**2)

# Partial View De-occlusion Recommender (PDR)
class PDR(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(PDR, self).__init__()
        """
        self.layers = nn.ModuleList()
        self.out_layers = nn.ModuleList()
        self.layers.append(SAGEConv(input_size, hidden_size))
        for _ in range(num_layers-1):
            self.layers.append(SAGEConv(hidden_size, hidden_size))
        for _ in range(num_layers-1):
            self.out_layers.append(SAGEConv(hidden_size, hidden_size))
        self.out_layers.append(SAGEConv(hidden_size, 1))
        """
        self.first_layer = SAGEConv(input_size, hidden_size)
        self.second_layer = SAGEConv(hidden_size, 1)
    
    def forward(self, x, edge_index):
        """
        h = self.layers[0](x, edge_index)
        for layer in self.layers[1:]:
            h = F.relu(h)
            h = layer(h, edge_index)
        
        out = F.relu(h)
        for layer in self.out_layers[:-1]:
            out = layer(out, edge_index)
            out = F.relu(out)
        out = self.out_layers[-1](out, edge_index)
        out = F.sigmoid(out)
        """
        h = self.first_layer(x, edge_index)
        out = F.relu(h)
        out = self.second_layer(out, edge_index)
        out = F.sigmoid(out)
        return out, h

# Preference, Occlusion, Sociality, Hybrid-aware Graph Neural Network (POSHGNN)
class POSHGNN(torch.nn.Module):
    def __init__(self, lwp_input_size, pdr_input_size, hidden_size):
        super(POSHGNN, self).__init__()
        self.lwp = LWP(lwp_input_size + hidden_size, hidden_size)
        self.pdr = PDR(pdr_input_size, hidden_size)
        self.hidden_size = hidden_size
    
    def forward(self, x, rec_prev, hidden_prev, edge_index, edge_index_prev):
        # MIA
        x = self.multi_info_aggregator(x, edge_index, edge_index_prev)

        # select preference, social presence, diff by indices
        p_and_s = x[:,[0, 1]]
        diff = x[:,[2,3,4]]
        mask = x[:,[5]]

        # LWP
        preserve, lwp_util = self.lwp(p_and_s, diff, rec_prev, hidden_prev, edge_index)

        # PDR
        sel, h = self.pdr(p_and_s, edge_index)

        # preserve gate
        rec = rec_prev * preserve + sel * (1-preserve)

        # mask
        rec = mask * rec

        # use 0.5 as threshold for calculating AFTER utility
        batch = edge_index.new_zeros(x.size(0))
        perm = topk(rec, batch, 0.5)

        return rec, perm, h, preserve, lwp_util


    def multi_info_aggregator(self, x, edge_index, edge_index_prev):
        # select preference, social presence, interface, distance by indices
        p_and_s = x[:,[0, 1]]
        i = x[:,[2]]
        d = x[:,[3]]
        d += 1 # to prevent 0 distance normalization
        
        # perform mia tasks
        mask = _mia_prune(d, i, edge_index)
        p_and_s = mask * p_and_s # the utility of occluded user by co-located participant is zero
        x_diff_0, x_diff_1, x_diff_2 = _mia_diff(x, edge_index, edge_index_prev)
        x = _mia_norm(p_and_s, d)

        # aggregate all values
        x = torch.cat((x, x_diff_0, x_diff_1, x_diff_2, mask), 1)

        return x # [p, s, x_diff_0, x_diff_1, x_diff_2, mask]
    
    def ablation_forward_only_PDR(self, x, rec_prev, hidden_prev, edge_index, edge_index_prev):
        ## MIA
        #x = self.multi_info_aggregator(x, edge_index, edge_index_prev)

        # select preference, social presence, diff by indices
        p_and_s = x[:,[0, 1]]
        diff = torch.ones(x.size(0), 3, device=x.device)
        mask = torch.ones(x.size(0), 1, device=x.device)

        # LWP
        #preserve, lwp_util = self.lwp(p_and_s, diff, rec_prev, hidden_prev, edge_index)

        # PDR
        rec, h = self.pdr(p_and_s, edge_index)

        # preserve gate
        #rec = rec_prev * preserve + sel * (1-preserve)

        # mask
        #rec = mask * rec

        # use 0.5 as threshold for calculating AFTER utility
        batch = edge_index.new_zeros(x.size(0))
        perm = topk(rec, batch, 0.5)

        return rec, perm, h, rec, None

    def ablation_forward_PDR_MIA(self, x, rec_prev, hidden_prev, edge_index, edge_index_prev):
        # MIA
        x = self.multi_info_aggregator(x, edge_index, edge_index_prev)

        # select preference, social presence, diff by indices
        p_and_s = x[:,[0, 1, 2,3,4]]
        mask = x[:,[5]]

        # LWP
        #preserve, lwp_util = self.lwp(p_and_s, diff, rec_prev, hidden_prev, edge_index)

        # PDR
        rec, h = self.pdr(p_and_s, edge_index)

        # preserve gate
        #rec = rec_prev * preserve + sel * (1-preserve)

        # mask
        rec = mask * rec

        # use 0.5 as threshold for calculating AFTER utility
        batch = edge_index.new_zeros(x.size(0))
        perm = topk(rec, batch, 0.5)

        return rec, perm, h, rec, None

if __name__ == '__main__':
    num_nodes = 5
    input_size = 8
    hidden_size = 16

    x = torch.rand(num_nodes, input_size)
    edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]])
    gnn = PDR(input_size, hidden_size)
    select, h = gnn(x=x, edge_index=edge_index)
    sel_edge_index = filter_adj(edge_index, select.squeeze().nonzero(), num_nodes=x.size(0))
    opp_edge_index = filter_adj(edge_index, (~select).squeeze().nonzero(), num_nodes=x.size(0))
    print(select)
    print(~select)
    print(h)
    print(h[select])
    print(sel_edge_index)
    print(opp_edge_index)

    
    interface = torch.LongTensor([[0],[0],[1],[0],[1]])
    d = torch.FloatTensor([[3.2], [3.4], [2.6], [3.0], [2.9]])
    edge_index = torch.tensor([[0,1,2,2,3,4], [1,0,3,4,2,2]])
    mia = MIAPruner()
    out = mia(d, interface, edge_index)
    print(out)
    
    print(x)
    print(x[:,[0,1]])
    print(out*x[:,[0,1]])

    posh = POSHGNN(2,2)
    x = posh.multi_info_aggregator(x, edge_index, edge_index)
    print(x)