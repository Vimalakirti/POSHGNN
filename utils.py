import torch
from torch_geometric.utils import to_dense_adj
from models import _mia_diff, _mia_norm, _mia_prune

# helper function to resolve view occlusion (occluded users should not be counted in AFTER utility)
def resolve_rec(edge_index, relative_distance, rec, interface):
    new_rec = rec.copy()
    new_rec += interface
    new_rec = (new_rec>=1).astype(int)
    
    rows, cols = edge_index
    for r, c in zip(rows, cols):
        if relative_distance[r] > relative_distance[c]:
            new_rec[r] = 0
        elif relative_distance[r] < relative_distance[c]:
            new_rec[c] = 0
    return new_rec

def AFTER_utility(x, rec, prev_resolved_rec, edge_index, beta, return_all=False):
    p_and_s = x[:,[0, 1]]
    i = x[:,[2]]
    d = x[:,[3]]
    d += 1
    
    # perform mia tasks
    mask = _mia_prune(d, i, edge_index)
    p_and_s = mask * p_and_s # the utility of occluded user by co-located participant is zero
    x = _mia_norm(p_and_s, d)
    
    p = x[:, 0].cpu().numpy()
    s = x[:, 1].cpu().numpy()
    interface = i.long().cpu().numpy()
    
    orig_rec_num = (rec>=0.5).long().cpu().detach().numpy().sum()
    rec *= mask
    rec_numpy = (rec>=0.5).long().cpu().detach().numpy()
    new_resolved_rec = resolve_rec(edge_index, d, rec_numpy, interface)
    new_resolved_rec_num = new_resolved_rec.sum()
    occlusion = (orig_rec_num-new_resolved_rec_num)/orig_rec_num
    
    preference, social = p * new_resolved_rec, s * new_resolved_rec * prev_resolved_rec
    after_utility = (1-beta)*preference + beta*social 
    after_utility = after_utility.sum()
    if return_all:
        return after_utility, new_resolved_rec, preference.sum(), social.sum(), occlusion
    
    return after_utility, new_resolved_rec

def POSHGNN_loss(x, rec, prev_rec, edge_index, beta, alpha, return_all=False):
    """
    Inputs:
    1. x: [p, s, i, d]
    2. rec \in [0,1]
    3. prev_rec \in [0,1]
    4. edge_index
    Terms:
    1. -gamma? all -(social+preference)
    2. +preference
    3. +social
    4. -rec occlusion
    (5. +delta utility between consecutive time steps for LWP)
    """ 
    # select preference, social presence, interface, distance by indices
    p_and_s = x[:,[0, 1]]
    i = x[:,[2]]
    d = x[:,[3]]
    d += 1
    
    # perform mia tasks
    mask = _mia_prune(d, i, edge_index)
    p_and_s = mask * p_and_s # the utility of occluded user by co-located participant is zero
    x = _mia_norm(p_and_s, d)

    p = x[:, [0]]
    s = x[:, [1]]
    batch = edge_index.new_zeros(x.size(0))
    adj = to_dense_adj(edge_index, batch)
    
    gamma = -(beta*s+(1-beta)*p).sum()
    preference = ((1-beta)*p*rec).sum()
    social = (beta*s*prev_rec*rec).sum()
    occlusion = -torch.matmul(torch.matmul(rec.t(), adj), rec).sum()*alpha
    if return_all:
        return (gamma+preference+social+occlusion), gamma, preference, social, occlusion
    return gamma+preference+social+occlusion#+continuity