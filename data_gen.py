import rvo2
import random
import torch
import numpy as np
import pandas as pd
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, temporal_signal_split

def read_dataset(dataset_name="example"):
    df = pd.read_csv('./dataset/{}.csv'.format(dataset_name))
    df.columns = ['User_U', 'User_V', 'Utility', 'Social']
    return df

def sample_dataset(df, num_users):
    df = df[df['User_U'] == random.choice(df['User_U'])]
    df = df.drop_duplicates(subset=['User_U', 'User_V'])
    if len(df) > num_users:
        df = df.reset_index(drop=True)
        df = df.iloc[:num_users,:]
    else:
        add_num = num_users - len(df) 
        u = df.iloc[0,0]
        v = max(df['User_V'])
        # create additional parts
        new_df = []
        for _ in range(add_num):
            v+=1
            new_df.append([u, v, 0.01, 0.01])
        new_df = pd.DataFrame(new_df, columns = ['User_U', 'User_V', 'Utility', 'Social'])
        df = pd.concat([df, new_df], axis=0)
    df = df.reset_index(drop=True)
    return df

def relabel(df):
    relabel_dict = dict()
    relabel_dict[df['User_U'][0]] = 0
    v_counter = 1
    for v in df['User_V']:
        if v not in relabel_dict:
            relabel_dict[v] = v_counter
            v_counter += 1
    
    df = df.replace({"User_U": relabel_dict, "User_V": relabel_dict})
    
    df.loc[-1] = [0, 0, 1, 1]  # adding a row
    df.index = df.index + 1  # shifting index
    df.sort_index(inplace=True) 
    return df, v_counter

def cart2pol(c):
    rho = np.sqrt(c[0]**2 + c[1]**2)
    phi = np.arctan2(c[1], c[0])
    return np.array([rho, phi])

# Location to Occlusion graph
def abs_loc2rel_pol(L):
    N = L.shape[0]
    relative_C = np.zeros((N, N, 2))
    relative_P = np.zeros((N, N, 2))
    for n in range(N):
        relative_C[n] = L - L[n]
    
    for n in range(N):
        for m in range(N):
            relative_P[n][m] = cart2pol(relative_C[n][m])
    return relative_P

# relative polar to occlusion range
def rel_pol2occ_range(r_P, digital_twin_r=0.5):
    N = r_P.shape[0]
    occ_range = np.zeros((N, N, 3))
    for n in range(N):
        for m in range(N):
            if n != m:
                theta = np.abs(np.arctan(digital_twin_r/(r_P[n][m][0]+1e-8)))
                # what if a user is too close? --> no need to deal with now
                lower_bound = r_P[n][m][1]-theta
                upper_bound = r_P[n][m][1]+theta
                if lower_bound < -np.pi:
                    lower_bound += 2*np.pi
                if upper_bound > np.pi:
                    upper_bound -= 2*np.pi
                occ_range[n][m] = np.array([r_P[n][m][0], lower_bound, upper_bound])
    return occ_range

# range intersect
def intersect(b, a_s, a_e):
    if a_s > a_e:
        if b >= a_s or b <= a_e:
            return True
    else:
        if b>=a_s and b<=a_e:
            return True
    return False

# a and b is a 3-dimensional vector (distance, lower_angle, upper_angle)
def occluded(a, b):
    return intersect(b[1],a[1],a[2]) or intersect(a[1],b[1],b[2])

def occ_range2occ_matrices(occ_range):
    N = occ_range.shape[0]
    occ = np.zeros((N,N,N))
    n = 0
    for m in range(N):
        for o in range(N):
            if occluded(occ_range[n][m],occ_range[n][o]) and n!=m and n!=o and m!=o:
                occ[n][m][o] = 1
    return occ[0]

def loc2occNdist(loc, digital_twin_r=0.5):
    r_P = abs_loc2rel_pol(loc)
    occ_range = rel_pol2occ_range(r_P, digital_twin_r)
    occ = occ_range2occ_matrices(occ_range)
    return occ, r_P[0]

def traj_generator(N, T=20, space_size=10):
    edge_indices = []
    distances = []
    
    # https://gamma.cs.unc.edu/RVO2/documentation/2.0/
    # timeStep,neighborDist,maxNeighbors,timeHorizon,timeHorizonObst,radius
    sim = rvo2.PyRVOSimulator(1/60., 0.5, 10, 1, 2, 0.5, 2)
    agent_list = []
    
    # create agents 
    for _ in range(N):
        agent = sim.addAgent(tuple(np.random.uniform(0,space_size,2)))
        sim.setAgentPrefVelocity(agent, tuple(np.random.uniform(-1,1,2)))
        agent_list.append(agent)
    
    # start simulation
    for step in range(T):
        sim.doStep()
        
        positions = [sim.getAgentPosition(agent_no)
              for agent_no in agent_list]
        positions = np.array(positions)
        
        occ, r_P = loc2occNdist(positions)
        occ = torch.tensor(occ)
        
        # the first agent's view and distance toward others
        edge_index = occ.nonzero().t().contiguous().numpy()
        distance = np.array([np.sqrt(r0**2+r1**2) for r0, r1 in r_P])
        # collect occlusion and distance
        edge_indices.append(edge_index)
        distances.append(distance)
    
    return edge_indices, distances

def integrate_features(df, N, T, edge_indices, distances, VR_probability=0.5):
    immersive = np.random.choice(2, N, p=[VR_probability, 1-VR_probability]) # interface indicator: MR (1) or VR (0)
    df['Interface'] = immersive
    static_feature = df.iloc[:, 2:].to_numpy()
    features = []
    for d in distances:
        distance = np.expand_dims(d, axis=1)
        temporal_feature = np.concatenate((static_feature, distance), axis=1)
        features.append(temporal_feature)
    return DynamicGraphTemporalSignal(edge_indices=edge_indices, features=features, edge_weights=[np.int8(1) for _ in range(T)], targets=[np.int8(0) for _ in range(T)])

def mr_dataset(df, num_users=200, T=3, sample=True, VR_probability=0.5):
    if sample:
        df = sample_dataset(df, num_users)
    df, num_users = relabel(df)
    edge_indices, distances = traj_generator(N=num_users, T=T)
    dataset = integrate_features(df, num_users, T, edge_indices, distances, VR_probability=VR_probability)
    return dataset

     