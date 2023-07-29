try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable
import time
import random
import torch
import numpy as np
from torch_geometric_temporal.signal import temporal_signal_split

from data_gen import read_dataset, mr_dataset
from models import POSHGNN
from utils import AFTER_utility, POSHGNN_loss

def run_train(model, optimizer, train_epoch, train_dataset, hidden_size, beta, alpha, device):
    model.train()
    cost_time = 0
    for ep in tqdm(range(train_epoch)):
        cost_time_epoch = 0
        loss = 0
        hidden_prev = None
        edge_index_prev = None
        rec_prev = None

        for timestep, snapshot in enumerate(train_dataset):
            snapshot.x = snapshot.x.to(device)
            snapshot.edge_index = snapshot.edge_index.to(device)
            if hidden_prev is None:
                hidden_prev = torch.zeros(snapshot.x.size(0), hidden_size).to(device)
            if edge_index_prev is None:
                edge_index_prev = snapshot.edge_index
            if rec_prev is None:
                rec_prev = torch.zeros(snapshot.x.size(0),1).to(device)
            
            start_time = time.time()
            rec, perm, h, preserve, lwp_util = model(snapshot.x, rec_prev, hidden_prev, snapshot.edge_index, edge_index_prev) 

            # calculate loss and time
            cost_time_epoch += time.time()-start_time
            loss += -POSHGNN_loss(snapshot.x, rec, rec_prev, snapshot.edge_index, beta, alpha)

            # set the previous results
            rec_prev = rec
            hidden_prev = h
            edge_index_prev = snapshot.edge_index
        
        loss = loss / (timestep+1)
        cost_time += cost_time_epoch / (timestep+1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return model, optimizer, cost_time/train_epoch

def run_test(model, test_dataset, hidden_size, beta, alpha, device): 
    model.eval()
    utility, loss = 0, 0
    hidden_prev = None
    edge_index_prev = None
    rec_prev = None
    prev_resolved_rec = None
    p, s, o, avg_time = 0,0,0,0
    for t, snapshot in enumerate(test_dataset):
        snapshot.x = snapshot.x.to(device)
        snapshot.edge_index = snapshot.edge_index.to(device)
        if hidden_prev is None:
            hidden_prev = torch.zeros(snapshot.x.size(0), hidden_size).to(device)
        if edge_index_prev is None:
            edge_index_prev = snapshot.edge_index
        if rec_prev is None:
            rec_prev = torch.zeros(snapshot.x.size(0),1).to(device)
        if prev_resolved_rec is None:
            prev_resolved_rec = np.zeros((snapshot.x.size(0),1))

        start = time.time()
        rec, perm, h, preserv, lwp_util = model(snapshot.x, rec_prev, hidden_prev, snapshot.edge_index, edge_index_prev)
        avg_time+= time.time()-start
        loss += -POSHGNN_loss(snapshot.x, rec, rec_prev, snapshot.edge_index, beta, alpha)
        after_utility, resolved_rec, preference, social, occlusion = AFTER_utility(snapshot.x, rec, prev_resolved_rec, snapshot.edge_index, beta, return_all=True)
        utility += after_utility
        p+=preference
        s+=social
        o+=occlusion
        
        # set the previous results
        rec_prev = rec
        hidden_prev = h
        edge_index_prev = snapshot.edge_index
        prev_resolved_rec = resolved_rec
    avg_time/=t
    o/=t
    loss = loss / t
    loss = loss.item()
    return loss, utility, p, s, o, avg_time

if __name__ == '__main__':
    # 0. fix random seed
    seed = 666
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # 1. set hyperparameters
    dataset_name = "example"
    lr = 1e-2
    epoch = 100
    evaluate_per_N_epoch = 1 # epoch should be fully divided by this number
    hidden_size = 8
    lwp_input_size = 6 # p, s, diff_0, diff_1, diff_2, rec_prev
    pdr_input_size = 2 # p, s
    beta = 0.5
    alpha = 1e-2
    T = 100

    # 2. Read dataset
    social_df = read_dataset(dataset_name)
    dataset = mr_dataset(social_df, T=T)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

    # 3. Initialize the model and optimizer
    model = POSHGNN(lwp_input_size, pdr_input_size, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epoch//evaluate_per_N_epoch):
        # 4. Train and evaluate the model
        loss, utility,  p, s, o, avg_time = run_test(model, test_dataset, hidden_size, beta,  alpha, device)
        print("POSH_LOSS: {:.4f}".format(loss))
        print("AFTER_UTIL: {:.4f}".format(utility))
        model, optimizer, cost_time = run_train(model, optimizer, evaluate_per_N_epoch, train_dataset, hidden_size, beta,  alpha, device)
        print("COST_TIME_PER_EPOCH: {:.4f}".format(cost_time))
        
    loss, utility,  p, s, o, avg_time = run_test(model, test_dataset, hidden_size, beta,  alpha, device)
    print("POSH_LOSS: {:.4f}".format(loss))
    print("AFTER_UTIL: {:.4f}".format(utility))
    print(p, s, o, avg_time)