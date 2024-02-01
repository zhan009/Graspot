import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from .GRASPOT import Graspot
from .OT_utils import distance_matrix, unbalanced_ot, unbalanced_ot_parameter

import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import scipy
import ot

def norm_and_center_coordinates(X):
    """
    Normalizes and centers coordinates at the origin.

    Args:
        X: Numpy array

    Returns:
        X_new: Updated coordiantes.
    """
    return (X-X.mean(axis=0))/min(scipy.spatial.distance.pdist(X))


def train_Graspot(adata, hidden_dims=[512, 30], n_epochs=200, lr=0.001, key_added='Graspot',
                             gradient_clipping=5., weight_decay=0.0001, verbose=False,
                             random_seed=2023, iter_comb=None, Batch_list=None, initial=None, Couple=None,
                             device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Train Graspot including GAT module and UOT alignment module.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    iter_comb
        iter_comb is used to specify the order of integration.
    Batch_list
        Multiple slices in integration. Default: None
    initial
        initial transport matrix setting in unbalanced optimal transport. Default: None
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    section_ids = np.array(adata.obs['batch_name'].unique())

    comm_gene = adata.var_names
    data_list = []
    for adata_tmp in Batch_list:
        adata_tmp = adata_tmp[:, comm_gene]
        edge_index = np.nonzero(adata_tmp.uns['adj'])
        data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_index[0], edge_index[1]])),
                              prune_edge_index=torch.LongTensor(np.array([])),
                              x=torch.FloatTensor(adata_tmp.X.todense())))

    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    model = Graspot(hidden_dims=[adata.X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)

    print('Train with Graspot...')
    pair_data_list = []
    for comb in iter_comb:
        #print(comb)
        i, j = comb[0], comb[1]
        batch_pair = adata[adata.obs['batch_name'].isin([section_ids[i], section_ids[j]])]

        edge_list_1 = np.nonzero(Batch_list[i].uns['adj'])
        max_ind = edge_list_1[0].max()
        edge_list_2 = np.nonzero(Batch_list[j].uns['adj'])
        edge_list_2 = (edge_list_2[0] + max_ind + 1, edge_list_2[1] + max_ind + 1)
        edge_list = [edge_list_1, edge_list_2]
        edge_pairs = [np.append(edge_list[0][0], edge_list[1][0]), np.append(edge_list[0][1], edge_list[1][1])]
        pair_data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_pairs[0], edge_pairs[1]])),
                                           x=torch.FloatTensor(batch_pair.X.todense())))

    pair_loader = DataLoader(pair_data_list, batch_size=1, shuffle=False)
    
    tran_list=[]
    for iters, batch in enumerate(pair_loader):
        
        
        if initial == True:
            ax = Batch_list[iter_comb[iters][0]].obsm['spatial']
            ay = Batch_list[iter_comb[iters][1]].obsm['spatial']
            dist = scipy.spatial.distance_matrix(norm_and_center_coordinates(ax),norm_and_center_coordinates(ay))
            #dist = scipy.spatial.distance_matrix(ax,ay)
            n1 = ax.shape[0]
            n2 = ay.shape[0]
            pi0 = pi = ot.sinkhorn(np.ones(n1)/n1, np.ones(n2)/n2, dist, reg=0.02)
            tran = torch.tensor(pi0, dtype=torch.float).to(device)
        else:
            tran = None
            
        num = []
        for i in [Batch_list[iter_comb[iters][0]],Batch_list[iter_comb[iters][1]]]:
            num.append(i.X.shape[0])
                
        for epoch in tqdm(range(0, n_epochs)):
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
        
            z, out = model(batch.x, batch.edge_index)
            mse_loss = F.mse_loss(batch.x, out)
        
            ds1 = z[0:num[0],:]
            ds2 = z[num[0]:num[0]+num[1],:]
            
            ot_loss, tran = unbalanced_ot(tran, ds1, ds2, device=device, Couple=Couple, reg=0.1, reg_m=1.0)
        
            loss = 0.8 * mse_loss + 0.2 * ot_loss
            #loss = mse_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
         
        tran_list.append(tran)
        
    
    #
    model.eval()
    with torch.no_grad():
        z_list = []
        for iters, batch in enumerate(data_list):
            z, _ = model.cpu()(batch.x, batch.edge_index)
            z_list.append(z.cpu().detach().numpy())
    adata.obsm[key_added] = np.concatenate(z_list, axis=0)    
    
    #z_sublist = []
    #for comb in iter_comb:
        #print(comb)
        #i, j = comb[0], comb[1]
        #z_sublist = [z_list[i],z_list[j]] 
        #adata.obsm[key_added + 'i'] = np.concatenate(z_sublist, axis=0)

    return adata, tran_list


def train_Graspot_Sub(adata, hidden_dims=[512, 30], n_epochs=200, lr=0.001, key_added='Graspot',
                             gradient_clipping=5., weight_decay=0.0001, verbose=False,
                             random_seed=2023, iter_comb=None, Batch_list=None, initial=None, Couple=None,
                             device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Train Graspot including GAT module and UOT alignment module with Pretrain.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    iter_comb
        iter_comb is used to specify the order of integration.
    Batch_list
        Multiple slices in integration. Default: None
    initial
        initial transport matrix setting in unbalanced optimal transport. Default: None
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    section_ids = np.array(adata.obs['batch_name'].unique())

    comm_gene = adata.var_names
    data_list = []
    for adata_tmp in Batch_list:
        adata_tmp = adata_tmp[:, comm_gene]
        edge_index = np.nonzero(adata_tmp.uns['adj'])
        data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_index[0], edge_index[1]])),
                              prune_edge_index=torch.LongTensor(np.array([])),
                              x=torch.FloatTensor(adata_tmp.X.todense())))

    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    model = Graspot(hidden_dims=[adata.X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)
    
    
    print('Pretrain...')
    for epoch in tqdm(range(0, 200)):
        for batch in loader:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            z_pre, out_pre = model(batch.x, batch.edge_index)

            loss = F.mse_loss(batch.x, out_pre)  # +adv_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

    with torch.no_grad():
        z_list_pre = []
        for batch in data_list:
            z_pre, _ = model.cpu()(batch.x, batch.edge_index)
            z_list_pre.append(z_pre.cpu().detach().numpy())
    adata.obsm['STAGATE'] = np.concatenate(z_list_pre, axis=0)
    model = model.to(device)
    

    print('Train with Graspot...')
    pair_data_list = []
    for comb in iter_comb:
        #print(comb)
        i, j = comb[0], comb[1]
        batch_pair = adata[adata.obs['batch_name'].isin([section_ids[i], section_ids[j]])]

        edge_list_1 = np.nonzero(Batch_list[i].uns['adj'])
        max_ind = edge_list_1[0].max()
        edge_list_2 = np.nonzero(Batch_list[j].uns['adj'])
        edge_list_2 = (edge_list_2[0] + max_ind + 1, edge_list_2[1] + max_ind + 1)
        edge_list = [edge_list_1, edge_list_2]
        edge_pairs = [np.append(edge_list[0][0], edge_list[1][0]), np.append(edge_list[0][1], edge_list[1][1])]
        pair_data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_pairs[0], edge_pairs[1]])),
                                           x=torch.FloatTensor(batch_pair.X.todense())))

    pair_loader = DataLoader(pair_data_list, batch_size=1, shuffle=False)
    
    tran_list=[]
    for iters, batch in enumerate(pair_loader):
                
        if initial == True:
            ax = Batch_list[iter_comb[iters][0]].obsm['spatial']
            ay = Batch_list[iter_comb[iters][1]].obsm['spatial']
            dist = scipy.spatial.distance_matrix(norm_and_center_coordinates(ax),norm_and_center_coordinates(ay))
            #dist = scipy.spatial.distance_matrix(ax,ay)
            n1 = ax.shape[0]
            n2 = ay.shape[0]
            pi0 = pi = ot.sinkhorn(np.ones(n1)/n1, np.ones(n2)/n2, dist, reg=0.02)
            tran = torch.tensor(pi0, dtype=torch.float).to(device)
        else:
            tran = None
        #ds1 = torch.tensor(z_list_pre[0])
        #ds2 = torch.tensor(z_list_pre[1])
        #ot_loss, tran = unbalanced_ot(tran, ds1, ds2, device=device, reg=0.1, reg_m=1.0)
            
        num = []
        for i in [Batch_list[iter_comb[iters][0]],Batch_list[iter_comb[iters][1]]]:
            num.append(i.X.shape[0])
            
        for epoch in tqdm(range(0, n_epochs)):
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
        
            z, out = model(batch.x, batch.edge_index)
            mse_loss = F.mse_loss(batch.x, out)
        
            ds1 = z[0:num[0],:]
            ds2 = z[num[0]:num[0]+num[1],:]
            
            ot_loss, tran = unbalanced_ot(tran, ds1, ds2, device=device, Couple=Couple, reg=0.1, reg_m=1.0)
        
            loss = 0.8 * mse_loss + 0.2 * ot_loss
            #loss = mse_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
         
        tran_list.append(tran)
        
    
    #
    model.eval()
    with torch.no_grad():
        z_list = []
        for iters, batch in enumerate(data_list):
            z, _ = model.cpu()(batch.x, batch.edge_index)
            z_list.append(z.cpu().detach().numpy())
    adata.obsm[key_added] = np.concatenate(z_list, axis=0)    
    
    #z_sublist = []
    #for comb in iter_comb:
        #print(comb)
        #i, j = comb[0], comb[1]
        #z_sublist = [z_list[i],z_list[j]] 
        #adata.obsm[key_added + 'i'] = np.concatenate(z_sublist, axis=0)

    return adata, tran_list


def train_Graspot_Para(adata, hidden_dims=[512, 30], n_epochs=200, lr=0.001, key_added='Graspot',
                             gradient_clipping=5., weight_decay=0.0001, verbose=False,
                             random_seed=2023, iter_comb=None, Batch_list=None, initial=None, Couple=None,
                             device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
   Train Graspot including GAT module and UOT alignment module with different Unbalanced OT parameters.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    iter_comb
        iter_comb is used to specify the order of integration.
    Batch_list
        Multiple slices in integration. Default: None
    initial
        initial transport matrix setting in unbalanced optimal transport. Default: None
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    section_ids = np.array(adata.obs['batch_name'].unique())

    comm_gene = adata.var_names
    data_list = []
    for adata_tmp in Batch_list:
        adata_tmp = adata_tmp[:, comm_gene]
        edge_index = np.nonzero(adata_tmp.uns['adj'])
        data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_index[0], edge_index[1]])),
                              prune_edge_index=torch.LongTensor(np.array([])),
                              x=torch.FloatTensor(adata_tmp.X.todense())))

    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    model = Graspot(hidden_dims=[adata.X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if verbose:
        print(model)

    print('Train with Graspot...')
    pair_data_list = []
    for comb in iter_comb:
        #print(comb)
        i, j = comb[0], comb[1]
        batch_pair = adata[adata.obs['batch_name'].isin([section_ids[i], section_ids[j]])]

        edge_list_1 = np.nonzero(Batch_list[i].uns['adj'])
        max_ind = edge_list_1[0].max()
        edge_list_2 = np.nonzero(Batch_list[j].uns['adj'])
        edge_list_2 = (edge_list_2[0] + max_ind + 1, edge_list_2[1] + max_ind + 1)
        edge_list = [edge_list_1, edge_list_2]
        edge_pairs = [np.append(edge_list[0][0], edge_list[1][0]), np.append(edge_list[0][1], edge_list[1][1])]
        pair_data_list.append(Data(edge_index=torch.LongTensor(np.array([edge_pairs[0], edge_pairs[1]])),
                                           x=torch.FloatTensor(batch_pair.X.todense())))

    pair_loader = DataLoader(pair_data_list, batch_size=1, shuffle=False)
    
    tran_list=[]
    for iters, batch in enumerate(pair_loader):
        
        
        if initial == True:
            ax = Batch_list[iter_comb[iters][0]].obsm['spatial']
            ay = Batch_list[iter_comb[iters][1]].obsm['spatial']
            #dist = scipy.spatial.distance_matrix(norm_and_center_coordinates(ax),norm_and_center_coordinates(ay))
            dist = scipy.spatial.distance_matrix(ax,ay)
            n1 = ax.shape[0]
            n2 = ay.shape[0]
            pi0 = pi = ot.sinkhorn(np.ones(n1)/n1, np.ones(n2)/n2, dist, reg=0.02)
            tran = torch.tensor(pi0, dtype=torch.float).to(device)
        else:
            tran = None
            
        num = []
        for i in [Batch_list[iter_comb[iters][0]],Batch_list[iter_comb[iters][1]]]:
            num.append(i.X.shape[0])
                
        for epoch in tqdm(range(0, n_epochs)):
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
        
            z, out = model(batch.x, batch.edge_index)
            mse_loss = F.mse_loss(batch.x, out)
        
            ds1 = z[0:num[0],:]
            ds2 = z[num[0]:num[0]+num[1],:]
            
            ot_loss, tran = unbalanced_ot_parameter(tran, ds1, ds2, device=device, Couple=Couple, 
                                                    reg=0.1, reg_m_1 = 0.01, reg_m_2 = 100)
        
            loss = 0.8 * mse_loss + 0.2 * ot_loss
            #loss = mse_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
         
        tran_list.append(tran)
        
    
    #
    model.eval()
    with torch.no_grad():
        z_list = []
        for iters, batch in enumerate(data_list):
            z, _ = model.cpu()(batch.x, batch.edge_index)
            z_list.append(z.cpu().detach().numpy())
    adata.obsm[key_added] = np.concatenate(z_list, axis=0)    
    
    #z_sublist = []
    #for comb in iter_comb:
        #print(comb)
        #i, j = comb[0], comb[1]
        #z_sublist = [z_list[i],z_list[j]] 
        #adata.obsm[key_added + 'i'] = np.concatenate(z_sublist, axis=0)

    return adata, tran_list