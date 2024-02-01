import torch

def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.

    Parameters
    ----------
    pts_src
        [R, D] matrix
    pts_dst
        C, D] matrix
    p
        p-norm
    
    Return
    ------
    [R, C] matrix
        distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance

def unbalanced_ot(tran, mu1, mu2, device, Couple, reg=0.1, reg_m=1.0):
    '''
    Calculate a unbalanced optimal transport matrix between batches.

    Parameters
    ----------
    tran
        transport matrix between the two batches sampling from the global OT matrix. 
    mu1
        mean vector of batch 1 from the encoder
    mu2
        mean vector of batch 2 from the encoder
    reg
        Entropy regularization parameter in OT. Default: 0.1
    reg_m
        Unbalanced OT parameter. Larger values means more balanced OT. Default: 1.0
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        training device

    Returns
    -------
    float
        minibatch unbalanced optimal transport loss
    matrix
        minibatch unbalanced optimal transport matrix
    '''

    ns = mu1.size(0)
    nt = mu2.size(0)

    cost_pp = distance_matrix(mu1, mu2)
    #tran = torch.tensor(pi0, dtype=torch.float).to(device)
    if Couple is not None:
        Couple = torch.tensor(Couple, dtype=torch.float).to(device)
    #cost_pp = ot.dist(mu1, mu2)

    #if query_weight is None: 
    p_s = torch.ones(ns, 1) / ns
    #else:
        #query_batch_weight = query_weight[idx_q]
        #p_s = query_batch_weight/torch.sum(query_batch_weight)

    #if ref_weight is None: 
    p_t = torch.ones(nt, 1) / nt
    #else:
        #ref_batch_weight = ref_weight[idx_r]
        #p_t = ref_batch_weight/torch.sum(ref_batch_weight)

    p_s = p_s.to(device)
    p_t = p_t.to(device)

    if tran is None:
        tran = torch.ones(ns, nt) / (ns * nt)
        #tran = tran_Init
        tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)
    f = reg_m / (reg_m + reg)

    for m in range(10):
        if Couple is not None:
            #print(cost_pp)
            #print(Couple)
            cost = cost_pp*Couple
        else:
            cost = cost_pp

        kernel = torch.exp(-cost / (reg*torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        # dual = p_s / (kernel @ b)
        for i in range(10):
            dual =( p_s / (kernel @ b) )**f
            b = ( p_t / (torch.t(kernel) @ dual) )**f
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    # pho = tran.mean()
    # h_func = 1 - 0.5 * ( 1 + torch.sign(pho - tran) )
    # hat_tran = tran * h_func
    # d_fgw1 = (cost_pp * hat_tran.detach().data).sum()
    # d_fgw2 = ((tran.detach().data - hat_tran.detach().data) * torch.log(1 + torch.exp(-cost_pp))).sum()
    # d_fgw = d_fgw1 + d_fgw2

    d_fgw = (cost * tran.detach().data).sum()

    return d_fgw, tran.detach()


def unbalanced_ot_parameter(tran, mu1, mu2, device, Couple, reg=0.1, reg_m_1 = 1, reg_m_2 = 1):
    '''
    Calculate a unbalanced optimal transport matrix between batches with different reg_m parameters.

    Parameters
    ----------
    tran
        transport matrix between the two batches sampling from the global OT matrix. 
    mu1
        mean vector of batch 1 from the encoder
    mu2
        mean vector of batch 2 from the encoder
    reg
        Entropy regularization parameter in OT. Default: 0.1
    reg_m_1
        Unbalanced OT parameter 1. Larger values means more balanced OT. Default: 1.0
    reg_m_2
        Unbalanced OT parameter 2. Larger values means more balanced OT. Default: 1.0
    Couple
        prior information about weights between cell correspondence. Default: None
    device
        training device

    Returns
    -------
    float
        minibatch unbalanced optimal transport loss
    matrix
        minibatch unbalanced optimal transport matrix
    '''

    ns = mu1.size(0)
    nt = mu2.size(0)

    cost_pp = distance_matrix(mu1, mu2)
    if Couple is not None:
        Couple = torch.tensor(Couple, dtype=torch.float).to(device)
    #cost_pp = ot.dist(mu1, mu2)

    #if query_weight is None: 
    p_s = torch.ones(ns, 1) / ns
    #else:
        #query_batch_weight = query_weight[idx_q]
        #p_s = query_batch_weight/torch.sum(query_batch_weight)

    #if ref_weight is None: 
    p_t = torch.ones(nt, 1) / nt
    #else:
        #ref_batch_weight = ref_weight[idx_r]
        #p_t = ref_batch_weight/torch.sum(ref_batch_weight)

    p_s = p_s.to(device)
    p_t = p_t.to(device)

    if tran is None:
        tran = torch.ones(ns, nt) / (ns * nt)
        #tran = tran_Init
        tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)
    f1 = reg_m_1 / (reg_m_1 + reg)
    f2 = reg_m_2 / (reg_m_2 + reg)

    for m in range(10):
        if Couple is not None:
            cost = cost_pp*Couple
        else:
            cost = cost_pp.to(device)

        kernel = torch.exp(-cost / (reg*torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        # dual = p_s / (kernel @ b)
        for i in range(10):
            dual =( p_s / (kernel @ b) )**f1
            b = ( p_t / (torch.t(kernel) @ dual) )**f2
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    # pho = tran.mean()
    # h_func = 1 - 0.5 * ( 1 + torch.sign(pho - tran) )
    # hat_tran = tran * h_func
    # d_fgw1 = (cost_pp * hat_tran.detach().data).sum()
    # d_fgw2 = ((tran.detach().data - hat_tran.detach().data) * torch.log(1 + torch.exp(-cost_pp))).sum()
    # d_fgw = d_fgw1 + d_fgw2

    d_fgw = (cost * tran.detach().data).sum()

    return d_fgw, tran.detach()