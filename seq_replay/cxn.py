"""
Functions for creating structured synaptic weight matrices
for sequence replay simulations.
"""
import numpy as np

from aux import lognormal_mu_sig


def make_w_e_pc_pc(pfxs, pfys, p):
    """
    Make proximally biased PC-PC weight matrix.
    """
    # make cxns
    n_pc = p['N_PC']
    
    ## build distance matrix
    dx = np.tile(pfxs[None, :], (n_pc, 1)) - np.tile(pfxs[:, None], (1, n_pc))
    dy = np.tile(pfys[None, :], (n_pc, 1)) - np.tile(pfys[:, None], (1, n_pc))
    d = np.sqrt(dx**2 + dy**2)
    
    ## build cxn probability matrix
    prb = np.clip(p['Z_PC_PC'] * np.exp(-d/p['L_PC_PC']), 0, 1)
    
    ## set diagonals to zero
    prb[np.eye(n_pc, dtype=bool)] = 0
    
    ## build cxn matrix
    c = np.random.rand(n, n) < prb
    
    # assign weights
    w = np.zeros(c.shape)
    w[c] = np.random.lognormal(
        *lognormal_mu_sig(p['W_E_PC_PC'], p['S_E_PC_PC']), c.sum())
    
    if np.any(np.isnan(w)):
        raise ValueError('NaNs detected in weight matrix.')
    
    return w


def make_w_e_inh_pc(pfxs_inh, pfys_inh, pfxs_pc, pfys_pc, p):
    """
    Make proximally biased PC->INH weight matrix.
    """
    # make cxns
    n_inh = p['N_INH']
    n_pc = p['N_PC']
    
    ## build distance matrix
    dx = np.tile(pfxs_pc[None, :], (n_inh, 1)) \
        - np.tile(pfxs_inh[:, None], (1, n_pc))
    dy = np.tile(pfys_pc[None, :], (n_inh, 1)) \
        - np.tile(pfys_inh[:, None], (1, n_pc))
    d = np.sqrt(dx**2 + dy**2)
    
    ## build cxn probability matrix
    prb = np.clip(p['Z_INH_PC'] * np.exp(-d/p['L_INH_PC']), 0, 1)
    
    ## build cxn matrix
    c = np.random.rand(n_inh, n_pc) < prb
    
    # assign weights
    w = np.zeros(c.shape)
    w[c] = np.random.lognormal(
        *lognormal_mu_sig(p['W_E_INH_PC'], p['S_E_INH_PC']), c.sum())
    
    if np.any(np.isnan(w)):
        raise ValueError('NaNs detected in weight matrix.')
    
    return w
    
    
def make_w_i_pc_inh(pfxs_pc, pfys_pc, pfxs_inh, pfys_inh, p):
    """
    Make center-surround structured INH->PC weight matrix.
    """
    # make cxns
    n_pc = p['N_PC']
    n_inh = p['N_INH']
    
    ## build distance matrix
    dx = np.tile(pfxs_inh[None, :], (n_pc, 1)) \
        - np.tile(pfxs_pc[:, None], (1, n_inh))
    dy = np.tile(pfys_inh[None, :], (n_pc, 1)) \
        - np.tile(pfys_pc[:, None], (1, n_inh))
    d = np.sqrt(dx**2 + dy**2)
    
    ## build cxn probability matrix
    prb_unclipped = p['Z_S_PC_INH'] * np.exp(-d/p['L_S_PC_INH']) \
        - p['Z_C_PC_INH'] * np.exp(-d/p['Z_C_PC_INH'])
    prb = np.clip(prb_unclipped, 0, 1)
    
    ## build cxn matrix
    c = np.random.rand(n_pc, n_inh) < prb
    
    # assign weights
    w = np.zeros(c.shape)
    w[c] = np.random.lognormal(
        *lognormal_mu_sig(p['W_I_PC_INH'], p['S_I_PC_INH']), c.sum())
    
    if np.any(np.isnan(w)):
        raise ValueError('NaNs detected in weight matrix.')
    
    return w


def join_w(targs, srcs, ws):
    """
    Combine multiple weight matrices specific to pairs of populations
    into a single, full set of weight matrices (one per synapse type).
    
    :param targs: dict of boolean masks indicating targ cell classes
    :param srcs: dict of boolean masks indicating source cell classes
    :param ws: dict of inter-population weight matrices, e.g.:
        ws = {
            'AMPA': {
                ('EXC', 'EXC'): np.array([[...]]),
                ('INH', 'EXC'): np.array([[...]]),
            },
            'GABA': {
                ('EXC', 'INH'): np.array([[...]]),
                ('INH', 'INH'): np.array([[...]]),
            }
        }
        note: keys given as (targ, src)
    
    :return: ws_full, a dict of full ws, one per synapse
    """
    # make sure all targ/src masks have same shape
    targ_shapes = [mask.shape for mask in targs.values()]
    src_shapes = [mask.shape for mask in srcs.values()]
    
    if len(set(targ_shapes)) > 1:
        raise Exception('All targ masks must have same shape.')
        
    if len(set(src_shapes)) > 1:
        raise Exception('All targ masks must have same shape.')
        
    n_targ = targ_shapes[0][0]
    n_src = src_shapes[0][0]
    
    # make sure weight matrix dimensions match sizes
    # of targ/src classes
    for syn, ws_ in ws.items():
        for (targ, src), w_ in ws_.items():
            if not w_.shape == (targs[targ].sum(), srcs[src].sum()):
                raise Exception(
                    'Weight matrix for {}: ({}, {}) does not match '
                    'dimensionality specified by targ/src masks.')
        
    # loop through synapse types
    dtype = list(list(ws.values())[0].values())[0].dtype
    ws_full = {}
    
    for syn, ws_ in ws.items():
        
        w = np.zeros((n_targ, n_src), dtype=dtype)
        
        # loop through population pairs
        for (targ, src), w_ in ws_.items():
            
            # get mask of all cxns from src to targ
            mask = np.outer(targs[targ], srcs[src])
            
            assert mask.sum() == w_.size
            
            w[mask] = w_.flatten()
            
        ws_full[syn] = w
        
    return ws_full
