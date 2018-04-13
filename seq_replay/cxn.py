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
    np.fill_diagonal(prb, False)
    
    ## build cxn matrix from cxn prb
    c = np.random.rand(n_pc, n_pc) < prb
    
    # assign weights
    w = np.zeros((n_pc, n_pc))
    w[c] = np.random.lognormal(
        *lognormal_mu_sig(p['W_E_PC_PC'], p['S_E_PC_PC']), c.sum())
    
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
    
    return w
