"""
Code for running full linear ridge simulations.
"""
import numpy as np
import os

import aux
from db import make_session, d_models
from ntwk import LIFNtwk
from search import trial_to_p, trial_to_stable_ntwk


def run_smln(trial_id, d_model, pre, C, P, save, cache_file=None, return_p=False):
    """
    Run a full simulation starting from either a replay-only or full
    trial.
    
    :param d_model: data model specifying which trial type/table
    :param save: whether to save response in database
    :param cache_file: temporary file containing recurrent ntwk cxn 
        matrices and replay triggering stimulus; if no cache_file is
        specified, look for the default one in the directory C.CACHE_DIR.
    """
    np.random.seed(0)
    
    # load replay-only trial
    session = make_session()
    trial = session.query(d_model).get(trial_id)

    if trial.__class__.__name__ == 'LinRidgeFullTrial':
        # get replay-only trial
        trial = trial.LinRidgeTrial

    session.close()
    
    # get trial params
    p = trial_to_p(trial)

    # construct default cache file path if not provided
    if cache_file is None:
        base_name = 'ws_rcr_replay_trigger_{}.npy'.format(trial_id)
        cache_file = os.path.join(C.CACHE_DIR, base_name)
    
    # make new cache file if it doesn't exist
    if not os.path.exists(cache_file):
        print('No cached file found at {}.'.format(cache_file))
        print('Running replay-only smln and computing replay trigger...')
        
        # otherwise run replay-only smln to extract ws_rcr:
        
        # run replay-only trial and get first stable ntwk
        ntwk, vs_0, gs_0, spks_forced = trial_to_stable_ntwk(trial, pre, C, P)
        
        # get recurrent cxns and pfcs
        ws_rcr = ntwk.ws_rcr
        pfcs = ntwk.pfcs
        
        # compute replay trigger
        trigger = make_replay_trigger(
            ntwk, p, vs_0, gs_0, spks_forced)
        
        # save cache file
        aux.save(cache_file, {'ws_rcr': ws_rcr, 'pfcs': pfcs, 'trigger': trigger})
        print('Cache file saved at {}.'.format(cache_file))
    
    # load cache file
    cached = aux.load(cache_file)

    ws_rcr = cached['ws_rcr']
    pfcs = cached['pfcs']
    trigger = cached['trigger']
    
    pc_mask = np.all(~np.isnan(pfcs), axis=0)
    pfcs_pc = pfcs[:, pc_mask]
    
    n_pc = pc_mask.sum()

    # run full smln
    print('Running full smln...')
    
    ## build ntwk
    ntwk = p_to_ntwk_plastic(p, P, ws_rcr, pfcs)
    
    ## build stim sequence
    t_final = C.T_REPLAY + (C.N_REPLAY * C.ITVL_REPLAY)
    
    ts = np.arange(0, t_final, P.DT)
    spks_up = np.zeros((len(ts), 2*n_pc))
    
    ### build linear trajectory-driven PL inputs
    t_mask_traj = (C.TRAJ_START_T <= ts) & (ts < C.TRAJ_END_T)
    ts_traj = ts[t_mask_traj]
    xys = np.array([
        np.linspace(C.TRAJ_START_X, C.TRAJ_END_X, len(ts_traj)),
        np.linspace(C.TRAJ_START_Y, C.TRAJ_END_Y, len(ts_traj)),
    ]).T
    
    pfws = P.L_PL * np.ones(pfcs_pc.shape[1])
    max_rates = P.R_MAX_PL * np.ones(pfcs_pc.shape[1])
    
    spks_up_pl = spks_up_from_traj(ts_traj, xys, pfcs_pc, pfws, max_rates)
    spks_up[t_mask_traj, :n_pc] = spks_up_pl
    
    ### build EC inputs
    t_mask_ec = (ts >= C.T_EC)
    spks_up_ec = np.random.poisson(p['FR_EC']*P.DT, (t_mask.sum(), n_pc))
    
    spks_up[t_mask_ec, -n_pc:] = spks_up_ec
    
    ### build replay-triggering inputs
    for ctr in range(C.N_REPLAY):
        t_trigger = C.T_REPLAY + ctr*C.ITVL_REPLAY
        t_idx_start = int(t_trigger/P.DT)
        t_idx_end = t_idx_start + trigger.shape[0]
        
        spks_up[t_idx_start:t_idx_end, :n_pc] = trigger
        
    # run ntwk
    rsp = ntwk.run(spks_up, dt=P.DT)
    rsp.pfcs = pfcs
    rsp.cell_types = cell_types
    
    if return_p:
        return rsp, p
    else:
        return rsp


def p_to_ntwk_plastic(p, P, ws_rcr, pfcs):
    """
    Create a plastic ntwk from a set of params.
    
    :param p: dict of ntwk params
    :param ws_rcr: recurrent weight matrices (optional)
    :param pfcs: place fields of PCs (nans for INHs)
    """
    cc = np.concatenate
    
    # ensure consistent cell counts
    n = pfcs.shape[1]
    
    for w_rcr in ws_rcr.values():
        assert w_rcr.shape[0] == w_rcr.shape[1] == n
    
    pc_mask = np.all(~np.isnan(pfcs), axis=0)
    inh_mask = ~pc_mask
    
    n_pc = pc_mask.sum()
    n_inh = inh_mask.sum()
    
    cell_types = np.repeat('', n)
    cell_types[pc_mask] = 'PC'
    cell_types[inh_mask] = 'INH'
    
    # build ntwk
    
    ## single-unit params
    t_m = cc([np.repeat(P.T_M_PC, n_pc), np.repeat(P.T_M_INH, n_inh)])
    e_l = cc([np.repeat(P.E_L_PC, n_pc), np.repeat(P.E_L_INH, n_inh)])
    v_th = cc([np.repeat(P.V_TH_PC, n_pc), np.repeat(P.V_TH_INH, n_inh)])
    v_reset = cc([np.repeat(P.V_RESET_PC, n_pc), np.repeat(P.V_RESET_INH, n_inh)])
    t_r = cc([np.repeat(P.T_R_PC, n_pc), np.repeat(P.T_R_INH, n_inh)])
    
    ## upstream cxns
    
    ### AMPA inputs from PL
    w_up_a = np.zeros((n, 2*n_pc), dtype=float)
    w_up_a[:n_pc, :n_pc] = P.W_A_PC_PL * np.eye(n_pc)
    
    ### NMDA inputs from EC
    w_up_n = np.zeros((n, 2*n_pc), dtype=float)
    w_up_n[:n_pc, -n_pc:] = P.W_N_PC_EC_I * np.eye(n_pc)
    
    ### no GABA inputs
    w_up_g = np.zeros((n, 2*n_pc), dtype=float)
    
    ws_up = {'AMPA': w_up_a, 'NMDA': w_up_n, 'GABA': w_up_g}
    
    ## plasticity params
    plasticity={
        'masks': {
            'AMPA': np.zeros((n, 2*n_pc), dtype=bool),
            'NMDA': w_up_n > 0,
            'GABA': np.zeros((n, 2*n_pc), dtype=bool),
        },
        'w_ec_ca3_maxs': {
            'AMPA': np.nan,
            'NMDA': P.W_N_PC_EC_F,
            'GABA': np.nan,
        },
        'T_W': P.T_W, 'T_C': P.T_C, 'C_S': P.C_S, 'BETA_C': P.B_C
    }
    
    # final ntwk
    ntwk = LIFNtwk(
        t_m=t_m, e_l=e_l, v_th=v_th, v_reset=v_reset, t_r=t_r,
        e_ahp=P.E_AHP_PC, t_ahp=np.inf, w_ahp=0,
        es_syn={'AMPA': P.E_A, 'NMDA': P.E_N, 'GABA': P.E_G},
        ts_syn={'AMPA': P.T_A, 'NMDA': P.T_N, 'GABA': P.T_G},
        ws_up=ws_up, ws_rcr=ws_rcr, plasticity=plasticity)
    
    ntwk.pfcs = pfcs
    ntwk.cell_types = cell_types
    
    return ntwk


def make_replay_trigger():
    """
    takes replay-only ntwk, corresponding params and initial
    conditions, forced_spk PC idxs, builds EC stim, and 
    determines an input to the forced PCs sufficient to 
    elicit replay under EC activation
    """
    pass
