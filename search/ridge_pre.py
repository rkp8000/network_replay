"""
Code for precomputing:
    w_n_ec_pc vs. distance distributions
    v_0, g_n_0 vs. w_n_ec_pc distributions
"""
import matplotlib.pyplot as plt
import numpy as np

import aux
from ntwk import LIFNtwk
from plot import set_font_size
from traj import Traj, spks_up_from_traj

cc = np.concatenate


def w_n_pc_ec_vs_dist(C, P):
    """
    Compute and save w_n_ec_pc distributions as
    function of distance from place-field center.
    """
    np.random.seed(C.SEED_PRE)
    n = len(C.DIST_PRE)
    
    # build place fields
    pfcs = np.array([C.DIST_PRE, np.zeros(n)])
    pfws = P.L_PL * np.ones(n)
    max_rates = P.R_MAX_PL * np.ones(n)
    
    # get place-tuned inputs
    ts = np.arange(0, C.DUR_W_PRE, P.DT)
    xys = np.zeros((len(ts), 2))
    traj = Traj(ts=ts, xys=xys)
       
    # build ntwk
    ws_up = {
        'AMPA': cc([P.W_A_PC_PL * np.eye(n), np.zeros((n, n))], axis=1),
        'NMDA': cc([np.zeros((n, n)), P.W_N_PC_EC_I * np.eye(n)], axis=1),
    }
    
    ws_rcr = {'AMPA': np.zeros((n, n)), 'NMDA': np.zeros((n, n))}
    
    masks_plastic = {
        'AMPA': np.zeros(ws_up['AMPA'].shape, dtype=bool),
        'NMDA': cc([np.zeros((n, n)), np.eye(n)], axis=1).astype(bool),
    }
    
    ntwk = LIFNtwk(
        t_m=P.T_M_PC, e_l=P.E_L_PC, v_th=P.V_TH_PC,
        v_reset=P.V_RESET_PC, t_r=np.repeat(P.T_R_PC, n),
        e_ahp=P.E_AHP_PC, t_ahp=np.inf, w_ahp=0,
        es_syn={'AMPA': P.E_A, 'NMDA': P.E_N},
        ts_syn={'AMPA': P.T_A, 'NMDA': P.T_N},
        ws_up=ws_up, ws_rcr=ws_rcr,
        plasticity={
            'masks': masks_plastic,
            'w_ec_ca3_maxs': {'AMPA': np.nan, 'NMDA': P.W_N_PC_EC_F},
            'T_W': P.T_W, 'T_C': P.T_C, 'C_S': P.C_S, 'BETA_C': P.B_C,
        })
   
    # run ntwk and store final w_n_pc_ec
    ws_n_pc_ec = np.nan * np.zeros((C.N_TRIALS_W_PRE, n))
    
    for ctr in range(C.N_TRIALS_W_PRE):
        
        spks_up_pl = spks_up_from_traj(ts, traj.xys, pfcs, pfws, max_rates)
        
        # set all upstream spikes outside stim itvl to 0
        mask = ~((ts >= C.STIM_W_PRE[0]) & (ts < C.STIM_W_PRE[1]))
        spks_up_pl[mask] = 0
        
        spks_up = np.zeros((len(ts), 2*n))
        spks_up[:, :n] = spks_up_pl
        
        rsp = ntwk.run(spks_up, dt=P.DT)
        ws_n_pc_ec[ctr, :] = rsp.ws_plastic['NMDA'][-1]
    
    # save results
    save_path = aux.save(
        C.PATH_W_N_PC_EC_VS_DIST,
        {'dist': C.DIST_PRE, 'w_n_pc_ec': ws_n_pc_ec})
    
    print('w_n_pc_ec vs dists file saved at "{}".'.format(save_path))
    
    # plot results
    fig, ax = plt.subplots(1, 1, figsize=(15, 4), tight_layout=True)
    
    for ctr in range(C.N_TRIALS_W_PRE):
        ax.scatter(C.DIST_PRE, ws_n_pc_ec[ctr], s=5, c='k', lw=0)
        
    y_min = ws_n_pc_ec.min()
    y_max = ws_n_pc_ec.max()
    y_range = y_max - y_min
    
    ax.set_ylim(y_min - .1*y_range, y_max + .1*y_range)
        
    ax.set_xlabel('pfc dist from rat\'s location (m)')
    ax.set_ylabel('w_n_pc_ec final')
    
    set_font_size(ax, 16)
    
    return fig
        

def v_g_n_vs_w_n_pc_ec_fr_ec(C, P, cmap='hot'):
    """
    Compute and save steady-state v and g_n distributions
    as function of w_n_pc_ec and fr_ec.
    """
    np.random.seed(C.SEED_PRE)
    n = len(C.W_N_PC_EC_PRE)
    
    # build PC ntwk
    ws_up = {'NMDA': np.diag(C.W_N_PC_EC_PRE)}
    ws_rcr = {'NMDA': np.zeros((n, n))}
    
    ntwk = LIFNtwk(
        t_m=P.T_M_PC, e_l=P.E_L_PC, v_th=P.V_TH_PC,
        v_reset=P.V_RESET_PC, t_r=np.repeat(P.T_R_PC, n),
        e_ahp=0, t_ahp=np.inf, w_ahp=0,
        es_syn={'NMDA': P.E_N}, ts_syn={'NMDA': P.T_N},
        ws_up=ws_up, ws_rcr=ws_rcr)
    
    # loop over EC rates
    ts = np.arange(0, C.DUR_V_G_PRE, P.DT)
    
    shape = (
        len(C.W_N_PC_EC_PRE),
        len(C.FR_EC_PRE),
        C.N_TIMEPOINTS_V_G_PRE,
    )
    
    vs = np.nan * np.zeros(shape)
    gs_n = np.nan * np.zeros(shape)
    
    for ctr_r, fr_ec in enumerate(C.FR_EC_PRE):
        
        # build upstream spks and run ntwk
        spks_up = np.random.poisson(P.DT * fr_ec, (len(ts), n))
        rsp = ntwk.run(spks_up, P.DT)
        
        # loop over neurons i.e. ws_n_pc_ec
        for ctr_w in range(n):
            
            # pick random timepoints to sample vs and gs at
            start = int(C.MEASURE_START_V_G_PRE / P.DT)
            end = int(C.DUR_V_G_PRE / P.DT) 
            
            tps_sample = np.random.choice(
                np.arange(start, end), C.N_TIMEPOINTS_V_G_PRE)
            
            vs[ctr_w, ctr_r, :] = rsp.vs[tps_sample, ctr_w]
            gs_n[ctr_w, ctr_r, :] = rsp.gs['NMDA'][tps_sample, ctr_w]
            
    # save results
    save_path = aux.save(
        C.PATH_V_G_N_VS_W_N_PC_EC_FR_EC,
        {
            'w_n_pc_ec': C.W_N_PC_EC_PRE,
            'fr_ec': C.FR_EC_PRE,
            'v': vs, 'g_n': gs_n
        })
    
    print('v, g_n vs w_n_pc_ec, fr_ec saved at "{}".'.format(save_path))
    
    # plot results
    w_, rate_ = np.meshgrid(C.W_N_PC_EC_PRE, C.FR_EC_PRE, indexing='ij')
    
    v_means = vs.mean(-1)
    v_stds = vs.std(-1)
    
    g_n_means = gs_n.mean(-1)
    g_n_stds = gs_n.std(-1)
    
    fig, axs = plt.subplots(1, 4, figsize=(15, 4), tight_layout=True)
    
    # mean v vs w_n_pc_ec, fr_ec
    axs[0].scatter(
        w_.flatten(), rate_.flatten(), c=v_means.flatten(),
        s=10, lw=0, vmin=v_means.min(), vmax=v_means.max(), cmap=cmap)
    axs[0].set_title('Mean PC voltage')
    
    # std v vs w_n_pc_ec, fr_ec
    axs[1].scatter(
        w_.flatten(), rate_.flatten(), c=v_stds.flatten(),
        s=10, lw=0, vmin=v_stds.min(), vmax=v_stds.max(), cmap=cmap)
    axs[1].set_title('Std of PC voltage')
    
    # mean g_n vs w_n_pc_ec, fr_ec
    axs[2].scatter(
        w_.flatten(), rate_.flatten(), c=g_n_means.flatten(),
        s=10, lw=0, vmin=g_n_means.min(), vmax=g_n_means.max(), cmap=cmap)
    axs[2].set_title('Mean EC->PC G_N')
    
    # std g_n vs w_n_pc_ec, fr_ec
    axs[3].scatter(
        w_.flatten(), rate_.flatten(), c=g_n_stds.flatten(),
        s=10, lw=0, vmin=g_n_stds.min(), vmax=g_n_stds.max(), cmap=cmap)
    axs[3].set_title('Std of EC->PC G_N')
    
    w_min = w_.min()
    w_max = w_.max()
    w_range = w_max - w_min
    
    for ax in axs:
        ax.set_xlim(w_min - .1*w_range, w_max + .1*w_range)
        
        ax.set_xticks([w_min, w_max])
        ax.set_xlabel('w_n_pc_ec')
        ax.set_ylabel('fr_ec (Hz)')
        
        set_font_size(ax, 16)
        
    return fig


def sample_w_n_pc_ec(dists, pre):
    """Sample a set of w_n_pc_ec values given distances."""
    w_n_pc_ec_vs_dist = pre['w_n_pc_ec_vs_dist']
    
    idxs_dist = aux.idx_closest(dists, w_n_pc_ec_vs_dist['dist'])
    idxs_rand = np.random.randint(
        0, w_n_pc_ec_vs_dist['w_n_pc_ec'].shape[1], len(dists))
    
    return w_n_pc_ec_vs_dist['w_n_pc_ec'][idxs_dist, idxs_rand]


def sample_v_g(ntwk, p, pre):
    """
    Sample a set of pc voltages and gs_n given w_n_ec_pc and fr_ec.
    """
    ws_n_pc_ec = ntwk.ws_up_init['NMDA'].diagonal()
    v_g_n_vs_w_n_pc_ec_fr_ec = pre['v_g_n_vs_w_n_pc_ec_fr_ec']
    
    idxs_w = aux.idx_closest(
        ws_n_pc_ec, v_g_n_vs_w_n_pc_ec_fr_ec['w_n_pc_ec'])
    
    idxs_r = aux.idx_closest(
        p['FR_EC'], v_g_n_vs_w_n_pc_ec_fr_ec['fr_ec'])
    
    idxs_rand = np.random.randint(
        0, v_g_n_vs_w_n_pc_ec_fr_ec['v'].shape[-1], len(ws_n_pc_ec))
    
    vs = v_g_n_vs_w_n_pc_ec_fr_ec['v'][idxs_w, idxs_r, idxs_rand]
    gs_n = v_g_n_vs_w_n_pc_ec_fr_ec['g_n'][idxs_w, idxs_r, idxs_rand]
    
    return vs, gs_n
