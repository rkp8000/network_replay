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


def w_n_ec_pc_vs_dist(P, C):
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
        v_reset=P.V_RESET_PC, t_r=P.T_R,
        e_ahp=P.E_AHP, t_ahp=np.inf, w_ahp=0,
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
        C.PATH_W_N_EC_PC_VS_DIST,
        {'dist': C.DISTS_PRE, 'w_n_ec_pc': ws_n_pc_ec})
    
    print('w_n_pc_ec vs dists file saved at "{}".'.format(save_path))
    
    # plot results
    fig, ax = plt.subplots(1, 1, figsize=(15, 4), tight_layout=True)
    
    for d_ctr, d in enumerate(C.DISTS_PRE):
        ax.scatter(d*np.ones(n), ws_n_pc_ec[d_ctr], s=5, c='k', lw=0)
        
    ax.set_xlabel('pfc dist from rat\'s location (m)')
    ax.set_ylabel('w_n_ec_pc final')
    
    set_font_size(ax, 16)
    
    return fig
        

def v_g_n_vs_w_n_ec_pc_rate_ec(P, C):
    """
    Compute and save steady-state v and g_n distributions
    as function of w_n_ec_pc and rate_ec.
    """
    np.random.seed(C.SEED_PRE)
    n = len(C.W_N_EC_PC_PRE)
    
    # build PC ntwk
    ws_up = {'NMDA': np.diag(ws_n_pc_ec)}
    ws_rcr = {'NMDA': np.zeros((n, n))}
    
    ntwk = LIFNtwk(
        t_m=P.T_M_PC, e_l=P.E_L_PC, v_th=P.V_TH_PC,
        v_reset=P.V_RESET_PC, t_r=P.T_R,
        e_ahp=0, t_ahp=np.inf, w_ahp=0,
        es_syn={'NMDA': P.E_N}, ts_syn={'NMDA': P.T_N},
        ws_up=ws_up, ws_rcr=ws_rcr)
    
    # loop over EC rates
    ts = np.arange(0, C.DUR_VG_PRE, P.DT)
    
    shape = (
        len(C.W_N_EC_PC_PRE),
        len(C.RATE_EC_PRE),
        len(C.N_TIMEPOINTS_V_G_PRE)
    )
    
    vs = np.nan * np.zeros(shape)
    gs_n = np.nan * np.zeros(shape)
    
    for ctr_r, rate_ec in enumerate(C.RATES_EC_PRE):
        
        # build upstream spks and run ntwk
        spks_up = np.random.poisson(P.DT*rate_ec, (len(ts), n))
        rsp = ntwk.run(spks_up, P.DT)
        
        # loop over neurons i.e. ws_n_pc_ec
        for ctr_w in range(n):
            
            # pick random timepoints to sample vs and gs at
            tps_sample = np.random.choice(
                np.arange(int(C.MEASURE_START_V_G/P.DT), int(DUR/P.DT)), 
                C.N_TIMEPOINTS_V_G_PRE)
            
            vs[ctr_r, ctr_w, :] = rsp.vs[tps_sample, ctr_w]
            gs_n[ctr_r, ctr_w, :] = rsp.gs['NMDA'][tps_sample, ctr_w]
            
    # save results
    save_path = aux.save(
        C.PATH_V_G_N_VS_W_N_EC_PC_RATE_EC,
        {
            'w_n_ec_pc': C.W_N_EC_PC_PRE,
            'rate_ec': C.RATE_EC_PRE,
            'v': vs, 'g_n': gs_n
        })
    
    print('v, g_n vs w_n_ec_pc, rate_ec saved at "{}".'.format(save_path))
    
    # plot results
    w_, rate_ = np.meshgrid(C.W_N_EC_PC_PRE, C.RATE_EC_PRE, indexing='ij')
    
    v_means = vs.mean(-1)
    v_stds = vs.std(-1)
    
    g_n_means = gs_n.mean(-1)
    g_n_stds = gs_n_std(-1)
    
    fig, axs = plt.subplots(1, 4, figsize=(15, 4), tight_layout=True)
    
    # mean v vs w_n_pc_ec, rate_ec
    axs[0].scatter(
        w_.flatten(), rate_.flatten(), c=v_means.flatten(),
        s=5, lw=0, v_min=v_means.min(), v_max=v_means.max())
    axs[0].set_title('Mean PC voltage')
    
    # std v vs w_n_pc_ec, rate_ec
    axs[1].scatter(
        w_.flatten(), rate_.flatten(), c=v_stds.flatten(),
        s=5, lw=0, v_min=v_stds.min(), v_max=v_stds.max())
    axs[1].set_title('Std of PC voltage')
    
    # mean g_n vs w_n_pc_ec, rate_ec
    axs[2].scatter(
        w_.flatten(), rate_.flatten(), c=g_n_means.flatten(),
        s=5, lw=0, v_min=g_n_means.min(), v_max=g_n_means.max())
    axs[2].set_title('Mean EC->PC G_N')
    
    # std g_n vs w_n_pc_ec, rate_ec
    axs[3].scatter(
        w_.flatten(), rate_.flatten(), c=g_n_stds.flatten(),
        s=5, lw=0, v_min=g_n_stds.min(), v_max=g_n_stds.max())
    axs[3].set_title('Std of EC->PC G_N')
    
    for ax in axs:
        ax.set_xlabel('w_n_ec_pc')
        ax.set_ylabel('rate_ec (Hz)')
        
        set_font_size(ax, 16)
        
    return fig


def sample_w_n_ec_pc(dists, w_n_ec_pc_vs_dist):
    """Sample a set of w_n_ec_pc values given distances."""
    
    idxs_dist = aux.idx_closest(dists, w_n_ec_pc_vs_dist['dist'])
    idxs_rand = np.random.randint(
        0, w_n_ec_pc_vs_dist['w_n_ec_pc'].shape[1], len(dists))
    
    return w_n_ec_pc_vs_dist['w_n_ec_pc'][idxs_dist, idxs_rand]


def sample_v_g(ws_n_ec_pc, rates_ec, v_g_n_vs_w_n_ec_pc_rate_ec):
    """
    Sample a set of pc voltages and gs_n given w_n_ec_pc and rate_ec.
    """
    
    idxs_w = aux.idx_closest(
        ws_n_ec_pc, v_g_n_vs_w_n_ec_pc_rate_ec['w_n_ec_pc'])
    idxs_r = aux.idx_closest(
        rates_ec, v_g_n_vs_w_n_ec_pc_rate_ec['rate_ec'])
    idxs_rand = np.random.randint(
        0, v_g_n_vs_w_n_ec_pc_rate_ec['v'].shape[-1], len(ws_n_ec_pc))
    
    vs = v_g_n_vs_w_n_ec_pc_rate_ec['v'][idxs_w, idxs_r, idxs_rand]
    gs_n = v_g_n_vs_w_n_ec_pc_rate_ec['g_n'][idxs_w, idxs_r, idxs_rand]
    
    return vs, gs_n
