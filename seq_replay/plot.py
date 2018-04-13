"""
Plotting functions for replay smln rslts.
"""
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from disp import set_font_size


def heat_maps(rslt):
    """
    Plot heatmaps showing:
        1. W_E_PC_ST values at start of trial.
        2. W_E_PC_ST values at replay trigger.
        3. # spks per PC within detection wdw.
        4. Firing order of first spikes.
    """
    # W_E_PC_ST
    pcs = rslt.ntwk.plasticity['masks']['E'].nonzero()[0]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

    ## at start
    w_e_pc_st_start = np.array(rslt.ws_plastic['E'][0, pcs]).flatten()
    ## at trigger
    t_idx_trg = int(rslt.schedule['TRG_START_T'] / rslt.s_params['DT'])
    w_e_pc_st_trg = np.array(rslt.ws_plastic['E'][t_idx_trg, pcs]).flatten()

    ## get corresponding place fields
    pfxs_plastic = rslt.ntwk.pfxs[pcs]
    pfys_plastic = rslt.ntwk.pfys[pcs]

    ## make plots
    v_min = rslt.p['W_E_INIT_PC_ST']
    v_max = rslt.p['W_E_INIT_PC_ST'] * rslt.p['A_P']

    ## at start
    im_0 = axs[0].scatter(
        pfxs_plastic, pfys_plastic, c=w_e_pc_st_start,
        s=25, vmin=v_min, vmax=v_max, cmap='hot')

    axs[0].set_title('At trial start')

    ## colorbar nonsense
    divider_0 = make_axes_locatable(axs[0])
    c_ax_0 = divider_0.append_axes('right', '5%', pad=0.05)
    cb_0 = fig.colorbar(
        im_0, cax=c_ax_0, ticks=[v_min, v_max])
    cb_0.set_ticklabels(['{0:.4f}'.format(v_min), '{0:.4f}'.format(v_max)])

    ## at trigger
    im_1 = axs[1].scatter(
        pfxs_plastic, pfys_plastic, c=w_e_pc_st_trg,
        s=25, vmin=v_min, vmax=v_max, cmap='hot')

    ## colorbar nonsense
    divider_1 = make_axes_locatable(axs[1])
    c_ax_1 = divider_1.append_axes('right', '5%', pad=0.05)
    cb_1 = fig.colorbar(
        im_1, cax=c_ax_1, ticks=[v_min, v_max])
    cb_1.set_ticklabels(['{0:.4f}'.format(v_min), '{0:.4f}'.format(v_max)])

    axs[1].set_title('At replay trigger')

    for ax in axs[:2]:
        ax.set_xlabel('PF X (m)')
        ax.set_ylabel('PF Y (m)')
        ax.set_aspect('equal')
        ax.set_facecolor((.7, .7, .7))
        set_font_size(ax, 16)

    for ax in [cb_0.ax, cb_1.ax]:
        ax.set_xlabel('W_E_PC_ST')
        set_font_size(ax, 16)

    figs = [fig]
    axss = [axs]

    # PC spike statistics
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

    ## detection wdw
    start = rslt.schedule['TRG_START_T']
    end = start + rslt.s_params['metrics']['WDW']

    t_mask = (start <= rslt.ts) & (rslt.ts < end)

    ## PC mask and PFs
    pc_mask = rslt.ntwk.types_rcr == 'PC'

    pfxs_pc = rslt.ntwk.pfxs[pc_mask]
    pfys_pc = rslt.ntwk.pfys[pc_mask]

    ## PC spk cts within detection window
    spks_wdw_pc = rslt.spks[t_mask][:, pc_mask]
    spk_ct_wdw_pc = spks_wdw_pc.sum(0)

    ## discrete colormap for showing spk cts
    c_map_tmp = plt.cm.jet
    c_map_list = [c_map_tmp(i) for i in range(c_map_tmp.N)]
    c_map_list[0] = (0., 0., 0., 1.)
    c_map = c_map_tmp.from_list('spk_ct', c_map_list, c_map_tmp.N)

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mpl.colors.BoundaryNorm(bounds, c_map.N)

    im_0 = axs[0].scatter(
        pfxs_pc, pfys_pc, c=spk_ct_wdw_pc, s=25, cmap=c_map, norm=norm)
    divider_0 = make_axes_locatable(axs[0])
    c_ax_0 = divider_0.append_axes('right', size='5%', pad=0.05)

    cb_0 = fig.colorbar(im_0, cax=c_ax_0, ticks=range(6))
    cb_0.set_ticklabels([0, 1, 2, 3, 4, '>4'])
    cb_0.set_label('spike count')

    axs[0].set_aspect('equal')

    axs[0].set_xlabel('PF X (m)')
    axs[0].set_ylabel('PF Y (m)')
    axs[0].set_title('detection wdw spk ct')

    for ax in [axs[0], cb_0.ax]:
        set_font_size(ax, 16)

    # PC spiking order
    if np.any(spk_ct_wdw_pc):

        ## black bkgd for all PCs
        axs[1].scatter(pfxs_pc, pfys_pc, c='k', s=25, zorder=-1)

        ## color PCs according to timing of first spike
        spk_mask = spk_ct_wdw_pc > 0
        spk_order = np.argmax(spks_wdw_pc[:, spk_mask], 0)

        v_min = spk_order.min()
        v_max = spk_order.max()

        im_1 = axs[1].scatter(
            pfxs_pc[spk_mask], pfys_pc[spk_mask], c=spk_order, s=25,
            vmin=v_min, vmax=v_max, cmap='hsv', zorder=0)

        divider_1 = make_axes_locatable(axs[1])
        c_ax_1 = divider_1.append_axes('right', size='5%', pad=0.05)

        cb_1 = fig.colorbar(im_1, cax=c_ax_1, ticks=[v_min, v_max])
        cb_1.set_ticklabels(['first', 'last'])

        axs[1].set_title('first spk order')

        for ax in [axs[1], cb_1.ax]:
            set_font_size(ax, 16)
    else:
        axs[1].set_title('No PC spks')
        set_font_size(axs[1], 16)

    figs.append(fig)
    axss.append(axs)
    
    return figs, axss
        
        
def raster(rslt, xys, nearest, epoch):
    """
    Generate a raster plot of spikes from a smln.
    
    :param xys: list of (x, y) locs to plot spks from nearby cells for
    :param nearest: # of cells per (x, y)
    :param epoch: 'replay', 'wdw', 'trj', or 'full', specifying which epoch
        to make raster for (replay, detection window, trajectory, or full smln)
    """
    # get ordered idxs of PCs to plot
    ## get pfs
    pc_mask = rslt.ntwk.types_rcr == 'PC'
    pfxs = rslt.ntwk.pfxs[pc_mask]
    pfys = rslt.ntwk.pfys[pc_mask]
    
    ## loop through (x, y) pairs and add idxs of nearest PCs
    pc_idxs = get_idxs_nearest(xys, pfxs, pfys, nearest) 
    
    # get all spks for selected PCs
    spks_pc_chosen = rslt.spks[:, pc_idxs]
    
    # get desired time window
    if epoch == 'replay':
        start = rslt.schedule['REPLAY_EPOCH_START_T']
        end = rslt.schedule['SMLN_DUR']
    elif epoch == 'wdw':
        start = rslt.schedule['TRG_START_T']
        end = start + rslt.s_params['metrics']['WDW']
    elif epoch == 'trj':
        start = rslt.schedule['TRJ_START_T']
        end = rslt.schedule['REPLAY_EPOCH_START_T']
    elif epoch == 'full':
        start = 0
        end = rslt.schedule['SMLN_DUR']
    
    t_mask = (start <= rslt.ts) & (rslt.ts < end)
    t_start = rslt.ts[t_mask][0]
    
    spk_t_idxs, pcs = spks_pc_chosen[t_mask].nonzero()
    spk_ts = spk_t_idxs * rslt.s_params['DT'] + t_start
    
    # make plots
    fig = plt.figure(figsize=(16, 4), tight_layout=True)
    gs = gridspec.GridSpec(1, 4)
    
    ## spks
    ax_0 = fig.add_subplot(gs[:3])
    # ax_0.scatter(spk_ts, pcs, c='k', s=10, marker='|', lw=0.5)
    
    ax_0.set_xlabel('t (s)')
    ax_0.set_ylabel('PC idx')
    ax_0.set_title('Raster plot for selected cells')
    
    ## cell PF locations
    ax_1 = fig.add_subplot(gs[3])
    ax_1.scatter(pfxs, pfys, c='k', s=25, lw=0)
    
    ax_1.scatter(
        pfxs[pc_idxs], pfys[pc_idxs], c=np.linspace(0, 1, len(pc_idxs)),
        s=25, lw=0, vmin=0, vmax=1, cmap='spring')
    
    ax_1.set_xlabel('x (m)')
    ax_1.set_ylabel('y (m)')
    ax_1.set_title('PC PF locations')
    
    for ax in [ax_0, ax_1]:
        set_font_size(ax, 20)
        
    return fig, [ax_0, ax_1]
        
        
def get_idxs_nearest(xys, pfxs, pfys, nearest):
    """
    Get ordered idxs of place fields nearest to a
    sequence of (x, y) points.
    """
    idxs = []
    
    for xy in xys:
        # get dists of all PFs to (x, y)
        dx = pfxs - xy[0]
        dy = pfys - xy[1]
        d = np.sqrt(dx**2 + dy**2)
        
        # add idxs of closest neurons to list
        idxs.extend(list(d.argsort()[:nearest]))
        
    return idxs
    