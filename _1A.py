"""Functions called by 1A.ipynb"""
from copy import deepcopy
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats

from aux import Generic, GenericFlexible
from ntwk import LIFNtwk
from disp import raster, set_font_size


# Ntwks with specific graph structures

## Unconnected

def unc_ntwk(P, STORE=None):
    """Make a new LIF ntwk with no recurrent connectivity."""
    # make upstream weight matrix
    ws_up = {'E': P.w_e_up * np.eye(P.n)}
    
    c_rcr = np.zeros((P.n, P.n), dtype=bool)
    
    # remake graph
    g = nx.from_numpy_matrix(c_rcr.T, create_using=nx.DiGraph())
    
    # set weights
    ws_rcr = {'E': 0. * c_rcr}
    
    # make ntwk
    ntwk = LIFNtwk(
        t_m=np.repeat(P.t_m, P.n),
        e_l=np.repeat(P.e_l, P.n),
        v_th=np.repeat(P.v_th, P.n),
        v_reset=np.repeat(P.v_reset, P.n),
        t_r=np.repeat(P.t_r, P.n),
        es_syn={'E': P.e_e},
        ts_syn={'E': P.t_e},
        ws_rcr=ws_rcr,
        ws_up=ws_up)
    
    STORE.g = g
    STORE.c_rcr = c_rcr.copy()
    STORE.ws_rcr = ws_rcr.copy()
    
    return ntwk, STORE


## Scale-free

def dctd_p_law_adj(n, gamma, seed):
    """
    Construct an adjacency matrix with equal but uncorrelated power-law
    in- and out-degree distributions.
    """
    deg_possible = np.arange(n, dtype=float)
    
    # get normalized power-law probability over deg_possible
    p_unnormed = (deg_possible + 1) ** (-gamma)
    p_normed = p_unnormed / np.sum(p_unnormed)

    # sample in-deg from power-law distribution
    np.random.seed(seed)
    in_deg = np.random.choice(deg_possible.astype(int), n, replace=True, p=p_normed)

    # shuffle out-deg to break corr. with in-deg
    out_deg = in_deg[np.random.permutation(n)]

    g = nx.directed_configuration_model(in_deg, out_deg, seed=seed)

    adj = np.array(nx.adjacency_matrix(g).todense())

    n_orig = np.sum(adj)

    # randomly reassign self and parallel edges
    
    ## self
    n_self = np.sum(adj.diagonal())
    np.fill_diagonal(adj, 0)
    
    ## parallel
    n_prll = np.sum(adj) - np.sum(adj > 0)
    adj[adj > 0] = 1
    
    n_reassign = n_self + n_prll

    ## get mask over available positions
    mask_avail = (adj == 0)
    np.fill_diagonal(mask_avail, False)

    if mask_avail.sum() < n_reassign:
        raise Exception('Too many edges ({}) for {} x {} DiGraph.'.format(n_orig, n, n))

    ## convert from boolean to numeric idxing
    idxs_avail = np.array(np.nonzero(mask_avail)).T
    
    ## select random subset of available idxs
    idxs_assign = idxs_avail[np.random.permutation(len(idxs_avail))[:n_reassign]]

    ## assign 1s at chosen idxs
    adj[idxs_assign[:, 0], idxs_assign[:, 1]] = 1
    
    return adj.astype(bool)


def dctd_p_law_mean_q(n, gam):
    """
    Calculate mean edge density for a directed power-law graph
    with n nodes and exponent gamma created using dctd_p_law_adj.
    
    Eqn:
    E[Q|gam,n] = (1/n) * (sum_0^{n-1} k* (k+1)^{-gam}) / (sum_0^{n-1} (k+1)^{-gam})
    """
    k = np.arange(n, dtype=float)
    
    num = np.sum(k * ((k + 1)**(-gam)))
    denom = np.sum((k + 1)**(-gam))
    
    return (1/n) * (num/denom)


def sf_ntwk(P, STORE=None):
    """Make a new LIF ntwk with scale-free connectivity."""
    # make upstream weight matrix
    ws_up = {'E': P.w_e_up * np.eye(P.n)}
    
    c_rcr = dctd_p_law_adj(P.n, P.gamma, P.seed)
    
    # remake graph
    g = nx.from_numpy_matrix(c_rcr.T, create_using=nx.DiGraph())
    
    # set weights
    ws_rcr = {'E': P.w_e_rcr * c_rcr}
    
    # make ntwk
    ntwk = LIFNtwk(
        t_m=np.repeat(P.t_m, P.n),
        e_l=np.repeat(P.e_l, P.n),
        v_th=np.repeat(P.v_th, P.n),
        v_reset=np.repeat(P.v_reset, P.n),
        t_r=np.repeat(P.t_r, P.n),
        es_syn={'E': P.e_e},
        ts_syn={'E': P.t_e},
        ws_rcr=ws_rcr,
        ws_up=ws_up)
    
    STORE.g = g
    STORE.c_rcr = c_rcr.copy()
    STORE.ws_rcr = ws_rcr.copy()
    
    return ntwk, STORE


## Small-world

def sw_adj(n, k, p, seed):
    """
    Construct an adjacency matrix for a small-world network.
    """
    g = nx.watts_strogatz_graph(n, k, p, seed)
    adj = np.array(nx.adjacency_matrix(g, weight=None).T.todense())
    
    return adj.astype(bool)


def sw_ntwk(P, STORE=None):
    """Make a new LIF ntwk with small-world connectivity."""
    
    ws_up = {'E': P.w_e_up * np.eye(P.n)}
    
    c_rcr = sw_adj(P.n, P.k, P.p, P.seed)
    
    # remake graph
    g = nx.from_numpy_matrix(c_rcr.T, create_using=nx.DiGraph())
    
    # set weights
    ws_rcr = {'E': P.w_e_rcr * c_rcr}
    
    # make ntwk
    ntwk = LIFNtwk(
        t_m=np.repeat(P.t_m, P.n),
        e_l=np.repeat(P.e_l, P.n),
        v_th=np.repeat(P.v_th, P.n),
        v_reset=np.repeat(P.v_reset, P.n),
        t_r=np.repeat(P.t_r, P.n),
        es_syn={'E': P.e_e},
        ts_syn={'E': P.t_e},
        ws_rcr=ws_rcr,
        ws_up=ws_up)
    
    STORE.g = g
    STORE.c_rcr = c_rcr.copy()
    STORE.ws_rcr = ws_rcr.copy()
    
    return ntwk, STORE


## Erdos-Renyi

def er_ntwk(P, STORE=None):
    """Make a new LIF ntwk with Erdos-Renyi connectivity."""
    # make upstream weight matrix
    ws_up = {'E': P.w_e_up * np.eye(P.n)}
    
    c_rcr = np.random.rand(P.n, P.n) < P.q
    
    # remake graph
    g = nx.from_numpy_matrix(c_rcr.T, create_using=nx.DiGraph())
    
    # set weights
    ws_rcr = {'E': P.w_e_rcr * c_rcr}
    
    # make ntwk
    ntwk = LIFNtwk(
        t_m=np.repeat(P.t_m, P.n),
        e_l=np.repeat(P.e_l, P.n),
        v_th=np.repeat(P.v_th, P.n),
        v_reset=np.repeat(P.v_reset, P.n),
        t_r=np.repeat(P.t_r, P.n),
        es_syn={'E': P.e_e},
        ts_syn={'E': P.t_e},
        ws_rcr=ws_rcr,
        ws_up=ws_up)
    
    STORE.g = g
    STORE.c_rcr = c_rcr.copy()
    STORE.ws_rcr = ws_rcr.copy()
    
    return ntwk, STORE


# Function to drive ntwk with random input spks

def run(ntwk, P, STORE=None):
    """Construct SF ntwk and run smln."""
    
    # make noisy input
    np.random.seed(P.seed)
    t = np.arange(0, P.dur, P.dt)
    spks_up = np.random.poisson(P.frq_up * P.dt, (len(t), P.n))
    spks_up[t >= P.stm_off] = 0
    
    # run ntwk
    rsp = ntwk.run(spks_up, P.dt)
    
    STORE.spks_up = spks_up.copy()
    
    return rsp, STORE


# Function to run ntwk and make basic plots
def rfcd(make_ntwk, P, STORE, STORE_UNC=None, plot=True):
    """
    Run ntwk and optionally show raster plot,
    firing rate distr, correlation distr, and degree distributions.
    """
    # make ntwk and run smln
    ntwk = make_ntwk(P, STORE)[0]
    rsp = run(ntwk, P, STORE)[0]

    # make plots
    if plot:
        gs = gridspec.GridSpec(4, 2)
        fig = plt.figure(figsize=(7.5, 10), tight_layout=True)
        axs = []

    # raster
    if plot:
        axs.append(fig.add_subplot(gs[0, :]))
        raster(axs[-1], rsp.ts, rsp.spks)

    # firing rate distribution
    if plot:
        axs.append(fig.add_subplot(gs[1, 0]))

    t_mask = (P.t_start <= rsp.ts) & (rsp.ts < P.t_end)
    fr = rsp.spks[t_mask, :].sum(0) / (P.t_end - P.t_start)
    bins = np.histogram(fr)[1]
    
    if plot:
        axs[-1].hist(fr, bins=bins, zorder=0)

        if STORE_UNC is not None:
            axs[-1].hist(STORE_UNC.fr, bins=bins, zorder=1, alpha=0.3)

        axs[-1].set_xlabel('Firing Rate (Hz)')
        axs[-1].set_ylabel('# Neurons')
        axs[-1].set_title('Mean = {0:.2f} Hz, STD = {1:.2f} Hz'.format(fr.mean(), fr.std()))
    
    STORE.fr = deepcopy(fr)

    # correlation distribution
    if plot:
        axs.append(fig.add_subplot(gs[1, 1]))

    t_bins = np.arange(P.t_start, P.t_end + P.t_bin_size, P.t_bin_size)

    spk_cts = []
    for t_bin_start, t_bin_end in zip(t_bins[:-1], t_bins[1:]):
        t_bin_mask = (t_bin_start <= rsp.ts) & (rsp.ts < t_bin_end)
        spks_t_bin = rsp.spks[t_bin_mask, :]
        spk_cts.append(spks_t_bin.sum(0))

    corrs = np.corrcoef(spk_cts, rowvar=False)
    corrs = corrs[np.triu_indices(P.n, 1)]

    bins = np.histogram(corrs[~np.isnan(corrs)], bins=20)[1]
    
    if plot:
        axs[-1].hist(corrs, bins=bins, zorder=0)

        if STORE_UNC is not None:
            axs[-1].hist(STORE_UNC.corrs, bins=bins, zorder=1, alpha=0.3)

        axs[-1].set_xlabel('Pairwise FR Correlation')
        axs[-1].set_ylabel('Counts')
        axs[-1].set_title('Mean = {0:.3f}, STD = {1:.3f}'.format(corrs.mean(), corrs.std()))
    
    STORE.corrs = deepcopy(corrs)

    # in & out-degree distributions
    if plot:
        axs.append(fig.add_subplot(gs[2, 0]))
    
    cells = np.arange(P.n)
    in_degs = [STORE.g.in_degree(cell) for cell in cells]
    
    if plot:
        axs[-1].scatter(cells, in_degs, s=10)
        axs[-1].set_xlabel('Neuron')
        axs[-1].set_ylabel('In-degree')
        axs[-1].set_title('Mean = {0:.2f}, Q = {1:.4f}'.format(
            np.mean(in_degs), np.mean(in_degs)/(P.n - 1)))

        axs.append(fig.add_subplot(gs[2, 1]))
    
    out_degs = [STORE.g.out_degree(cell) for cell in cells]
    
    if plot:
        axs[-1].scatter(cells, out_degs, s=10)
        axs[-1].set_xlabel('Neuron')
        axs[-1].set_ylabel('Out-degree')
        axs[-1].set_title('Mean = {0:.2f}, Q = {1:.4f}'.format(
            np.mean(out_degs), np.mean(out_degs)/(P.n - 1)))

        axs.append(fig.add_subplot(gs[3, 0]))
        axs[-1].hist(in_degs, bins=10)
        axs[-1].set_xlabel('In-degree')
        axs[-1].set_ylabel('# Neurons')

        axs.append(fig.add_subplot(gs[3, 1]))
        axs[-1].hist(out_degs, bins=10)
        axs[-1].set_xlabel('Out-degree')
        axs[-1].set_ylabel('# Neurons')

        for ax in axs:
            set_font_size(ax, 12)
    
    # return firing rate distribution
    return fr


# Excitability analysis

def run_e_leak_change_smlns(make_ntwk, nrns, e_leaks, P, STORE, rsp_0=None):
    """
    Run identical simulations, except for a set of neurons with
    modified leak potentials.
    """
    
    if rsp_0 is None:
        
        ntwk_0 = make_ntwk(P, STORE)[0]
        rsp_0 = run(ntwk_0, P, STORE)[0]

        rsp_0.ntwk = deepcopy(ntwk_0)
    
    ntwk_1 = deepcopy(rsp_0.ntwk)
    ntwk_1.e_l[nrns] = e_leaks
    ntwk_1.v_reset[nrns] = e_leaks
    
    rsp_1 = run(ntwk_1, P, STORE)[0]
    
    rsp_1.ntwk = deepcopy(ntwk_1)
    
    return rsp_0, rsp_1


def get_frs(rsp, P):
    """Get firing rate distribution over cells."""
    
    t_wdw = (P.t_start, P.stm_off)
    
    t_mask = (t_wdw[0] <= rsp.ts) & (rsp.ts < t_wdw[1])
    spks = rsp.spks[t_mask, :]
    
    fr = spks.sum(0) / (t_wdw[1] - t_wdw[0])
    
    return fr


def dual_raster(ax, rsp_0, rsp_1, nrns_shown=None, nrns_changed=None):
    """Show two overlaid raster plots."""
    
    if nrns_shown is None:
        nrns_shown = range(rsp_0.ntwk.n)
        
    if nrns_changed is None:
        nrns_changed = []
        
    raster(ax, rsp_0.ts, rsp_0.spks, order=nrns_shown, c='b', lw=1, zorder=1)
    raster(ax, rsp_1.ts, rsp_1.spks, order=nrns_shown, c='r', lw=1, zorder=0)
    
    for nrn in nrns_changed:
        if nrn in nrns_shown:
            y = nrns_shown.index(nrn)
            ax.axhline(y, color='gray', alpha=0.4)
    
    return ax


def run_example(make_ntwk, nrns, e_leak, P, STORE):
    """Show raster overlay and firing rate distribution overlay."""
    
    rsp_0, rsp_1 = run_e_leak_change_smlns(make_ntwk, nrns, e_leak, P, STORE)

    gs = gridspec.GridSpec(2, 3)
    fig = plt.figure(figsize=(7.5, 5), tight_layout=True)
    axs = []
    
    axs.append(fig.add_subplot(gs[0, 1:]))

    dual_raster(axs[-1], rsp_0, rsp_1, nrns_changed=nrns)

    axs[-1].set_xlim(0, .2)
    axs[-1].set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    axs[-1].set_facecolor((.95, .95, .95))
    
    axs.append(fig.add_subplot(gs[1, :]))

    nrns = range(rsp_0.ntwk.n)

    axs[-1].bar(nrns, get_frs(rsp_1, P) - get_frs(rsp_0, P), color='k', align='center')

    axs[-1].set_ylim(-5, 60)
    
    axs[-1].set_xlabel('Neuron')
    axs[-1].set_ylabel('∆ firing rate (Hz)')
    
    axs.append(axs[-1].twinx())
    
    axs[-1].bar(nrns, get_frs(rsp_0, P), color='g', alpha=0.3, align='center')
    
    axs[-1].set_ylim(-5, 60)
    
    axs[-1].set_ylabel('Initial firing rate (Hz)', color='g')
    
    for ax in axs:
        set_font_size(ax, 12)
    
    return fig, axs


# Excitability influence distributions
def calc_xblt_ifl(make_ntwk, P, e_l_1, return_frs=False):
    """Calculate excitability influence, in-degree, and out-degree across neurons."""
    
    # loop over all neurons in network
    in_deg = []
    out_deg = []
    xblt_ifl = []
    
    frs_0 = []
    frs_1 = []

    rsp_0 = None
    
    STORE = GenericFlexible()
    
    for nrn in range(P.n):

        if nrn % 10 == 0:
            sys.stdout.write('|')
        else:
            sys.stdout.write('.')
        
        if rsp_0 is None:
            rsp_0, rsp_1 = run_e_leak_change_smlns(
                make_ntwk, [nrn], e_l_1, P, STORE)
        else:
            rsp_1 = run_e_leak_change_smlns(
                make_ntwk, [nrn], e_l_1, P, STORE, rsp_0=rsp_0)[1]

        # a few checks
        ## both ntwks have equal weight matrices
        assert(np.all(rsp_0.ntwk.ws_rcr['E'] == rsp_1.ntwk.ws_rcr['E'].todense()))

        ## in & out-degree match weight matrix calc
        assert((rsp_0.ntwk.ws_rcr['E'][nrn, :] > 0).sum() == STORE.g.in_degree(nrn))
        assert((rsp_0.ntwk.ws_rcr['E'][:, nrn] > 0).sum() == STORE.g.out_degree(nrn))

        # store in- & out-degree
        in_deg.append(STORE.g.in_degree(nrn))
        out_deg.append(STORE.g.out_degree(nrn))

        # calc & store excitability influence
        fr_0 = get_frs(rsp_0, P)
        fr_1 = get_frs(rsp_1, P)

        if return_frs:
            frs_0.append(deepcopy(fr_0))
            frs_1.append(deepcopy(fr_1))
            
        # mask out neuron whose excitability was changed
        fr_0[nrn] = np.nan
        fr_1[nrn] = np.nan

        xblt_ifl.append(np.sqrt(np.nansum((fr_0 - fr_1)**2) / P.n))
    
    if return_frs:
        return xblt_ifl, in_deg, out_deg, frs_0, frs_1
    else:
        return xblt_ifl, in_deg, out_deg
    

def plot_xblt_ifl_vs_deg(xblt_ifl, in_deg, out_deg):
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(7.5, 5), tight_layout=True)
    axs = []
    
    axs.append(fig.add_subplot(gs[0, 0]))

    axs[-1].scatter(in_deg, xblt_ifl, s=10)

    r, p = stats.pearsonr(in_deg, xblt_ifl)
    slp, icpt = stats.linregress(in_deg, xblt_ifl)[:2]

    x = np.array([np.nanmin(in_deg), np.nanmax(in_deg)])
    y = slp*x + icpt

    axs[-1].plot(x, y, c='r')

    axs[-1].set_xlabel('In-deg')
    axs[-1].set_ylabel('Exc. Ifl. (Hz)')
    axs[-1].set_title('R = {0:.3f}, P = {1:.3f}'.format(r, p))

    axs.append(fig.add_subplot(gs[0, 1]))
    
    axs[-1].scatter(out_deg, xblt_ifl, s=10)

    r, p = stats.pearsonr(out_deg, xblt_ifl)
    slp, icpt = stats.linregress(out_deg, xblt_ifl)[:2]

    x = np.array([np.nanmin(out_deg), np.nanmax(out_deg)])
    y = slp*x + icpt

    axs[-1].plot(x, y, c='r')

    axs[-1].set_xlabel('Out-deg')
    axs[-1].set_ylabel('Exc. Ifl. (Hz)')
    axs[-1].set_title('R = {0:.3f}, P = {1:.3f}'.format(r, p))

    axs.append(fig.add_subplot(gs[1, 0]))
    
    axs[-1].scatter(
        in_deg, out_deg, c=xblt_ifl, s=10, 
        vmin=np.nanmin(xblt_ifl), vmax=np.nanmax(xblt_ifl))

    axs[-1].set_xlabel('In-deg')
    axs[-1].set_ylabel('Out-deg')
    axs[-1].set_facecolor((.9, .9, .9))

    for ax in axs:
        set_font_size(ax, 12)