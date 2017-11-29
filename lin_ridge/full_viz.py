"""
Code for visualizing full linear ridge simulations.
"""
import matplotlib.pyplot as plt
import numpy as np

from db import make_session, d_models
from lin_ridge import full
from lin_ridge import search
from plot import raster as _raster


def raster(ts, spks, pfcs, cell_types, p, C, C_, fig_size, **scatter_kwargs):
    """
    Make a raster plot, ordering cells based on cell type,
    within-ridge status, and x-position.
    """
    # set default scatter plot kwargs
    scatter_kwargs = deepcopy(scatter_kwargs)
    
    if 'c' not in scatter_kwargs:
        scatter_kwargs['c'] = 'k'
    if 'lw' not in scatter_kwargs:
        scatter_kwargs['lw'] = 0
    if 's' not in scatter_kwargs:
        scatter_kwargs['s'] = 10
        
    class Rsp(object):
        pfcs = pfcs
        cell_types = cell_types
    
    # order cells by cell type, ridge status, and x-position
    ridge_mask = search.get_ridge_mask(Rsp(), p, C_)
    inh_mask = (cell_types == 'INH')
    non_ridge_pc_mask = ~(ridge_mask | inh_mask)

    categories = np.zeros(len(cell_types), dtype='int8')

    categories[ridge_mask] = 0
    categories[non_ridge_pc_mask] = 1
    categories[inh_mask] = 2

    order = np.lexsort((pfcs[0], categories))

    fig, axs = plt.subplots(2, 1, figsize=fig_size, tight_layout=True)
    
    t_masks = [(ts < C.T_EC), (C.T_EC <= ts)]
    
    for t_mask, ax in zip(t_masks, axs):
        _raster(ax, ts[t_mask], spks[t_mask], order, **scatter_kwargs)

        # draw lines separating ridge, non-ridge, and inh cell types
        y_0 = ridge_mask.sum() - 0.5
        y_1 = ridge_mask.sum() + non_ridge_pc_mask.sum() - 0.5

        ax.axhline(0, y_0, color='gray', ls='--', zorder=-1)
        ax.axhline(0, y_1, color='gray', ls='--', zorder=-1)

        set_font_size(ax, 16)

    return fig, axs


def raster_from_trial(trial_id, pre, C, P, C_, fig_size, **scatter_kwargs):
    """
    Make a raster plot given a trial/trial ID.
    """
    # set default scatter plot kwargs
    scatter_kwargs = deepcopy(scatter_kwargs)
    
    if 'c' not in scatter_kwargs:
        scatter_kwargs['c'] = 'k'
    if 'lw' not in scatter_kwargs:
        scatter_kwargs['lw'] = 0
    if 's' not in scatter_kwargs:
        scatter_kwargs['s'] = 10
        
    # get trial params
    session = make_session()
    trial_id = deepcopy(trial)
    trial = session.query(d_models.LinRidgeFullTrial).get(trial_id)
    session.close()

    if trial is None:
        print('Trial ID {} not found.'.format(trial_id))
        return

    rsp = full.run_smln(
        trial.id, d_models.LinRidgeFullTrial, pre, C, P, C_,
        save=False, seed=trial.seed, commit='none')
    
    return raster(
        rsp.ts, rsp.spks, rsp.pfcs, rsp.cell_types, rsp.p,
        C, C_, fig_size, **scatter_kwargs)


def animate():
    pass
