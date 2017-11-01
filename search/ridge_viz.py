"""
Code for visualizing search results.
"""
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from . import ridge
from db import make_session, d_models
from plot import raster as _raster
from plot import set_font_size


def raster(
        trial, pre, C, P, ax_height=6, colors=(('PC', 'k'), ('INH', 'r')),
        show_all_rsps=False, **scatter_kwargs):
    """
    Display a raster plot for each of the ntwk responses used in a given trial.
    
    :param trial: trial identifier or instance
    """
    # set default scatter plot kwargs
    scatter_kwargs = deepcopy(scatter_kwargs)
    
    if 'marker' not in scatter_kwargs:
        scatter_kwargs['marker'] = '|'
    if 'c' not in scatter_kwargs:
        scatter_kwargs['c'] = 'k'
    if 'lw' not in scatter_kwargs:
        scatter_kwargs['lw'] = 3
    if 's' not in scatter_kwargs:
        scatter_kwargs['s'] = 10
        
    # get trial params
    if isinstance(trial, int):
        session = make_session()
        trial_id = deepcopy(trial)
        trial = session.query(d_models.RidgeTrial).get(trial_id)
        session.close()
    
        if trial is None:
            print('Trial ID {} not found.'.format(trial_id))
            return

    p = ridge.trial_to_p(trial)
    
    # run ntwk obj function
    rslts, rsps = ridge.ntwk_obj(p, pre, C, P, trial.seed, test=True)
    
    # get final rsps for each ntwk
    if show_all_rsps:
        
        # show all runs for each ntwk (for debugging)
        rsps_final = sum(rsps, [])
        titles = []
        
        for n_ctr, rsps_ in enumerate(rsps):
            
            titles.extend([
                'Ntwk {}: Run {}'.format(n_ctr + 1, r_ctr + 1)
                for r_ctr in range(len(rsps_))
            ])
            
    else:
        rsps_final = [rsps_[-1] for rsps_ in rsps]
        titles = [
            'Ntwk {}: Final Run'.format(n_ctr)
            for n_ctr in range(len(rsps_final))
        ]
    
    # plot rasters
    n = len(rsps_final)
    
    fig_size = (15, ax_height*n)
    fig, axs = plt.subplots(
        n, 1, figsize=fig_size, tight_layout=True, squeeze=False)
    axs = axs[:, 0]
    
    for rsp, title, ax in zip(rsps_final, titles, axs):
        
        # order cells by place fields
        if rsp.pfcs is None:
            print('WARNING: No place fields found in ntwk response.')
            order = np.arange(rsp.n)
        else:
            # sort from left to right
            order = np.argsort(rsp.pfcs[0, :])
        
        # color by cell types
        if colors is not None:
            
            cs = np.empty(rsp.n, dtype=object)
            cs[:] = 'k'
            
            for ct, c in colors:
                
                if ct not in rsp.cell_types:
                    print(
                        'WARNING: Cell type {} not found in ntwk rsp.'.format(ct))
                else:
                    cs[rsp.cell_types == ct] = c
                
            scatter_kwargs['c'] = cs
        
        _raster(ax, rsp.ts, rsp.spks, order, **scatter_kwargs)
        
        ax.set_title(title)
        
        set_font_size(ax, 16)
        
    return fig, axs, rslts, rsps_final
