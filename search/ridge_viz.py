"""
Code for visualizing search results.
"""
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy.sql.expression import func

from . import ridge
from db import make_session, d_models
from plot import raster as _raster
from plot import set_font_size


def rslt_scatter(smln_id, filt, lmt=None, fig_size=(10, 10), **scatter_kwargs):
    """
    Make a scatter plot of activity and speed values for a set of
    ridge trials.
    
    :param smln_id: simulation id to get trials for
    :param filt: list of sqlalchemy filters to apply
    :param max_trials: max number of trials
    """
    
    session = make_session()
    
    trials = session.query(
        d_models.RidgeTrial.activity,
        d_models.RidgeTrial.speed).join(
        d_models.RidgeSearcher).filter(
        d_models.RidgeSearcher.smln_id == smln_id,
        *filt).limit(lmt)
    
    session.close()
    
    activities, speeds = np.array(trials.all()).T
    
    fig, ax = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
    ax.scatter(activities, speeds, **scatter_kwargs)
    
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    ax.set_xlabel('Activity (m^2/s)')
    ax.set_ylabel('Speed (m/s)')
    
    set_font_size(ax, 16)
    
    return ax


def select_trials(smln_id, filt, order_by=None, lmt=None, df=True):
    """
    Select a specific subset of ridge trials from the db.
    
    :param filt: list of sqlalchemy filters to apply
    """
    session = make_session()
    
    if order_by == 'rand':
        order_by = func.random()
    
    trials = session.query(d_models.RidgeTrial).join(
        d_models.RidgeSearcher).filter(
        d_models.RidgeSearcher.smln_id == smln_id,
        *filt).order_by(order_by).limit(lmt)
    
    session.close()
    
    if df:
        return pd.read_sql(trials.statement, trials.session.bind)
    else:
        return trials
    

def trial_set_scatter(
        smln_id, filts, cs, lmt=10000,
        params=(
            'P_INH', 'RHO_PC', 'Z_PC', 'L_PC', 'W_A_PC_PC', 'P_A_INH_PC',
            'W_A_INH_PC', 'P_G_PC_INH', 'W_G_PC_INH', 'FR_EC'),
        n_col=2, seed=0, **kwargs):
    """
    Select trial sets according to set of filters and plot
    scatter plots of parameters in different colors.
    """
    np.random.seed(0)
    
    row_height = 1 + .3 * len(filts)
    n_rows = int(np.ceil(len(params) / n_col))
    fig_size = (15, n_rows * row_height)
    fig, axs = plt.subplots(
        n_rows, n_col, figsize=fig_size,
        sharey=True, tight_layout=True, squeeze=False)
    
    for ctr, (filt, c) in enumerate(zip(filts, cs)):
        
        trials = select_trials(
            smln_id=smln_id, filt=filt, lmt=lmt, order_by='rand')
        
        y = -ctr * np.ones(len(trials))
        
        for param, ax in zip(params, axs.flat):
            ax.scatter(trials[param.lower()], y, c=c, **kwargs)
        
    for param, ax in zip(params, axs.flat):
        ax.set_ylim(-len(filts), 1)
        ax.set_xlabel(param)
        ax.yaxis.set_visible(False)
        set_font_size(ax, 16)
        
    return fig, axs


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
