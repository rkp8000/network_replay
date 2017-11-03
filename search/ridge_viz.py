"""
Code for visualizing search results.
"""
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import pandas as pd
from sqlalchemy.sql.expression import func

import aux
from anim import build_frames, create_mp4
from anim import random_oval
from search import ridge
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


def animate(
        save_dir, trial_id, run, pre, C, P,
        positions='pfcs', fig_size=(15, 7.5), report_every=60):
    """
    Animate the activity of a ridge trial.
    
    :param run: index of run to animate (since trials comprise multiple runs)
    """
    mpl.rcParams['agg.path.chunksize'] = 10000
    
    # get trial params
    session = make_session()
    trial = session.query(d_models.RidgeTrial).get(trial_id)
    p = ridge.trial_to_p(trial)
    session.close()
    
    print('\nRunning network simulations...')
    rslts, rsps = ridge.ntwk_obj(p, pre, C, P, trial.seed, test=True)
    rsp = rsps[run][-1]
    print('Results: ')
    print(rslts)
    
    print('\nSaving response file...')
    ntwk_path = rsp.save(
        os.path.join(save_dir, 'ntwk-{}-{}.npy'.format(trial_id, run)))
    
    # set cell positions
    if positions == 'pfcs':
        positions = rsp.pfcs.copy()
        inh_mask = rsp.cell_types == 'INH'

        # random points on circle
        pos_mean_inh = (0, -2*p['RIDGE_H'])
        pos_rad_inh = (2*p['RIDGE_H'], p['RIDGE_H'])

        positions[:, inh_mask] = random_oval(
            pos_mean_inh, pos_rad_inh, inh_mask.sum()).T
        
        box = [
            -1.05 * p['RIDGE_W']/2, 1.05 * p['RIDGE_W']/2,
            -3.05 * p['RIDGE_H'], 1.05 * p['RIDGE_H']/2,
        ]
    elif positions == 'random':
        positions = np.random.rand(2, rsp.pfcs.shape[1])
        box = [-.1, 1.1, -.1, 1.1]
    
    print('\nBuilding frames...\n')
    frame_prfx = os.path.join(save_dir, 'frames-{}-{}'.format(trial_id, run), 'f-')
    frames = build_frames.ntwk(
        save_prfx=frame_prfx,
        ntwk_file=ntwk_path,
        fps=1000,
        box=box,
        resting_size=30,
        spk_size=300,
        amp=3,
        positions=positions,
        default_color={'PC': 'k', 'INH': 'g'},
        cxn_color={
            ('PC', 'PC'): 'gray', ('INH', 'PC'): 'gray',
            ('PC', 'INH'): 'b', ('INH', 'INH'): 'b'
        },
        cxn_lw={
            ('PC', 'PC'): .04, ('INH', 'PC'): .01,
            ('PC', 'INH'): .01, ('INH', 'INH'): 0
        },
        cxn_zorder={
            ('PC', 'PC'): -1, ('INH', 'PC'): -2,
            ('PC', 'INH'): -2, ('INH', 'INH'):-2 
        },
        frames_per_spk=2,
        title='Trial {}:{}'.format(trial_id, run),
        x_label='pfc_x (m)',
        y_label='pfc_y (m)',
        fig_size=fig_size,
        verbose=True,
        report_every=report_every)
    
    print('\nMaking animation...\n')
    mp4_path = os.path.join(save_dir, 'ntwk-{}-{}'.format(trial_id, run))
    create_mp4(frames, mp4_path, playback_fps=30, verbose=True)
