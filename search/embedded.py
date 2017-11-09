from copy import deepcopy
from datetime import datetime, timedelta
import importlib
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from statsmodels.robust.scale import mad as med_abs_dev
import threading
import time
import traceback

import aux
from db import make_session, d_models
from ntwk import cxns_pcs_rcr, LIFNtwk
from search import embedded_pre

cc = np.concatenate


# SEARCH FUNCTIONS

def launch_searchers(role, obj, C, n, seed=None, commit=None):
    """
    Wrapper around search that launches one or more
    searcher daemons and returns corresponding threads.
    """
    if commit is None:
        commit = input(
            'Please commit relevant files and enter commit id (or "q" to quit): ')
        if commit.lower() == 'q':
            raise KeyboardInterrupt('Execution halted by user.')
     
    threads = []
    for ctr in range(n):
        
        # define argument-less target function
        def targ(): 
            return search(
                role, obj, C, seed=seed, commit=commit)
        
        # launch searcher thread
        thread = threading.Thread(target=targ)
        thread.start()
        
        threads.append(thread)
        
    print('\n{} searchers launched.'.format(n))
        
    return threads
    
    
def search(
        role, obj, C, seed=None, commit=None, verbose=False):
    """
    Launch instance of searcher exploring potentiated ridge trials.
    
    :param role: searcher role, which should correspond to search config file
    :param obj: objective function
    :param C: global configuration module
    :param seed: RNG seed
    :param config_root: module to look for config files in
    :param commit: current git commit id
    :param verbose: set to True to print log info to console
    """
    if commit is None:
        commit = input(
            'Please commit relevant files and enter commit id (or "q" to quit): ')
        if commit.lower() == 'q':
            raise KeyboardInterrupt('Execution halted by user.')
        
    np.random.seed(seed)
    
    # import initial config
    cfg = importlib.import_module('.'.join([C.CONFIG_ROOT, role]))
    
    # connect to db
    session = make_session()
    
    # make new searcher
    searcher = d_models.EmbeddedSearcher(
        smln_id=cfg.SMLN_ID,
        role=role,
        last_active=datetime.now(),
        error=None,
        traceback=None,
        commit=commit)
    
    session.add(searcher)
    session.commit()
    
    if verbose:
        print('SEARCHER ID = {}'.format(searcher.id))
        
    # initialize useful aux. vbls
    since_jump = 0
    last_force = None
    forces_left = 0
    
    # loop over search iterations
    for ctr in range(C.MAX_SEARCH_ITER):
        
        # attempt iteration
        try:
            # reload search config
            importlib.reload(cfg)
            
            # validate config
            validate(cfg, ctr)
            
            error_printed = False
            
            # define parameter conversions
            p_to_x, x_to_p = make_param_conversions(cfg)

            # "additional step required" flag
            step = True
            
            # get forced param sets
            if hasattr(cfg, 'FORCE') and searcher.id in cfg.FORCE:
                
                # if change detected since last force
                if str(cfg.FORCE[searcher.id]) != str(last_force):
                    
                    # store new forces
                    force = cfg.FORCE[searcher.id]
                    forces_left = len(force)
                    last_force = deepcopy(force)
                    
                    if verbose:
                        print('New forcing sequence detected.')
                    
                # do nothing if no change detected
                # ...
            else:
                force = None
                forces_left = 0
            
            # first iter, forced, jump or stay
            if ctr == 0 or forces_left:  # first iter or forced
                
                if ctr == 0:  # first iter
                    force_ = cfg.START
                elif forces_left:  # forced
                    force_ = force[-forces_left]
                    forces_left -= 1
                
                x_cand = force_to_x(cfg, force_)
                x_prev = x_cand
                step = False
                
                if force_ in ['random', 'center']:
                    since_jump = 0
                else:
                    since_jump += 1
               
            elif np.random.rand() < cfg.Q_JUMP:  # jump
                
                # new point or prev
                if np.random.rand() < cfg.Q_NEW:  # new
                    
                    # get next candidate x
                    x_cand = sample_x_rand(cfg)
                    x_prev = x_cand
                    step = False
                    
                else:  # prev
                    x_prev = sample_x_prev(cfg, session, p_to_x)
                
                since_jump = 0
                    
            else:  # stay
                x_prev = x.copy()
                since_jump += 1
            
            # take new step if necessary
            if step:
                x_cand = sample_x_step(
                    cfg, session, searcher, x_prev, since_jump, p_to_x)
            
            # ensure x_cand is within bounds
            x_cand = fix_x_if_out_of_bounds(cfg, x_cand)
            
            # convert x_cand to param dict and compute objective function
            p_cand = x_to_p(x_cand)
            seed_ = np.random.randint(C.MAX_SEED)
            
            rslt = obj(p_cand, seed_)
            
            save_embedded_trial(
                session=session,
                searcher=searcher,
                seed=seed_,
                p=p_cand,
                rslt=rslt)
            
            # move to x_cand if stable else x_prev
            if rslt['STABILITY'] > 0:
                x = x_cand.copy()
            else:
                x = x_prev.copy()
                
            # update searcher error msg
            searcher.error = None
            searcher.traceback = None
            
        except Exception as e:
            
            tb = traceback.format_exc()
            
            if verbose and not error_printed:
                print('Configuration error detected.')
                print(tb)
                error_printed = True
                
            # update searcher error msg
            searcher.error = e.__class__.__name__
            searcher.traceback = tb
        
        searcher.last_active = datetime.now()
        session.add(searcher)
        session.commit()
        
        if searcher.error is not None:
            time.sleep(C.WAIT_AFTER_ERROR)
    
    session.close()
            
    return searcher.id


def search_status(smln_id=None, role=None, recent=30):
    """Return IDs of recently active searchers under specified smln/role.
    
    :param smln_id: simulation id
    :param role: conditioning role for searcher lookup
    :param recent: max time searcher must have been active within to be recent
    """
    earliest = datetime.now() - timedelta(0, recent)
    
    # get searchers
    session = make_session()
    searchers = session.query(d_models.EmbeddedSearcher).filter(
        d_models.EmbeddedSearcher.last_active >= earliest)
    
    if smln_id is not None:
        searchers = searchers.filter(d_models.EmbeddedSearcher.smln_id == smln_id)
    if role is not None:
        searchers = searchers.filter(d_models.EmbeddedSearcher.role == role)
    
    searchers = searchers.order_by(
        d_models.EmbeddedSearcher.smln_id,
        d_models.EmbeddedSearcher.id,
        d_models.EmbeddedSearcher.role)
    
    session.close()
    
    # check for searchers with error messages
    suspended = []
    errors = []
    
    for searcher in searchers:
        if searcher.error is not None:
            suspended.append(searcher.id)
            errors.append(searcher.error)
    
    # print results
    print(
        'The following searchers were active in the last {}'
        ' s:\n'.format(searchers.count(), recent))
    
    for searcher in searchers:
        print('{}   {}   {}'.format(
            searcher.smln_id, searcher.id, searcher.role))
        
    print('')
    
    if suspended:
        print('The following searchers were suspended by errors:')

        for suspended_, error in zip(suspended, errors):
            print('{}: ERROR: "{}".'.format(suspended_, error))

        print('\nLook up error tracebacks using read_search_error(id).\n')
    else:
        print('No searchers were suspended by errors.')


def read_search_error(searcher_id):
    """Print error traceback for specific searcher_id."""
    
    # get searcher
    session = make_session()
    searcher = session.query(d_models.EmbeddedSearcher).get(searcher_id)
    session.close()
    
    if searcher is None:
        print('Searcher with ID {} not found.'.format(searcher_id))
        return
    
    if searcher.error is None and searcher.traceback is None:
        print('No error found in searcher {}.'.format(searcher_id))
        return
        
    else:
        print('ID: {}'.format(searcher_id))
        print('ERROR: {}\n'.format(searcher.error))
        print(searcher.traceback)
        return
    

# EMBEDDED-TRIAL-SPECIFIC OBJECTIVE FUNCTION AND HELPERS

def ntwk_obj(p, pre, C, P, seed, test=False):

    np.random.seed(seed)
    
    # loop over ntwks
    stabilities = np.nan * np.zeros(C.N_NTWKS)
    angles = np.nan * np.zeros(C.N_NTWKS)
    activities = np.nan * np.zeros(C.N_NTWKS)
    speeds = np.nan * np.zeros(C.N_NTWKS)
    
    rsps = []
    
    for n_ctr in range(C.N_NTWKS):
        
        # make ntwk
        ntwk = p_to_ntwk(p, pre, P)
        
        # stabilize ntwk
        rsps_, stability, angle, activity, speed = stabilize(ntwk, p, pre, C, P)
        
        # store results
        stabilities[n_ctr] = stability
        angles[n_ctr] = angle
        activities[n_ctr] = activity
        speeds[n_ctr] = speed
        
        if test:
            rsps.append(rsps_)
    
    # average activity/speed over stable runs
    mask = stabilities == 1
    
    if mask.sum():
        rslts = {
            'STABILITY': np.mean(stabilities),
            'ANGLE': np.mean(angles[mask]),
            'ACTIVITY': np.mean(activities[mask]),
            'SPEED': np.mean(speeds[mask])
        }
    else:
        rslts = {'STABILITY': 0., 'ANGLE': np.nan, 'ACTIVITY': 0., 'SPEED': 0.}
    
    if test:
        return rslts, rsps
    else:
        return rslts


def p_to_ntwk(p, pre, P):
    """
    Instantiate a new ntwk from a param dict.
    """
    ## make ridge
    pfcs, ws_n_pc_ec = embedded_hz(p, pre)
    
    n_pc = pfcs.shape[1]
    n_ec = n_pc
    
    assert len(ws_n_pc_ec) == n_pc

    ## make INHs
    n_inh = np.random.poisson(p['P_INH'] * n_pc)
    n = n_pc + n_inh

    ## make upstream cxns
    ws_up = {
        'AMPA': np.zeros((n, n_ec)),
        'NMDA': cc([np.diag(ws_n_pc_ec), np.zeros((n_inh, n_ec))]),
        'GABA': np.zeros((n, n_ec))
    }

    ## make recurrent cxns
    ws_rcr = {
        'AMPA': np.zeros((n, n)),
        'NMDA': np.zeros((n, n)),
        'GABA': np.zeros((n, n)),
    }

    ### pc -> pc cxns
    ws_rcr['AMPA'][:n_pc, :n_pc] = p['W_A_PC_PC'] * \
        cxns_pcs_rcr(pfcs, p['Z_PC'], p['L_PC']).astype(float)

    ### pc -> inh cxns
    ws_rcr['AMPA'][-n_inh:, :n_pc] = p['W_A_INH_PC'] * \
        np.random.binomial(1, p['P_A_INH_PC'], (n_inh, n_pc))

    ### inh -> pc cxns
    ws_rcr['GABA'][:n_pc, -n_inh:] = p['W_G_PC_INH'] * \
        np.random.binomial(1, p['P_G_PC_INH'], (n_pc, n_inh))

    ## instantiate ntwk
    ntwk = LIFNtwk(
        t_m=cc([np.repeat(P.T_M_PC, n_pc), np.repeat(P.T_M_INH, n_inh)]),
        e_l=cc([np.repeat(P.E_L_PC, n_pc), np.repeat(P.E_L_INH, n_inh)]),
        v_th=cc([np.repeat(P.V_TH_PC, n_pc), np.repeat(P.V_TH_INH, n_inh)]),
        v_reset=cc([
            np.repeat(P.V_RESET_PC, n_pc), np.repeat(P.V_RESET_INH, n_inh)]),
        t_r=cc([np.repeat(P.T_R_PC, n_pc), np.repeat(P.T_R_INH, n_inh)]),
        
        e_ahp=P.E_AHP_PC, t_ahp=P.T_AHP_PC,
        w_ahp=cc([np.repeat(P.W_AHP_PC, n_pc), np.repeat(P.W_AHP_INH, n_inh)]),
        
        es_syn={'AMPA': P.E_A, 'NMDA': P.E_N, 'GABA': P.E_G},
        ts_syn={'AMPA': P.T_A, 'NMDA': P.T_N, 'GABA': P.T_G},
        
        ws_up=ws_up, ws_rcr=ws_rcr, plasticity=None)

    ntwk.n_pc = n_pc
    ntwk.n_ec = n_ec
    ntwk.n_inh = n_inh

    ntwk.pfcs = cc([pfcs, np.nan * np.zeros((2, n_inh))], 1)
    ntwk.cell_types = cc([np.repeat('PC', n_pc), np.repeat('INH', n_inh)])
    
    return ntwk


def embedded_hz(p, pre):
    """
    Randomly sample PCs from along a horizontal "ridge", assigning them each
    a place-field center and an EC->PC cxn weight.
    
    :return: place field centers, EC->PC cxn weights
    """
    shape = (p['AREA_W'], p['AREA_H'])
    y = p['RIDGE_Y']
    dens= p['RHO_PC']
    
    # sample number of nrns
    n_pcs = np.random.poisson(shape[0] * shape[1] * dens)
    
    # sample place field positions
    pfcs = np.random.uniform(
        (-shape[0]/2, -shape[1]/2), (shape[0]/2, shape[1]/2),
        (n_pcs, 2)).T
    
    # compute dists to ridge centerline
    dists = np.abs(pfcs[1, :] - y)
    
    # sample and return EC->PC NMDA weights
    return pfcs, embedded_pre.sample_w_n_pc_ec(dists, pre)


def stabilize(ntwk, p, pre, C, P, test=False):
    """
    Run ntwk cyclically until activity from beginning to end of sim
    does not change.
    """
    # sample initial vs and gs and get min nonzero firing rate
    vs_0, gs_0, fr_nz = sample_v_0_g_0_fr_nz(ntwk, p, pre, C, P)
   
    # get ridge y-bounds
    y_min, y_max = ridge_y_bounds(p, C)
    
    # get x-ordered nrn idxs for nrns within ridge
    ridge_mask = get_ridge_mask(ntwk, p, C)
    
    xs_all = ntwk.pfcs[0, :].copy()
    xs_all[~ridge_mask] = np.nan
    
    x_order = np.argsort(xs_all)
    
    # get subregion of ridge to force initial spks
    x_max = -(p['AREA_W'] / 2) + (C.N_L_PC_FORCE * p['L_PC'])
    
    force_mask = ridge_mask.copy()
    force_mask[ridge_mask] = (ntwk.pfcs[0, ridge_mask] < x_max)
    
    if test:
        print('{} initial forced spks.'.format(force_mask.sum()))
    
    idx_stim = int(C.T_FORCE / P.DT)
    spks_forced = np.zeros((idx_stim + 1, ntwk.n))
    spks_forced[idx_stim, force_mask] = 1
    
    vs_forced = None

    # make upstream EC spks
    ts = np.arange(0, C.SMLN_DUR, P.DT)
    spks_up = np.random.poisson(
        p['FR_EC'] * P.DT, (len(ts), ntwk.n_ec))
    
    # loop over repeats until steady state is reached
    rsps = []
    fr_decay_prev = np.inf

    for ctr in range(C.MAX_RUNS_STABILIZE):
        
        # run ntwk with forced spks and vs
        rsp = ntwk.run(
            spks_up, P.DT, vs_0=vs_0, gs_0=gs_0,
            vs_forced=vs_forced, spks_forced=spks_forced)

        # attach useful attributes
        rsp.pfcs = ntwk.pfcs
        rsp.cell_types = ntwk.cell_types
        
        rsps.append(rsp)
        
        # get time window of longest within-ridge propagation
        wdw_prop = check_propagation(rsp, fr_nz, p, C, P)

        # break if no propagation
        if wdw_prop is None:
            stability = 0
            break
            
        # get mean within-ridge PC fr during decay-check wdw
        fr_decay = get_fr_decay(rsp, wdw_prop, p, C, P)
        
        if test:
            print('Run {0}: {1} forced spks, fr_decay = {2:.3} Hz'.format(
                ctr+1, spks_forced.sum(), fr_decay))
        
        # compare activity in decay-check wdw to last run
        if fr_decay < fr_decay_prev * C.DECAY_RATIO:
            # update reference fr
            fr_decay_prev = fr_decay
        else:
            # declare ntwk stable
            stability = 1
            break
            
        # copy final within-ridge vs and spks
        vs_final, spks_final = copy_final(rsp, wdw_prop, p, C)
        
        # create vs/spks forcing matrices from finals for next run
        n_forced = vs_final.shape[1]
        
        vs_forced_ = np.nan * np.zeros((len(vs_final), ntwk.n))
        spks_forced_ = np.zeros((len(spks_final), ntwk.n))
        
        vs_forced_[:, x_order[:n_forced]] = vs_final
        spks_forced_[:, x_order[:n_forced]] = spks_final
        
        # buffer with pre-force time
        pre_force = (int(C.T_FORCE/P.DT), ntwk.n)
        vs_forced = cc([np.nan * np.zeros(pre_force), vs_forced_])
        spks_forced = cc([np.zeros(pre_force), spks_forced_])
        
    else:
        # assume ntwk is close enough to stable if stability not
        # reached in max number of runs
        stability = 1
        
    if not stability:
        angle = np.nan
        activity = 0.
        speed = 0.
    else:
        # get angle, activity, and speed of final ntwk response
        angle = get_angle(rsp, wdw_prop, p, C)
        activity = get_activity(rsp, wdw_prop, p, C, P)
        speed = get_speed(rsp, wdw_prop, p, C)
    
    return rsps, stability, angle, activity, speed


def get_ridge_mask(ntwk_or_rsp, p, C):
    """Return mask over all PCs within ridge."""
    
    y_min = p['RIDGE_Y'] - C.RIDGE_H / 2
    y_max = p['RIDGE_Y'] + C.RIDGE_H / 2
    
    mask = ntwk_or_rsp.cell_types == 'PC'
    mask[mask] = (y_min <= ntwk_or_rsp.pfcs[0, mask]) \
        & (ntwk_or_rsp.pfcs[0, mask] < y_max)
    
    return mask


def sample_v_0_g_0_fr_nz(ntwk, p, pre, C, P, test=False):
    """
    Sample initial voltages and gs, as well as minimum average firing
    rate required for a ntwk's activity to be considered non-zero.
    """
    # sample initial vs and g_ns
    vs_0_pc, gs_n_0_pc = embedded_pre.sample_v_g(ntwk, p, pre)
    
    vs_0 = cc([vs_0_pc, P.E_L_INH * np.ones(ntwk.n_inh)])
    
    gs_0 = {
        'AMPA': np.zeros(ntwk.n),
        'NMDA': cc([gs_n_0_pc, np.zeros(ntwk.n_inh)]),
        'GABA': np.zeros(ntwk.n)
    }
    
    ts = np.arange(0, C.SMLN_DUR, P.DT)

    spks_up = np.random.poisson(p['FR_EC'] * P.DT, (len(ts), ntwk.n_ec))

    ## run ntwk
    rsp_bkgd = ntwk.run(spks_up, P.DT, vs_0=vs_0, gs_0=gs_0)

    ## calculate background within-ridge PC rate mean and std
    ridge_mask = get_ridge_mask(ntwk, p, C)
    
    rate_bkgd = np.mean(rsp_bkgd.spks[:, ridge_mask] / P.DT, 1)
    rate_bkgd_mean = np.mean(rate_bkgd)
    rate_bkgd_std = np.std(rate_bkgd)
    
    fr_nz = max(
        rate_bkgd_mean + (C.MIN_PC_FR_NZ_SGMS * rate_bkgd_std), C.MIN_FR_NZ)
    
    if not test:
        return vs_0, gs_0, fr_nz
    else:
        return vs_0, gs_0, fr_nz, rsp_bkgd, rate_bkgd


def check_propagation(rsp, fr_nz, p, C, P):
    """
    Check if propagation occurred in single ntwk response.
    If so, return start and end times of longest propagation, otherwise
    return None.
    
    :param rsp: ntwk response instance
    :param fr_nz: min per-cell firing rate for pop activity to be nonzero
    """
    if not hasattr(rsp, 'pfcs'):
        raise KeyError(
            'Ntwk response must include place field '
            'centers in attribute rsp.pfcs.')
    if not hasattr(rsp, 'cell_types'):
        raise KeyError(
            'Ntwk response must include cell types '
            'in attribute rsp.cell_types.')
        
    # get pfcs and spks from within-ridge pc_pop
    ridge_mask = get_ridge_mask(rsp, p, C)
    pfcs_ridge = rsp.pfcs[:, ridge_mask]
    spks_ridge = rsp.spks[:, ridge_mask]
    
    # get smoothed ridge firing rate, avg'd over cells
    fr_ridge = spks_ridge.mean(1) / P.DT
    fr_ridge_smooth = aux.running_mean(fr_ridge, int(C.PPGN_WDW / P.DT))
    
    # divide into segments where ridge fr exceeds bkgd
    segs = aux.find_segs(fr_ridge_smooth > fr_nz)
    
    if not len(segs):
        return None
    
    # get longest segment and convert to continuous time
    wdw = segs[np.argmax(segs[:, 1] - segs[:, 0])] * P.DT
    
    # return seg if activity was still going on at end of run
    if wdw[1] >= (rsp.ts[-1] - C.PPGN_WDW/2):
        return wdw
    
    # check if end of ridge was active during final moments of seg
    
    ## get mask of final cells (x greater than ridge end minus length scale)
    x_min = p['AREA_W'] / 2 - p['L_PC']
    final_ridge_mask = pfcs_ridge[0, :] >= x_min
    n_final = final_ridge_mask.sum()
    
    ## select final time window as approx time for constant speed
    ## propagation to cover one length scale
    speed = (p['RIDGE_W']) / (wdw[1] - wdw[0])
    t_start = wdw[1] - (C.PPGN_LOOK_BACK * p['L_PC'] / speed)
    t_end = wdw[1]
    t_mask = (t_start <= rsp.ts) & (rsp.ts < t_end)
    
    if not t_mask.sum():
        return None
    
    ## compute average per-cell firing rate of final cells during final window
    fr_final = np.mean(spks_ridge[t_mask, :][:, final_ridge_mask] / P.DT)
    
    ## check if firing rate for final cells in final time window above baseline
    if fr_final > fr_nz:
        return wdw
    else:
        return None

    
def get_fr_decay(rsp, wdw, p, C, P):
    """
    Get firing rate over decay window in C.
    
    :param rsp: ntwk response object with cell_types attribute
    :param wdw: (start, end) times (in s) of propagation epoch
    """
    dur = wdw[1] - wdw[0]
    
    start = wdw[0] + C.DECAY_WDW[0] * dur
    end = wdw[0] + C.DECAY_WDW[1] * dur
    
    t_mask_decay = (start <= rsp.ts) & (rsp.ts < end)
    ridge_mask = get_ridge_mask(rsp, p, C)

    return rsp.spks[t_mask_decay, :][:, ridge_mask].mean()


def copy_final(rsp, wdw, p, C):
    """
    Copy final vs & spks in a ntwk run that exhibited propagation.
    
    :param rsp: ntwk response instance
    :param wdw: start and end of propagation epoch
    """
    if not hasattr(rsp, 'pfcs'):
        raise KeyError(
            'Ntwk response must include place field '
            'centers in attribute rsp.pfcs.')
    if not hasattr(rsp, 'cell_types'):
        raise KeyError(
            'Ntwk response must include cell types '
            'in attribute rsp.cell_types.')
        
    # fit line to ridge spks contained between t_start and t_end
    ridge_mask = get_ridge_mask(rsp, p, C)
    t_mask = (wdw[0] <= rsp.ts) & (rsp.ts < wdw[1])
    
    vs_ridge = rsp.vs[:, ridge_mask]
    spks_ridge = rsp.spks[:, ridge_mask]
    pfcs_ridge = rsp.pfcs[:, ridge_mask]
    
    # get time and cell idxs
    spk_t_idxs, spk_cells = spks_pc[t_mask, :].nonzero()
    
    # convert time idxs to time
    spk_ts = rsp.ts[t_mask][spk_t_idxs]
    
    # convert cell idxs to x-positions
    spk_xs = pfcs_ridge[0, :][spk_cells]
    
    # fit line
    rgr = Lasso()
    rgr.fit(spk_ts[:, None], spk_xs)
    
    # label spks as belonging to wave if within fraction length scales of line
    dists = np.abs(rgr.predict(spk_ts[:, None]) - spk_xs)
    in_wave = dists <= C.WAVE_TOL * p['L_PC']
    
    # let x_2 be x-pos of rightmost spk belonging to wave and x_1 be
    # x_2 minus a "look-back factor" times the length scale
    x_2 = spk_xs[in_wave].max()
    x_1 = x_2 - (C.LOOK_BACK_X * p['L_PC'])
    
    # let t_2 be final time of wave and t_1 be time of first in-wave
    # spk from any cell between x_1 and x_2
    t_2 = spk_ts[in_wave].max()
    t_1 = spk_ts[(x_1 <= spk_xs) & (spk_xs < x_2) & in_wave].min()

    # make cell and time mask
    ridge_mask_final = (x_1 <= pfcs_ridge[0, :]) & (pfcs_ridge[0, :] < x_2)
    t_mask_final = (t_1 <= rsp.ts) & (rsp.ts < t_2)
    
    # get final vs & spks
    vs_final = vs_ridge[t_mask_final, :][:, ridge_mask_final]
    spks_final = spks_ridge[t_mask_final, :][:, ridge_mask_final]
    
    # reorder cells according to xs
    xs_final = pfcs_ridge[0, ridge_mask_final]
    x_order_final = np.argsort(xs_final)
    
    return vs_final[:, x_order_final], spks_final[:, x_order_final]


def get_angle(rsp, wdw):
    """
    Calculate the angle of the trajectory reconstructed
    from activity propagation.
    
    :param rsp: ntwk response object
    :param wdw: window during which propagation occurs
    """
    
    # get all pc spks within propagation wdw
    t_mask = (wdw[0] <= rsp.ts) & (rsp.ts < wdw[1])
    pc_mask = rsp.cell_types == 'PC'
    
    spks_pc = rsp.spks[t_mask, :][:, pc_mask]
    
    # get times, xs, and ys of pc spks
    t_idxs, pcs = spks_pc.nonzero()
    
    ts_spk = rsp.ts[t_mask][t_idxs]
    xs_spk = rsp.pfcs[0, pc_mask][pcs]
    ys_spk = rsp.pfcs[1, pc_mask][pcs]
    
    # get dx/dt and dy/dt by fitting lines to x & y vs t
    rgr_x = LinearRegression()
    rgr_x.fit(ts_spk[:, None], xs_spk)
    dx_dt = rgr_x.coef_[0]
    
    rgr_y = LinearRegression()
    rgr_y.fit(ts_spk[:, None], ys_spk)
    dy_dt = rgr_y.coef_[0]

    return aux.angle_from_dx_dt_dy_dt(dx_dt, dy_dt, 'deg')

    
def get_activity(rsp, wdw, p, C, P):
    """
    Get ratio of time-averaged within-ridge firing rate during
    propagation period to PC density (in final units of m^2/s).
    
    :param rsp: ntwk response object
    :param wdw: (start, end) of propagation epoch
    """
    # convert activity measurement times to absolute
    dur = wdw[1] - wdw[0]
    
    start = wdw[0] + C.ACTIVITY_WDW[0] * dur
    end = wdw[0] + C.ACTIVITY_WDW[1] * dur
    
    # get ridge-pop. firing rate in window
    ridge_mask = get_ridge_mask(rsp, p, C)
    t_mask = (start <= rsp.ts) * (rsp.ts < end)
    
    spk_cts_pop = np.sum(rsp.spks[:, ridge_mask][t_mask, :], 1)
    fr_pop = np.mean(spk_cts_pop / P.DT)
    
    # divide population firing rate by density
    return fr_pop / p['RHO_PC']


def get_speed(rsp, wdw, p, C):
    """
    Get linear speed of propagating ntwk response within ridge.
    
    :param rsp: ntwk response object
    :param wdw: (start, end) of propagation epoch
    """
    ridge_mask = get_ridge_mask(rsp, p, C)
    spks_ridge = rsp.spks[:, ridge_mask]
    pfcs_ridge = rsp.pfcs[:, ridge_mask]
    
    # convert speed measurement times to absolute
    dur = wdw[1] - wdw[0]
    
    start = wdw[0] + C.SPEED_WDW[0] * dur
    end = wdw[0] + C.SPEED_WDW[1] * dur
    
    t_mask = (start <= rsp.ts) & (rsp.ts < end)
    
    # get times and x-positions of all spks
    spk_t_idxs, spk_cells = spks_ridge[t_mask, :].nonzero()
    
    spk_ts = rsp.ts[t_mask][spk_t_idxs]
    spk_xs = pfcs_ridge[0, :][spk_cells]
    
    # fit line to get approximate speed
    rgr = LinearRegression()
    rgr.fit(spk_ts[:, None], spk_xs)
    
    speed = rgr.coef_[0]
    
    return speed


def decode_traj(rsp, wdw, smooth=0.003, mad_max=0.25):
    """
    Decode a traj from a population spike train and place field
    centers of corresponding PCs.
    
    :param wdw: (start, end) window over which propagation occurs (s)
    :param smooth: length of smoothing window to use when decoding
        x and y (s)
    :param mad_max: maximum median absolute deviation of x or y pos to decode
        x, y during a smoothing window
    """
    # loop over all smoothing windows
    t_starts = np.arange(*wdw, smooth)
    pc_mask = rsp.cell_types == 'PC'
    
    ts = np.nan * np.zeros(len(t_starts))
    xs = np.nan * np.zeros(len(t_starts))
    ys = np.nan * np.zeros(len(t_starts))
    
    for ctr, t_start in enumerate(t_starts):
        t_mask = (t_start <= rsp.ts) & (rsp.ts < t_start + smooth)
        
        ts[ctr] = np.median(rsp.ts[t_mask])
        
        # get x & y place-field centers of cells that spiked
        pcs = rsp.spks[t_mask, :][:, pc_mask].nonzero()[1]
        
        if len(pcs):
            
            xs_ = rsp.pfcs[0, pc_mask][pcs]
            ys_ = rsp.pfcs[1, pc_mask][pcs]
            
            if med_abs_dev(xs_) < mad_max:
                xs[ctr] = np.median(xs_)
                
            if med_abs_dev(ys_) < mad_max:
                ys[ctr] = np.median(ys_)
                
    return ts, xs, ys


# AUXILIARY SEARCH FUNCTIONS

def validate(cfg, ctr):
    """Validate config module, raising exception if invalid."""
    
    # check all settings present
    required = [
        'P_RANGES', 'Q_JUMP', 'Q_NEW', 'SGM_RAND',
        'A_PREV', 'B_PREV_Y', 'B_PREV_U', 'B_PREV_K', 'B_PREV_S', 'B_PREV_SUM',
        'L_STEP', 'L_PHI', 'N_PHI',
        'A_PHI', 'B_PHI_Y', 'B_PHI_U', 'B_PHI_K', 'B_PHI_S', 'B_PHI_SUM',
        'U_TARG', 'K_TARG', 'S_TARG', 'ETA_U', 'ETA_K', 'ETA_S'
    ]
    
    if ctr == 0:
        required = ['SMLN_ID', 'START'] + required
        
    missing = []
    for setting in required:
        if not hasattr(cfg, setting):
            missing.append(setting)
            
    if missing:
        raise Exception(
            'CFG file missing the following settings: {}.'.format(missing))
    
    # validate P_RANGES
    if not np.all([len(p_range) == 2 for p_range in cfg.P_RANGES]):
        raise Exception('CFG P_RANGES must be two-element tuples.')
        
    required = [
        'AREA_H', 'AREA_W', 'RIDGE_Y', 
        'P_INH', 'RHO_PC', 'Z_PC', 'L_PC', 'W_A_PC_PC',
        'P_A_INH_PC', 'W_A_INH_PC', 'P_G_PC_INH', 'W_G_PC_INH', 'FR_EC'
    ]
    
    keys = [p_range[0] for p_range in cfg.P_RANGES]
    missing = []
    for key in required:
        if key not in keys:
            missing.append(key)
            
    if missing:
        raise Exception(
            'CFG.P_RANGES missing the following keys: {}.'.format(missing))
    
    for key, p_range in cfg.P_RANGES:
        if not isinstance(p_range, list) or (len(p_range) not in [1, 3]):
            raise Exception('"{}" range must be 1- or 3-elmnt tuple.'.format(key))
        if len(p_range) == 3 and (p_range[1] <= p_range[0]):
            raise Exception('UB must exceed LB for "{}" range.'.format(key))
    
    # validate START and FORCES
    forces_all = [cfg.START] 
    keys_all = ['START']
    
    if hasattr(cfg, 'FORCE'):
        if not np.all([isinstance(v, list) for v in cfg.FORCE.values()]):
            raise Exception('All forces be lists.')
            
        forces_all += sum([v for v in cfg.FORCE.values()], [])
        keys_all += sum([len(v) * [k] for k, v in cfg.FORCE.items()], [])
    
    for key, force in zip(keys_all, forces_all):
        
        if isinstance(force, dict):
            
            missing = []
            for key_ in required:
                if key_ not in force:
                    missing.append(key_)
                    
            if missing:
                raise Exception(
                    'Force "{}" missing keys: {}'.format(key, missing))
        else:
            
            if force not in ['center', 'random']:
                raise Exception(
                    'Force "{}": "{}" not understood.'.format(key, force))
    
    # validate int/float settings
    settings = [
        'Q_JUMP', 'Q_NEW', 'SGM_RAND',
        'A_PREV', 'B_PREV_Y', 'B_PREV_U', 'B_PREV_K', 'B_PREV_S', 'B_PREV_SUM',
        'L_STEP', 'L_PHI', 'N_PHI',
        'A_PHI', 'B_PHI_Y', 'B_PHI_U', 'B_PHI_K', 'B_PHI_S', 'B_PHI_SUM',
        'U_TARG', 'K_TARG', 'S_TARG', 'ETA_U', 'ETA_K', 'ETA_S'
    ]
    
    invalid = []
    for setting in settings:
        if not isinstance(getattr(cfg, setting), (int, float)):
            invalid.append(setting)
        elif not getattr(cfg, setting) >= 0:
            invalid.append(setting)
    
    if invalid:
        raise Exception('Settings {} must be non-neg. int/float.'.format(invalid))

    # validate beta normalization
    if not (cfg.B_PREV_Y + cfg.B_PREV_U + cfg.B_PREV_K + cfg.B_PREV_S) \
            == cfg.B_PREV_SUM:
        raise Exception('B_PREV components must add to B_PREV_SUM.')
        
    if not (cfg.B_PHI_Y + cfg.B_PHI_U + cfg.B_PHI_K + cfg.B_PHI_S) \
            == cfg.B_PHI_SUM:
        raise Exception('B_PHI components must add to B_PHI_SUM.')
        
    return True
    
    
def trial_to_p(trial):
    """Extract param dict from trial instance."""
    
    return {
        'AREA_H': trial.ridge_h,
        'AREA_W': trial.ridge_w,
        'RIDGE_Y': trial.ridge_y,
        'P_INH': trial.p_inh,
        'RHO_PC': trial.rho_pc,
        
        'Z_PC': trial.z_pc,
        'L_PC': trial.l_pc,
        'W_A_PC_PC': trial.w_a_pc_pc,
        
        'P_A_INH_PC': trial.p_a_inh_pc,
        'W_A_INH_PC': trial.w_a_inh_pc,
        
        'P_G_PC_INH': trial.p_g_pc_inh,
        'W_G_PC_INH': trial.w_g_pc_inh,
        
        'FR_EC': trial.fr_ec,
    }


def trial_to_rslt(trial):
    """Extract result dict from trial instance."""
    
    return {
        'STABILITY': trial.stability,
        'ANGLE': trial.angle,
        'ACTIVITY': trial.activity,
        'SPEED': trial.speed,
    }


def make_param_conversions(cfg):
    """
    Return functions for converting from p to x and x to p.
    :param cfg: config module containing P_RANGES attribute, which is:
        list of two-element tuples; first element is param name,
        second element is range, which can either be single-element list
        specifying fixed value, or three-element list giving lower bound, upper
        bound, and scale (approx. resolution).
    """
    
    def p_to_x(p):
        """Convert param dict to x."""
        assert len(p) == len(cfg.P_RANGES)
        
        x = np.nan * np.zeros(len(p))
        
        for ctr, (name, p_range) in enumerate(cfg.P_RANGES):
            
            if len(p_range) == 1:  # fixed val
                x[ctr] = 0
                
            else:
                # compute x from p and param range
                lb, ub, scale = p_range
                x[ctr] = scale * (p[name] - ((ub + lb)/2)) / (ub - lb)
                
        return x
    
    def x_to_p(x):
        """Convert x to param dict."""
        assert len(x) == len(cfg.P_RANGES)
        
        p = {}
        
        for ctr, (name, p_range) in enumerate(cfg.P_RANGES):
            
            if len(p_range) == 1:  # fixed val
                p[name] = p_range[0]
                
            else:
                # compute p from x and param range
                lb, ub, scale = p_range
                p[name] = (x[ctr] * (ub - lb) / scale) + ((lb + ub) / 2)
                
        return p
    
    return p_to_x, x_to_p


def force_to_x(cfg, f):
    """Convert force object to x location"""
    
    if isinstance(f, dict):  # f is param dict
        # convert param dict to x
        x = make_param_conversions(cfg)[0](f)
    elif f == 'random':
        x = sample_x_rand(cfg)
    elif f == 'center':
        x = np.zeros(len(cfg.P_RANGES))
        
    return x


def fix_x_if_out_of_bounds(cfg, x_cand):
    """Return a corrected x that is not out of bounds."""
    x_correct = []
    
    for x_, (_, p_range) in zip(x_cand, cfg.P_RANGES):
        
        if len(p_range) == 1:
            lb = 0
            ub = 0
        else:
            lb = -p_range[2] / 2
            ub = p_range[2] / 2
            
        if x_ < lb:
            x_correct.append(lb)
        elif x_ > ub:
            x_correct.append(ub)
        else:
            x_correct.append(deepcopy(x_))
    
    return np.array(x_correct)
            
    
def sample_x_rand(cfg):
    """Sample random x."""
    x = np.nan * np.ones(len(cfg.P_RANGES))
    
    # loop over all parameters
    for ctr, (_, p_range) in enumerate(cfg.P_RANGES):
        
        # fixed or random
        if len(p_range) == 1:  # fixed
            x[ctr] = 0
            
        else:  # random
            scale = p_range[-1]
            
            # keep sampling until valid param found
            while True:
                x_ = np.random.normal(0, scale*cfg.SGM_RAND)
                
                if -scale/2 <= x_ < scale/2:
                    break
                    
            x[ctr] = x_
            
    return x


def _sample_x_prev(cfg, ps, rslts):
    """Sample previously visited params."""
    
    # convert rslts to usable arrays
    temp = [
        (rslt['STABILITY'], rslt['ANGLE'], rslt['ACTIVITY'], rslt['SPEED'])
        for rslt in rslts
    ]
    y, u, k, s = np.array(temp).T
    
    # get distances to targets
    d_u = np.exp(-np.abs(u - cfg.U_TARG)/cfg.ETA_U)
    d_k = np.exp(-np.abs(k - cfg.K_TARG)/cfg.ETA_K)
    d_s = np.exp(-np.abs(s - cfg.S_TARG)/cfg.ETA_S)
    
    # calculate weights
    temp = cfg.B_PREV_Y*y + cfg.B_PREV_U*d_u \
        + cfg.B_PREV_K*d_k + cfg.B_PREV_S*d_s
    w = np.exp(cfg.A_PREV * temp / cfg.B_PREV_SUM)
    w /= w.sum()
    
    # sample idx of previous location
    idx = np.random.choice(len(ps), p=w)
    
    return ps[idx], idx

    
def sample_x_prev(cfg, session, p_to_x):
    """Sample previously visited x."""
    
    # get all past trials
    trials = session.query(d_models.EmbeddedTrial).join(
        d_models.EmbeddedSearcher).filter(
        d_models.EmbeddedSearcher.smln_id == cfg.SMLN_ID).all()
    
    # get param and rslt dicts
    ps = [trial_to_p(trial) for trial in trials]
    rslts = [trial_to_rslt(trial) for trial in trials]
    
    p = _sample_x_prev(cfg, ps, rslts)[0]
    
    return p_to_x(p)


def compute_phi_mean(cfg, xs, rslts):
    """Compute optimal direction when stepping."""
    
    # convert rslts to usable arrays
    temp = [
        (rslt['STABILITY'], rslt['ANGLE'], rslt['ACTIVITY'], rslt['SPEED'])
        for rslt in rslts
    ]
    y, u, k, s = np.array(temp).T

    # compute optimal direction

    ## get distances to targets
    d_u = np.exp(-np.abs(u - cfg.U_TARG)/cfg.ETA_U)
    d_k = np.exp(-np.abs(k - cfg.K_TARG)/cfg.ETA_K)
    d_s = np.exp(-np.abs(s - cfg.S_TARG)/cfg.ETA_S)

    ## calculate weights
    temp = cfg.B_PHI_Y*y + cfg.B_PHI_U*d_u \
        + cfg.B_PHI_K*d_k + cfg.B_PHI_S*d_s
    w = np.exp(cfg.A_PHI * temp / cfg.B_PHI_SUM)
    w /= w.sum()

    ## get xs for past results and dists to center of mass
    dxs = xs - xs.mean(0)

    ## take weighted sum of dxs
    dx_sum = dxs.T.dot(w)

    ## get optimal direction
    if np.linalg.norm(dx_sum) != 0:
        phi_best = dx_sum / np.linalg.norm(dx_sum)
    else:
        phi_best = np.zeros(dx_sum.shape)
    
    return cfg.L_PHI*phi_best


def sample_x_step(cfg, session, searcher, x_prev, since_jump, p_to_x):
    """Sample new x using stepping algorithm."""
    
    n = min(cfg.N_PHI, since_jump)
    
    # sample step size
    l = np.random.exponential(cfg.L_STEP)
    
    if n == 0:
        phi_mean = np.zeros(len(x_prev))
    
    else:
        # get past n trials with this searcher id
        trials = session.query(d_models.EmbeddedTrial).\
            filter_by(searcher_id=searcher.id).\
            order_by(d_models.EmbeddedTrial.id.desc()).limit(n).all()
            
        ## get params, xs, and measurables for past results
        ps = [trial_to_p(trial) for trial in trials]
        xs = np.array([p_to_x(p) for p in ps])
        
        # get results
        rslts = [trial_to_rslt(trial) for trial in trials]
        
        phi_mean = compute_phi_mean(cfg, xs, rslts)

    # sample final step direction
    phi_ = np.random.normal(phi_mean)
    phi = phi_/np.linalg.norm(phi_)
    
    # return final x
    return x_prev + l*phi


def save_embedded_trial(session, searcher, seed, p, rslt):
    
    trial = d_models.EmbeddedTrial(
        searcher_id=searcher.id,
        seed=seed,
        
        area_h=p['AREA_H'],
        area_w=p['AREA_W'],
        ridge_y=p['RIDGE_Y'],
        p_inh=p['P_INH'],
        rho_pc=p['RHO_PC'],
        
        z_pc=p['Z_PC'],
        l_pc=p['L_PC'],
        w_a_pc_pc=p['W_A_PC_PC'],
        
        p_a_inh_pc=p['P_A_INH_PC'],
        w_a_inh_pc=p['W_A_INH_PC'],
        
        p_g_pc_inh=p['P_G_PC_INH'],
        w_g_pc_inh=p['W_G_PC_INH'],
        
        fr_ec=p['FR_EC'],
        
        stability=rslt['STABILITY'],
        angle=rslt['ANGLE'],
        activity=rslt['ACTIVITY'],
        speed=rslt['SPEED'],
    )
    
    session.add(trial)
    session.commit()
