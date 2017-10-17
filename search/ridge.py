from copy import deepcopy
from datetime import datetime, timedelta
import importlib
import numpy as np
import threading
import time
import traceback

from db import make_session, d_models


# CONFIGURATION

CONFIG_ROOT = 'search.config.ridge'
MAX_SEED = 10000
WAIT_AFTER_ERROR = 5


# SEARCH FUNCTIONS

def launch_searchers(role, obj, n, P, max_iter=10, seed=None, commit=None):
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
                role, obj, P, max_iter=max_iter, seed=seed, commit=commit)
        
        # launch searcher daemon
        thread = threading.Thread(target=targ)
        thread.daemon = True
        thread.start()
        
        threads.append(thread)
        
    print('\n{} searchers launched.'.format(n))
        
    return threads
    
    
def search(
        role, obj, P, max_iter=10000, seed=None,
        config_root=CONFIG_ROOT, commit=None, verbose=False):
    """
    Launch instance of searcher exploring potentiated ridge trials.
    
    :param role: searcher role, which should correspond to search config file
    :param obj: objective function
    :param P: general project parameters module
    :param max_iter: max number of search iterations before quitting
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
    cfg = importlib.import_module('.'.join([config_root, role]))
    
    # connect to db
    session = make_session()
    
    # make new searcher
    searcher = d_models.RidgeSearcher(
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
    for ctr in range(max_iter):
        
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
            seed_ = np.random.randint(MAX_SEED)
            
            rslt = obj(p_cand, seed_)
            
            save_ridge_trial(
                session=session,
                searcher=searcher,
                seed=seed_,
                p=p_cand,
                rslt=rslt)
            
            # move to x_cand if propagation occurred else x_prev
            if rslt['PROPAGATION']:
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
            time.sleep(WAIT_AFTER_ERROR)
    
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
    searchers = session.query(d_models.RidgeSearcher).filter(
        d_models.RidgeSearcher.last_active >= earliest)
    
    if smln_id is not None:
        searchers = searchers.filter(d_models.RidgeSearcher.smln_id == smln_id)
    if role is not None:
        searchers = searchers.filter(d_models.RidgeSearcher.role == role)
    
    session.close()
    
    # check for searchers with error messages
    suspended = []
    errors = []
    
    for searcher in searchers:
        if searcher.error is not None:
            suspended.append(searcher.id)
            errors.append(searcher.error)
    
    # print results
    print('{} searchers active in last {} s.\n'.format(searchers.count(), recent))
    print('The following searchers were suspended by errors:')
    
    for suspended_, error in zip(suspended, errors):
        print('{}: ERROR: "{}".'.format(suspended_, error))
        
    print('\nLook up error tracebacks using read_search_error(id).\n')


def read_search_error(searcher_id):
    """Print error traceback for specific searcher_id."""
    
    # get searcher
    session = make_session()
    searcher = session.query(d_models.RidgeSearcher).get(searcher_id)
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
    

# RIDGE-TRIAL-SPECIFIC OBJECTIVE FUNCTION AND HELPER

def activity_and_speed(rsp):
    """Calculate propagation activity level and speed from ntwk response."""
    # identify start and ending window
    pass


def ntwk_obj(p, seed, return_rsps=False):
    """
    Run a trial given dict of param values and return dict of results.
    
    :param p: dict of params, which must include:
        RIDGE_H
        RIDGE_W
        
        RHO_PC
        
        Z_PC
        L_PC
        W_A_PC_PC
        
        P_INH_PC
        W_A_INH_PC
        
        P_PC_INH
        W_G_PC_INH
        
        W_A_PC_EC_I
        RATE_EC
    :param seed: random seed, so that trial is reproducible
    :param return_responses: whether to return list of ntwk responses
        as additional return variable
        
    :return: dict of measured vals for:
        PROPAGATION
        ACTIVITY
        SPEED
    """
    np.random.seed(seed)
    
    # loop over ntwks
    propagations = []
    activities = []
    speeds = []
    
    rsps = []
    
    for n_ctr in range(N_NTWKS):
        
        # make ntwk
        
        # get baseline activity level from EC input alone
        
        # sample upstream spks from EC
        
        # compute max forced spks
        
        vs_forced = ...
        spks_forced = ...
        
        # loop over repeats until steady state is reached
        rsps_ = []
        while True:
            
            # run ntwk with forced spks and vs
            rsp = ntwk.run(...)
            
            # attach place fields and cell types
            
            rsps_.append(rsp)
        
        # calculate activity and speed from last ntwk response
        activity, speed = activity_and_speed(rsp)
        
        # store results
        propagations.append(0 if activity == 0 else 1) 
        activities.append(activity)
        speeds.append(speed)
        
        rsps.append(rsps_)
        
    rslts = {
        'PROPAGATION': np.mean(propagations),
        'ACTIVITY': np.mean(activities),
        'SPEED': np.mean(speeds)
    }
    
    return rslts if not return_responses else rslts, rsps


# AUXILIARY FUNCTIONS

def validate(cfg, ctr):
    """Validate config module, raising exception if invalid."""
    
    # check all settings present
    required = [
        'P_RANGES', 'Q_JUMP', 'Q_NEW', 'SGM_RAND',
        'A_PREV', 'B_PREV_Y', 'B_PREV_K', 'B_PREV_S', 'B_PREV_SUM',
        'L_STEP', 'L_PHI', 'N_PHI',
        'A_PHI', 'B_PHI_Y', 'B_PHI_K', 'B_PHI_S', 'B_PHI_SUM',
        'K_TARG', 'S_TARG', 'ETA_K', 'ETA_S'
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
        'RIDGE_H', 'RIDGE_W', 'RHO_PC', 'Z_PC', 'L_PC', 'W_A_PC_PC',
        'P_A_INH_PC', 'W_A_INH_PC', 'P_G_PC_INH', 'W_G_PC_INH',
        'W_N_PC_EC_I', 'RATE_EC'
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
            raise Exception('"{}" range must be 1- or 3-element tuple.'.format(key))
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
        'A_PREV', 'B_PREV_Y', 'B_PREV_K', 'B_PREV_S', 'B_PREV_SUM',
        'L_STEP', 'L_PHI', 'N_PHI',
        'A_PHI', 'B_PHI_Y', 'B_PHI_K', 'B_PHI_S', 'B_PHI_SUM',
        'K_TARG', 'S_TARG', 'ETA_K', 'ETA_S'
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
    if not (cfg.B_PREV_Y + cfg.B_PREV_K + cfg.B_PREV_S) == cfg.B_PREV_SUM:
        raise Exception('B_PREV components must add to B_PREV_SUM.')
        
    if not (cfg.B_PHI_Y + cfg.B_PHI_K + cfg.B_PHI_S) == cfg.B_PHI_SUM:
        raise Exception('B_PHI components must add to B_PHI_SUM.')
        
    return True
    
    
def trial_to_p(trial):
    """Extract param dict from trial instance."""
    
    return {
        'RIDGE_H': trial.ridge_h,
        'RIDGE_W': trial.ridge_w,
        
        'RHO_PC': trial.rho_pc,
        
        'Z_PC': trial.z_pc,
        'L_PC': trial.l_pc,
        'W_A_PC_PC': trial.w_a_pc_pc,
        
        'P_A_INH_PC': trial.p_a_inh_pc,
        'W_A_INH_PC': trial.w_a_inh_pc,
        
        'P_G_PC_INH': trial.p_g_pc_inh,
        'W_G_PC_INH': trial.w_g_pc_inh,
        
        'W_N_PC_EC_I': trial.w_n_pc_ec_i,
        'W_RATE_EC': trial.rate_ec,
    }


def trial_to_rslt(trial):
    """Extract result dict from trial instance."""
    
    return {
        'PROPAGATION': trial.propagation,
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
        (rslt['PROPAGATION'], rslt['ACTIVITY'], rslt['SPEED'])
        for rslt in rslts
    ]
    y, k, s = np.array(temp).T
    
    # get distances to targets
    d_k = np.exp(-np.abs(k - cfg.K_TARG)/cfg.ETA_K)
    d_s = np.exp(-np.abs(s - cfg.S_TARG)/cfg.ETA_S)
    
    # calculate weights
    temp = cfg.B_PREV_Y*y + cfg.B_PREV_K*d_k + cfg.B_PREV_S*d_s
    w = np.exp(cfg.A_PREV * temp / cfg.B_PREV_SUM)
    w /= w.sum()
    
    # sample idx of previous location
    idx = np.random.choice(len(ps), p=w)
    
    return ps[idx], idx

    
def sample_x_prev(cfg, session, p_to_x):
    """Sample previously visited x."""
    
    # get all past trials
    trials = session.query(d_models.RidgeTrial).join(
        d_models.RidgeSearcher).filter(
        d_models.RidgeSearcher.smln_id == cfg.SMLN_ID).all()
    
    # get param and rslt dicts
    ps = [trial_to_p(trial) for trial in trials]
    rslts = [trial_to_rslt(trial) for trial in trials]
    
    p = _sample_x_prev(cfg, ps, rslts)[0]
    
    return p_to_x(p)


def compute_phi_mean(cfg, xs, rslts):
    """Compute optimal direction when stepping."""
    
    # convert rslts to usable arrays
    temp = [
        (rslt['PROPAGATION'], rslt['ACTIVITY'], rslt['SPEED'])
        for rslt in rslts
    ]
    y, k, s = np.array(temp).T

    # compute optimal direction

    ## get distances to targets
    d_k = np.exp(-np.abs(k - cfg.K_TARG)/cfg.ETA_K)
    d_s = np.exp(-np.abs(s - cfg.S_TARG)/cfg.ETA_S)

    ## calculate weights
    temp = cfg.B_PHI_Y*y + cfg.B_PHI_K*d_k + cfg.B_PHI_S*d_s
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
        trials = session.query(d_models.RidgeTrial).\
            filter_by(searcher_id=searcher.id).\
            order_by(d_models.RidgeTrial.id.desc()).limit(n).all()
            
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


def save_ridge_trial(session, searcher, seed, p, rslt):
    
    trial = d_models.RidgeTrial(
        searcher_id=searcher.id,
        seed=seed,
        
        ridge_h=p['RIDGE_H'],
        ridge_w=p['RIDGE_W'],
        
        rho_pc=p['RHO_PC'],
        
        z_pc=p['Z_PC'],
        l_pc=p['L_PC'],
        w_a_pc_pc=p['W_A_PC_PC'],
        
        p_a_inh_pc=p['P_A_INH_PC'],
        w_a_inh_pc=p['W_A_INH_PC'],
        
        p_g_pc_inh=p['P_G_PC_INH'],
        w_g_pc_inh=p['W_G_PC_INH'],
        
        w_n_pc_ec_i=p['W_N_PC_EC_I'],
        rate_ec=p['RATE_EC'],
        
        propagation=rslt['PROPAGATION'],
        activity=rslt['ACTIVITY'],
        speed=rslt['SPEED'],
    )
    
    session.add(trial)
    session.commit()
