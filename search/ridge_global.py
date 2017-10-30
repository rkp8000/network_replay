"""
GLOBAL CONFIG FOR RIDGE SEARCH AND OBJECTIVE FUNCTION.
"""
import numpy as np


# PRE-COMPUTATION CONFIG

SEED_PRE = 0


## W_N_PC_EC VS DIST COMPUTATION

PATH_W_N_PC_EC_VS_DIST = 'search/pre/ridge/w_n_pc_ec_vs_dist.npy'

DUR_W_PRE = 5
STIM_W_PRE = (0, 2)
N_TRIALS_W_PRE = 200
DIST_PRE = np.linspace(0, 0.3, 300)

## V, G_N VS W_N_PC_EC, RATE_EC COMPUTATION

PATH_V_G_N_VS_W_N_PC_EC_RATE_EC = 'search/pre/ridge/v_g_n_vs_w_n_pc_ec_rate_ec.npy'

DUR_V_G_PRE = 21
MEASURE_START_V_G_PRE = 1
N_TIMEPOINTS_V_G_PRE = 1000

W_N_PC_EC_PRE = np.linspace(0, 0.002, 200)
RATE_EC_PRE = np.linspace(20, 70, 100)


# OBJECTIVE FUNCTION CONFIG

N_NTWKS = 3
SMLN_DUR = 0.2  # (s)
MAX_RUNS_STABILIZE = 10

T_STIM = 0.001  # (s)

MIN_BKGD_PC_FR_SGMS = 3  # minimum stds above bkgd fr to be considered nonzero
PRPGN_WDW = 0.01  # (s)

WAVE_TOL = 0.25
LOOK_BACK_X = 2

N_L_PC_FORCE = 2

DECAY_CHECK = [(0, .2), (.7, .9)]
DECAY_MIN = 0.95

ACTIVITY_READ = (.1, .9)
SPEED_READ = (.1, .9)


# SEARCH CONFIG

CONFIG_ROOT = 'search.config.ridge'
MAX_SEED = 10000
WAIT_AFTER_ERROR = 5
