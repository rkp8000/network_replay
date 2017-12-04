"""
Configuration file for full linear ridge simulations.
"""
import numpy as np

CACHE_DIR = 'lin_ridge/cache/full'

MAX_SEED = 100000
REPORT_EVERY = 30
STORE = {
    'vs': np.float16,
    'spks': bool
    'gs': None,
    'g_ahp': None,
    'ws_plastic': None,
    'cs': None
}

TRAJ_START_T = 0.1
TRAJ_END_T = 13.4333

T_EC = 14.
T_REPLAY = 14.5

ITVL_REPLAY = 0.4
N_REPLAY = 10

PL_TRIGGER_FR_INCMT = 1
PL_TRIGGER_FR_MAX = 20

PL_TRIGGER_RUN_TIME = 0.005  # s
PL_TRIGGER_TEST_RUN_TIME = 0.03 # s
MIN_SPK_CT_FACTOR_TRIGGER_TEST = 2

N_STRIPES = 10
