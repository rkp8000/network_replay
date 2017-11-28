"""
Configuration file for full linear ridge simulations.
"""

CACHE_DIR = 'lin_ridge/cache/full'

PATH_W_N_PC_EC_VS_DIST = 'lin_ridge/rslts_pre/w_n_pc_ec_vs_dist.npy'
PATH_V_G_N_VS_W_N_PC_EC_FR_EC = 'lin_ridge/rslts_pre/v_g_n_vs_w_n_pc_ec_fr_ec.npy'

MAX_SEED = 100000

TRAJ_START_X = 1.
TRAJ_END_X = -1.

TRAJ_START_Y = -0.125
TRAJ_END_Y = -0.125

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
