SMLN_ID = 'test'

START = {
    'RIDGE_H': 0.25, 'RIDGE_W': 4, 'RHO_PC': 1000,
    'Z_PC': 0.8, 'L_PC': 0.2, 'W_A_PC_PC': 0.01,
    'P_A_INH_PC': 0.1, 'W_A_INH_PC': 0.017,
    'P_G_PC_INH': 0.1, 'W_G_PC_INH': 0.005,
    'W_N_PC_EC_I': 0.0008, 'RATE_EC': 50,
}

FORCE = {
    0: ['center', 'random', 'center', 'random'],
    1: ['center', 'center', 'center', 'center'],
}

# ranges can be:
#     3-element list, where first elements are lb, ub,
#     and final element is scale (resolution)
#
#     1-element list specifying fixed param value

P_RANGES = (
    ('RIDGE_H', [0.25]),
    ('RIDGE_W', [4]),
    
    ('RHO_PC', [500, 2000, 1]),
        
    ('Z_PC', [0.2, 1.6, 1]),
    ('L_PC', [0.02, .3, 1]),
    ('W_A_PC_PC', [0.002, 0.02, 1]),

    ('P_A_INH_PC', [0.05, 0.2, 1]),
    ('W_A_INH_PC', [0.002, 0.02, 1]),

    ('P_G_PC_INH', [0.05, 0.2, 1])
    ('W_G_PC_INH', [0, 0.01, 1]),

    ('W_N_PC_EC_I', [0.0002, 0.001, 1]),
    ('RATE_EC', [30, 60, 1]),
)

Q_JUMP = 1

Q_NEW = 1

SGM_RAND = 0.1

A_PREV = 1
B_PREV_Y = 1
B_PREV_K = 0
B_PREV_S = 1
B_PREV_SUM = B_PREV_Y + B_PREV_K + B_PREV_S

L_STEP = 0.05
L_PHI = 1
N_PHI = 20

A_PHI = 1
B_PHI_Y = 1
B_PHI_K = 0
B_PHI_S = 1
B_PHI_SUM = B_PHI_Y + B_PHI_K + B_PHI_S

K_TARG = 1
S_TARG = 8

ETA_K = 1
ETA_S = 1
