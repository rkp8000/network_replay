SMLN_ID = 'test_guided'

START = {
    'RIDGE_H': 1, 'RIDGE_W': 2, 'RHO_PC': 1,
    'Z_PC': 2, 'L_PC': 1, 'W_A_PC_PC': 2,
    'P_A_INH_PC': 1, 'W_A_INH_PC': 2,
    'P_G_PC_INH': 3, 'W_G_PC_INH': 5,
    'RATE_EC': 9,
}

# ranges can be:
#     3-element list, where first elements are lb, ub,
#     and final element is scale (resolution)
#
#     1-element list specifying fixed param value

P_RANGES = (
    ('RIDGE_H', [0, 10, 1]),
    ('RIDGE_W', [0, 10, 1]),
    
    ('RHO_PC', [0, 10, 1]),
        
    ('Z_PC', [0, 10, 5]),
    ('L_PC', [0, 10, 5]),
    ('W_A_PC_PC', [0, 10, 1]),

    ('P_A_INH_PC', [0, 10, 1]),
    ('W_A_INH_PC', [0, 10, 1]),

    ('P_G_PC_INH', [3]),
    ('W_G_PC_INH', [5]),

    ('RATE_EC', [9]),
)

Q_JUMP = 0

Q_NEW = 1

SGM_RAND = 0.1

A_PREV = 1
B_PREV_Y = 1
B_PREV_K = 0
B_PREV_S = 1
B_PREV_SUM = B_PREV_Y + B_PREV_K + B_PREV_S

L_STEP = 0.1
L_PHI = 1
N_PHI = 5

A_PHI = 5
B_PHI_Y = 0
B_PHI_K = 5
B_PHI_S = 5
B_PHI_SUM = B_PHI_Y + B_PHI_K + B_PHI_S

K_TARG = 9
S_TARG = 9

ETA_K = 50
ETA_S = 50
