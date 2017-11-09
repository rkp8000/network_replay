SMLN_ID = 'test_random'

START = 'center'

# ranges can be:
#     3-element list, where first elements are lb, ub,
#     and final element is scale (resolution)
#
#     1-element list specifying fixed param value

P_RANGES = [
    ('AREA_H', [0, 10, 1]),
    ('AREA_W', [0, 10, 1]),
    ('RIDGE_Y', [0, 10, 1]),
    
    ('RHO_PC', [0, 10, 1]),
    ('P_INH', [0, 10, 1]),
    
    ('Z_PC', [0, 10, 1]),
    ('L_PC', [10, 30, 1]),
    ('W_A_PC_PC', [10, 30, 1]),
    
    ('P_A_INH_PC', [10, 30, 1]),
    ('W_A_INH_PC', [4]),

    ('P_G_PC_INH', [5]),
    ('W_G_PC_INH', [6]),
    
    ('FR_EC', [9]),
]

Q_JUMP = 1

Q_NEW = 1

SGM_RAND = 0.3

A_PREV = 1
B_PREV_Y = 1
B_PREV_U = 1
B_PREV_K = 0
B_PREV_S = 1
B_PREV_SUM = B_PREV_Y + B_PREV_U + B_PREV_K + B_PREV_S

L_STEP = 0.1
L_PHI = 1
N_PHI = 5

A_PHI = 5
B_PHI_Y = 0
B_PHI_U = 5
B_PHI_K = 5
B_PHI_S = 5
B_PHI_SUM = B_PHI_Y + B_PHI_U + B_PHI_K + B_PHI_S

U_TARG = 9
K_TARG = 9
S_TARG = 9

ETA_U = 50
ETA_K = 50
ETA_S = 50
