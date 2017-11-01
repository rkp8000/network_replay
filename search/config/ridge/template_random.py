SMLN_ID = 'smln_0'

START = 'random'

# ranges can be:
#     3-element list, where first elements are lb, ub,
#     and final element is scale (resolution)
#
#     1-element list specifying fixed param value

P_RANGES = (
    ('RIDGE_H', [.25]),
    ('RIDGE_W', [4.]),
    ('P_INH', [0., .2, 1]),
    ('RHO_PC', [100., 2100., 1]),
        
    ('Z_PC', [0.1, 2, 1]),
    ('L_PC', [0., .5, 1]),
    ('W_A_PC_PC', [0, .02, 1]),

    ('P_A_INH_PC', [0., .2, 1]),
    ('W_A_INH_PC', [0. .02, 1]),

    ('P_G_PC_INH', [0., .2, 1]),
    ('W_G_PC_INH', [0., .02, 1]),

    ('FR_EC', [20, 70, 1]),
)

Q_JUMP = 1

Q_NEW = 1

SGM_RAND = 0.2

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
B_PHI_K = 0
B_PHI_S = 5
B_PHI_SUM = B_PHI_Y + B_PHI_K + B_PHI_S

K_TARG = 24
S_TARG = 8

ETA_K = 150
ETA_S = 50
