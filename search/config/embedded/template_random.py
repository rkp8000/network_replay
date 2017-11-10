SMLN_ID = 'smln_0'

START = {
    'AREA_H': 0.25, 'AREA_W': 2, 'RIDGE_Y': 0., 'P_INH': 0.1, 'RHO_PC': 3000, 
    'Z_PC': 1, 'L_PC': 0.07, 'W_A_PC_PC': 0.0085,
    'P_A_INH_PC': 0.1, 'W_A_INH_PC': 0.009, 'P_G_PC_INH': 0.04, 'W_G_PC_INH': 0.003,
    'FR_EC': 35
}

# ranges can be:
#     3-element list, where first elements are lb, ub,
#     and final element is scale (resolution)
#
#     1-element list specifying fixed param value

P_RANGES = (
    ('AREA_H', [.25]),
    ('AREA_W', [2.]),
    ('RIDGE_Y', [0.]),
    ('P_INH', [.05, .15, 1]),
    ('RHO_PC', [2000., 4000., 1]),
        
    ('Z_PC', [.7, 1.3, 1]),
    ('L_PC', [0., .1, 1]),
    ('W_A_PC_PC', [0, .012, 1]),

    ('P_A_INH_PC', [.06, .12, 1]),
    ('W_A_INH_PC', [0., .01, 1]),

    ('P_G_PC_INH', [.02, .1, 1]),
    ('W_G_PC_INH', [.001, .02, 1]),

    ('FR_EC', [30, 45, 1]),
)

Q_JUMP = 1

Q_NEW = 1

SGM_RAND = 0.2

A_PREV = 5
B_PREV_Y = 1
B_PREV_U = 5
B_PREV_K = 0
B_PREV_S = 5
B_PREV_SUM = B_PREV_Y + B_PREV_U + B_PREV_K + B_PREV_S

L_STEP = 0.002
L_PHI = 1
N_PHI = 30

A_PHI = 5
B_PHI_Y = 0
B_PHI_U = 5
B_PHI_K = 5
B_PHI_S = 5
B_PHI_SUM = B_PHI_Y + B_PHI_U + B_PHI_K + B_PHI_S

U_TARG = 0
K_TARG = 1
S_TARG = 8

ETA_U = 20
ETA_K = 10
ETA_S = 1
