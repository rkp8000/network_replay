SMLN_ID = 'smln_2'

START = {
    'FR_EC': 37.6806786512987,
    'L_PC': 0.02706,
    'P_A_INH_PC': 0.0829937389666363,
    'P_G_PC_INH': 0.04156,
    'P_INH': 0.105712005676233,
    'RHO_PC': 10507.7466,
    'AREA_H': 0.25,
    'AREA_W': 2.0,
    'RIDGE_Y': 0.,
    'W_A_INH_PC': 0.00213977444725152,
    'W_A_PC_PC': 0.00684639671173546,
    'W_G_PC_INH': 0.0147915757807821,
    'Z_PC': 1.09118462967164
}

FORCE = {
    -1: 20 * [{
        'FR_EC': 37.1540719080865,
        'L_PC': 0.0256167529633419,
        'P_A_INH_PC': 0.0837256760217976,
        'P_G_PC_INH': 0.0414177348411754,
        'P_INH': 0.0996587153033559,
        'RHO_PC': 11578.0671230118,
        'AREA_H': 0.25,
        'AREA_W': 2.0,
        'RIDGE_Y': 0.,
        'W_A_INH_PC': 0.00235735925373169,
        'W_A_PC_PC': 0.00692566504489662,
        'W_G_PC_INH': 0.0152438151476479,
        'Z_PC': 1.08675762262723
    }],
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
    ('P_INH', [.08, .14, 1]),
    ('RHO_PC', [10000., 12000., 1]),
        
    ('Z_PC', [1., 1.3, 1]),
    ('L_PC', [0., .03, 1]),
    ('W_A_PC_PC', [0, .01, 1]),

    ('P_A_INH_PC', [.06, .1, 1]),
    ('W_A_INH_PC', [0., .005, 1]),

    ('P_G_PC_INH', [.03, .06, 1]),
    ('W_G_PC_INH', [.015, .024, 1]),

    ('FR_EC', [35, 45, 1]),
)

Q_JUMP = 0.001

Q_NEW = 0.

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
S_TARG = 5

ETA_U = 20
ETA_K = 10
ETA_S = 1
