"""
model params
params bound to P are fixed
params given as scalars are "frozen"
params given as [lb, ub] lists are variable and will be searched over
"""

import PARAMS as P


p_ranges = {
    # membrane potential
    'T_M_PC': P.T_M_PC,
    'E_L_PC': P.E_L_PC,
    'V_TH_PC': P.V_TH_PC,
    'V_R_PC': P.V_R_PC,
    'T_R_PC': 0.02,

    'T_M_INH': P.T_M_INH,
    'E_L_INH': P.E_L_INH,
    'V_TH_INH': P.V_TH_INH,
    'V_R_INH': P.V_R_INH,
    'T_R_INH': P.T_R_INH,

    # synaptic params
    'E_E': P.E_E,
    'E_I': P.E_I,
    'T_E': P.T_E,
    'T_I': P.T_I,

    # AHP
    'T_AHP_PC': P.T_AHP_PC,
    'E_AHP_PC': P.E_AHP_PC,
    'W_AHP_PC': P.W_AHP_PC,

    # place-tuning
    'L_PL': .2,
    'R_MAX': 600,
    'S_TH': .1,
    'B_S': .01,
    
    # PL --> PC connectivity
    'W_E_PC_PL': [.02, .02],
    'S_E_PC_PL': 0,

    # STATE --> PC connectivity
    'W_E_INIT_PC_ST': [.004, .013],
    'S_E_INIT_PC_ST': 0,
    
    # plasticity
    'T_C': P.T_C,
    'T_W': P.T_W,
    'A_P': P.A_P,
    'C_S': P.C_S,
    'B_C': P.B_C,
    
    # PC --> PC connectivity
    'N_PC': 1000,
    'L_PC_PC': [.05, .12],
    'Z_PC_PC': [.9, 1.5],
    'W_E_PC_PC': [.01, .03],
    'S_E_PC_PC': 0,
    
    # PC --> INH connectivity
    'N_INH': 100,
    'L_INH_PC': [.03, .09],
    'Z_INH_PC': [.9, 1.5],
    'W_E_INH_PC': [.004, .02],
    'S_E_INH_PC': 0,
    
    # INH --> PC connectivity
    'L_C_PC_INH': [.03, .07],
    'Z_C_PC_INH': [2.5, 3.5],
    'L_S_PC_INH': [.06, .12],
    'Z_S_PC_INH': [2.5, 3.5],
    'W_I_PC_INH': 0,  # [.004, .025],
    'S_I_PC_INH': 0,
    
    # ST --> PC inputs
    'FR_TRJ_PC_ST': 3,
    'FR_RPL_PC_ST': [110, 150],
    
    # replay trigger
    'D_T_TR': [.004, .008],
    'A_TR': [.008, .01],
    'R_TR': .4,
}

STD = .5
