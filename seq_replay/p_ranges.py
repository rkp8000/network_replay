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
    'T_RP_PC': 0.02,

    'T_M_INH': P.T_M_INH,
    'E_L_INH': P.E_L_INH,
    'V_TH_INH': P.V_TH_INH,
    'V_R_INH': P.V_R_INH,
    'T_RP_INH': P.T_RP_INH,

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
    'L_PL': ?,
    'R_MAX': ?,
    'S_TH': ?,
    'B_S': ?,
    
    # PL --> PC connectivity
    'W_E_PC_PL': [?, ?],
    'S_E_PC_PL': ?,

    # STATE --> PC connectivity
    'W_E_INIT_PC_ST': [?, ?],
    'S_E_INIT_PC_ST': ?,
    
    # plasticity
    'T_C': P.T_C,
    'T_W': P.T_W,
    'A_P': P.A_P,
    'C_S': P.C_S,
    'B_C': P.B_C,
    
    # PC --> PC connectivity
    'N_PC': ?,
    'L_PC_PC': [?, ?],
    'Z_PC_PC': [?, ?],
    'W_E_PC_PC': [?, ?],
    'S_E_PC_PC': ?,
    
    # PC --> INH connectivity
    'N_INH': ?,
    'Z_INH_PC': ?,
    'L_INH_PC': ?,
    'W_E_INH_PC': ?,
    'S_E_INH_PC': ?,
    
    # INH --> PC connectivity
    'L_C_PC_INH': ?,
    'Z_C_PC_INH': ?,
    'L_S_PC_INH': ?,
    'Z_S_PC_INH': ?,
    'W_I_PC_INH': ?,
    'S_I_PC_INH': ?,
    
    # ST --> PC inputs
    'R_TRJ_PC_ST': ?,
    'R_RPL_PC_ST': ?,
    
    # replay trigger
    'D_T_TR': ?,
    'A_TR': ?,
    'X_TR': ?,
    'Y_TR': ?,
    'R_TR': [?, ?],
}

STD = 1
