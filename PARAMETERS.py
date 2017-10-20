# PLACE-TUNING AND TRAJECTORY GENERATION

L_PL = 0.2  # (m)
R_MAX_PL = 350  # (Hz)

BOX_L = -1  # (m)
BOX_R = 1  # (m)
BOX_B = -0.5  # (m)
BOX_T = 0.5  # (m)

S_TRAJ = 0.2  # (m/s)
T_TRAJ = 1  # (s)

X_0 = -0.9  # (m)
Y_0 = -0.4  # (m)
VX_0 = 0.45  # (m/s)
VY_0 = 0.45  # (m/s)

# NETWORK ARCHITECTURE

N_PC = 1000
N_EC = 1000

P_A_PC_PC = 0.01
P_N_PC_PC = 0.01

P_A_PC_EC = 0.01
P_N_PC_EC = 0.01

W_A_PC_PC = 0.0085
W_N_PC_PC = 0

W_A_PC_EC = 0

W_N_PC_EC_I = 0.00065
W_N_PC_EC_F = 0.00135

W_A_PC_PL = 0.017
W_N_PC_PL = 0

W_G_PC_INH = 0.024

W_A_INH_PC = 0.013

# MEMBRANE POTENTIAL DYNAMICS

T_M_PC = 0.05  # (s)
V_REST_PC = -0.068  # (V)
V_TH_PC = -0.036  # (V)
V_RESET_PC = -0.068  # (V)

T_M_INH = 0.009  # (s)
V_REST_INH = -0.058  # (V)
V_TH_INH = -0.036  # (V)
V_RESET_INH = -0.058  # (V)

T_R = 0.002  # (s)

E_AHP = -0.07  # (V)

# SYNAPTIC CONDUCTANCE DYNAMICS

E_L_PC = -0.068  # (V)
E_L_INH = -0.058  # (V)

E_A = 0  # (V)
E_N = 0  # (V)
E_G = -0.08  # (V)

T_A = 0.002  # (s)
T_N = 0.08  # (s)
T_G = 0.005  # (s)

# EC-CA3 PLASTICITY
T_W = 1  # (s)
T_C = 1.5  # (s)
C_S = 5
B_C = 0.2

# SIMULATION

DT = 0.0005  # (s)
