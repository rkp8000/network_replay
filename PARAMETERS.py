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

N_CA3 = 1000
N_EC = 1000

P_A_CA3_CA3 = 0.01
P_N_CA3_CA3 = 0.01

P_A_EC_CA3 = 0.01
P_N_EC_CA3 = 0.01

W_A_CA3_CA3 = 0
W_N_CA3_CA3 = 0

W_A_EC_CA3_I = 0.008
W_A_EC_CA3_F = 0.017

W_N_EC_CA3 = 0

W_A_PL_CA3 = 0.017
W_N_PL_CA3 = 0

# MEMBRANE POTENTIAL DYNAMICS

V_REST = -0.068  # (V)
V_TH = -0.036  # (V)
V_RESET = -0.068  # (V)
T_M = 0.05  # (s)
T_R = 0.002  # (s)

# SYNAPTIC CONDUCTANCE DYNAMICS

E_LEAK = -0.068  # (V)
E_AMPA = 0  # (V)
E_NMDA = 0  # (V)
E_GABA = -0.08  # (V)

T_AMPA = 0.002  # (V)
T_NMDA = None
T_GABA = None

# EC-CA3 PLASTICITY
T_W = 1  # (s)
T_C = 1.5  # (s)
C_S = 5
BETA_C = 0.2

# SIMULATION

T_TRAJ_START = 1  # (s)
T_TRAJ_END = 11  # (s)
DT = 0.0005  # (s)
