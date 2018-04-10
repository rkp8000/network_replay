"""
smln params
"""

s_params = {
    'RNG_SEED': 0,
    'DT': 0.0005,
    
    # trajectory
    'BOX_W': 2,
    'BOX_H': 2,
    
    'START_X': -1,
    'START_Y': .75,
    'TURN_X': 0,
    'TURN_Y': -.75,
    'END_X': -1,
    
    'SPEED': .15,
    
    'schedule': {
        'SMLN_DUR': 22,
        'TRJ_START_T': 1,
        'REPLAY_EPOCH_START_T': 20,
        'TRG_START_T': 21,
    },
    
    'metrics': {
        'RADIUS': 0.2,
        'PITCH': 10,
        'MIN_SCALE_TRJ': 1.25,
        'WDW': 0.1,
        'MIN_FRAC_SPK_TRJ': .75,
        'MAX_AVG_SPK_CT_TRJ': 3,
        'TRJ_NON_TRJ_SPK_RATIO': .75,
    },
}
