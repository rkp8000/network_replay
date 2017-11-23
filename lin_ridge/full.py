"""
Code for running full linear ridge simulations.
"""


def run_smln():
    """
    takes in LinRidgeTrial/ID, LinRidgeFullTrial/ID, and optional 
    temp file, then:
    
    Gets params from LinRidgeTrial.
    
    Checks for temp file containing PC-PC recurrent cxns 
    and replay triggering input and loads it if found, 
    otherwise runs LinRidgeTrial to extract ntwk, calls 
    make_replay_trigger, and saves PC-PC cxns and replay trigger in temp file.
    
    Loads PC-PC rcr cxns and replay trigger from temp file.
    
    Calls p_to_ntwk_plastic to make ntwk from LinRidgeTrial,
    builds full DG→PC and EC→PC stim sequences, presents
    them to ntwk, quantifies replay epoch activation, saves
    trial info to database, outputs raster plot if desired,
    and returns ntwk response object.
    """
    pass


def p_to_ntwk_plastic():
    """
    takes in LinRidgeTrial param dict, plasticity params,
    and optional PC-PC recurrent connectivity matrix and 
    builds ntwk with PC-activity-driven EC→PC plasticity"""
    pass


def make_replay_trigger():
    """
    takes replay-only ntwk, corresponding params and initial
    conditions, forced_spk PC idxs, builds EC stim, and 
    determines an input to the forced PCs sufficient to 
    elicit replay under EC activation
    """
    pass
