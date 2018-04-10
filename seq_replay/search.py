import numpy as np
import time
import sys

from p_ranges import p_ranges, STD
from s_params import s_params, apxn
import smln


def search(group, commit, wait=None):

    while True:
        p = sample_params()
        
        # run smln
        rslt = smln.run(p=p, s=s_params, apxn=apxn)
        smln.save(rslt)
        
        if wait:
            time.sleep(wait)


def sample_params():
    p = {}
    
    # loop over all items in p_ranges, using
    # params directly if scalars, or sampling if 
    # [lb, ub] are given
    
    for k, v in p_ranges.items():
        
        if not isinstance(v, list):
            p[k] = v
        else:
            # sample from within lb, ub
            lb, ub = v
            x = np.clip(STD * np.random.randn(), -1, 1)
            
            # convert x \in [-1, 1] to param val
            p[k] = ((ub - lb)/2) * x + ((ub + lb)/2)
            
    return p
        

if __name__ == '__main__':
    
    if not (len(sys.argv) - 1) in range(2, 4):
        raise Exception('2 or 3 arguments required.')
        
    group = sys.argv[1]
    commit = sys.argv[2]
    wait = int(sys.argv[3]) if len(sys.argv) == 4 else None
    
    search(group, commit, wait)
