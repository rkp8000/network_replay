import numpy as np
import time

from p_ranges import p_ranges, STD
from s_params import s_params, apxn
import smln


def search(wait=None):

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
    search()
