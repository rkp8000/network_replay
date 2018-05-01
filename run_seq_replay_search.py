"""
Main search code executable.

Call using:

$ python search.py <group_name> <commit id> [<wait_time>]
"""

import numpy as np
import time
import sys
from sys import stdout

from seq_replay.p_ranges import p_ranges, STD
from seq_replay.s_params import s_params
from seq_replay import smln


def search(group, commit, wait=None):

    ctr = 0
    
    while True:
        if ctr % 50 == 0:
            stdout.write('\n{}'.format(ctr))
            stdout.flush()
        
        np.random.seed()
        p = sample_params()
        
        # run smln
        rslt = smln.run(p=p, s_params=s_params, apxn=True)
        smln.save(rslt, group, commit)
        
        if wait:
            time.sleep(wait)
            
        stdout.write('.')
        stdout.flush()
        
        ctr += 1


def sample_params():
    # loop over all items in p_ranges, using
    # params directly if scalars, or sampling if 
    # [lb, ub] are given
    
    p = {}
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
    args = sys.argv[1:]
    
    if not len(args) in [2, 3]:
        raise Exception('2 or 3 arguments required.')
        
    group = args[0]
    commit = args[1]
    wait = int(args[2]) if len(args) == 3 else None
    
    print('Begin smln in group "{}" with commit "{}..." at {} s wait time?'.format(
        group, commit[:6], wait))
    confirm = input('[Y/N] ')
    
    if confirm.lower() == 'y':
        print('Commencing parameter search.\n')
        search(group, commit, wait)
