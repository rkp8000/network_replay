import matplotlib.pyplot as plt
import numpy as np
import os
import shelve


def network_activity(
        save_prefix, time_file, activity_file, fps=None, resting_size=50, spiking_size=1000,
        default_color=(0, 0, 0), spiking_color=(1, 0, 0), box=(-1, 1, -1, 1), fig_size=(7, 7),
        verbose=False):
    """
    Convert a time-series of membrane potentials and spikes into viewable frames.
    
    :param save_prefix: prefix of frame files
    :param time_file: shelved file containing the following fields:
        'timestamps': 1-D array containing timestamps corresponding to neural activity
        'fs': scalar sampling frequency
    :param activity_file: shelved file containing the following fields:
        'vs': 2-D array containing membrane potential values (in V) where rows are time points
            and cols are neurons
        'spikes': 2-D logical array indicating spike times of individual neurons
        'positions': 2-D array containing (x, y) positions of neurons (cols are neurons)
        'w': square matrix indicating recurrent connection weights among neurons
        'v_rest': resting membrane potential (in V)
        'v_th': threshold membrane potential (in V)
    :param fps: frame rate
    :param resting_size: size of neurons at rest
    :param spiking_size: size of neurons when they've reached spiking threshold
    :param default_color: neuron color
    :param spiking_color: color of neuron when spiking
    :param box: bounding box to display neurons in: (x_min, x_max, y_min, y_max)
    :param fig_size: size of figure to make
    """
    
    # load time stamps
    data_t = shelve.open(time_file)
    
    if 'timestamps' not in data_t:
        raise KeyError('Item with key "timestamps" not found in file "{}".'.format(time_file))
        
    ts = data_t['timestamps']
    
    # load activity data
    data_a = shelve.open(activity_file)
    
    for key in ('vs', 'spikes', 'v_rest', 'v_th'):
        if key not in data_a:
            raise KeyError('Item with key "{}" not found in file "{}".'.format(key, activity_file))
            
    vs = data_a['vs']
    spikes = data_a['spikes']
    v_rest = data_a['v_rest']
    v_th = data_a['v_th']
    
    # extract positions from data or generate random ones
    if 'positions' in data_a:
        positions = data_a['positions']
    else:
        positions = np.random.normal(0, 1, (2, vs.shape[1]))
    
    # make sure timestamp vector is same length as activity vectors
    assert len(ts) == len(vs) == len(spikes)
    
    # convert membrane potentials to scatter sizes
    slope = (spiking_size - resting_size) / (v_th - v_rest)
    sizes = slope * (vs - v_rest) + resting_size
    
    # make sure save directory exists
    save_dir = os.path.dirname(save_prefix)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # set up figure and position neurons
    fig, ax = plt.subplots(1, 1)
    
    ax.set_xlim(*box[:2])
    ax.set_ylim(*box[2:])
    
    sca = ax.scatter(positions[0], positions[1], c=default_color, s=10, lw=0)
    
    # loop over frames
    save_files = []
    
    for f_ctr, (t, sizes_, spikes_) in enumerate(zip(ts, sizes, spikes)):
        
        # set sizes
        sca.set_sizes(sizes_)
        
        # set colors
        if not any(spikes_):
            sca.set_color(default_color)
        else:
            colors = [spiking_color if s else default_color for s in spikes_]
            sca.set_color(colors)
            
        plt.draw()
        
        save_file = '{}_{}.png'.format(save_prefix, f_ctr+1)
        save_files.append(save_file)
        
        fig.savefig(save_file)
        
    plt.close()
    return save_files
