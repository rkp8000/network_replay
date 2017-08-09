from copy import copy
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import os

from aux import load_time_file
from plot import set_font_size
import shelve


def downsample_spikes(spikes, num):
    """
    Downsample a vector spike train to have num equally spaced samples.
    The downsampled spike value at a given time point is True if any spike occurred
    in the window surrounding that time point, and zero otherwise.
    
    :param spikes: 2-D logical array indicating spike times (rows are times, cols
        are neurons)
    :param num: number of time points in the resampled signal
    """
    window = len(spikes) / num
    
    if window < 1:
        err_msg = ('Provided "num" value must be less than len(spikes); '
                   'upsampling is not supported')
        raise ValueError(err_msg)
    
    # just use a loop for now
    
    spikes_down = np.zeros((num, spikes.shape[1]), dtype=bool)
    
    for f_ctr in range(num):
        
        # get start and end of window for this downsampled time point
        start = int(round(window * f_ctr))
        end = int(round(window * (f_ctr + 1)))
        
        # assign the downsampled value to True for each neuron in which
        # any spike occurred in this time window
        spikes_down[f_ctr] = np.any(spikes[start:end], axis=0)
    
    return spikes_down


def downsample_ma(xs, num):
    """
    Downsample an array to have num equally spaced samples, where the downsampled
    value at each time point is a moving average of the values in the corresponding window.
    
    :param xs: N-D array of values (1st dim is times, higher dims are variables)
    :param num: number of time points in the resampled signal
    """
    window = len(xs) / num
    
    if window < 1:
        err_msg = ('Provided "num" value must be less than len(xs); '
                   'upsampling is not supported')
        raise ValueError(err_msg)
    
    # just use a loop for now
    xs_down = np.nan * np.zeros((num,) + xs.shape[1:])
    
    for f_ctr in range(num):
        
        # get start and end of window for this downsampled time point
        start = int(round(window * f_ctr))
        end = int(round(window * (f_ctr + 1)))
        
        # assign the downsampled value to the average of the values in
        # the corresponding window
        xs_down[f_ctr] = np.mean(xs[start:end], axis=0)
    
    return xs_down


def ntwk_activity(
        save_prefix, time_file, activity_file, fps=30, resting_size=50, spiking_size=1000,
        default_color=(0, 0, 0), spiking_color=(1, 0, 0), frames_per_spike=5,
        box=None, title='', x_label='', y_label='', show_timestamp=True,
        fig_size=(6.4, 4.8), font_size=16, verbose=False):
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
    :param show_timestamp: whether or not to show timestamp in figure title
    :param fig_size: size of figure to make
    :param verbose: whether or not to print progress details
    """
    
    if verbose:
        print('Using timestamp file "{}" and activity file "{}".'.format(time_file, activity_file))
        print('Frames will be saved with prefix "{}".'.format(save_prefix))
        print('Loading timestamps and network activity data...')
        
    # load time stamps
    ts, fs = load_time_file(time_file)

    if fps > fs:
        err_msg = ('Provided "fps" value must be smaller than original sampling frequency; '
               'upsampling is not supported.')
        raise ValueError(err_msg)
    
    # load activity data
    data_a = shelve.open(activity_file)
    
    for key in ('vs', 'spikes', 'v_rest', 'v_th'):
        if key not in data_a:
            raise KeyError('Item with key "{}" not found in file "{}".'.format(key, activity_file))
            
    vs = data_a['vs']
    spikes = data_a['spikes']
    v_rest = data_a['v_rest']
    v_th = data_a['v_th']
    
    # make sure timestamp vector is same length as activity vectors
    assert len(ts) == len(vs) == len(spikes)
    
    if verbose:
        print('Data loaded.')
        
    # downsample data if necessary
    if fps < fs:
        if verbose:
            print('Downsampling data from {} Hz to {} fps...'.format(fs, fps))
        
        n_down = int(round((ts[-1] - ts[0]) * fps))
        ts = downsample_ma(ts, n_down)
        
        # membrane potential and spikes
        vs = downsample_ma(vs, n_down)
        spikes = downsample_spikes(spikes, n_down)
    
        if verbose:
            print('Data downsampled.')
    
    # extract positions from data or generate random ones
    if 'place_field_centers' in data_a:
        positions = data_a['place_field_centers']
    else:
        positions = np.random.normal(0, 1, (2, vs.shape[1]))
        
    # automatically compute box size if not provided
    if box is None:
        x_min = positions[0].min()
        x_max = positions[0].max()
        x_r = x_max - x_min
        
        y_min = positions[1].min()
        y_max = positions[1].max()
        y_r = y_max - y_min
        
        box = [
            x_min - 0.1*x_r,
            x_max + 0.1*x_r,
            y_min - 0.1*y_r,
            y_max + 0.1*y_r,
        ]
    
    # automatically adjust box if either dimension is zero
    box = correct_box_dims(box)

    # convert membrane potentials to scatter sizes
    slope = (spiking_size - resting_size) / (v_th - v_rest)
    sizes = slope * (vs - v_rest) + resting_size
    
    # make sure save directory exists
    save_dir = os.path.dirname(save_prefix)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # set up figure and position neurons
    fig, ax = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
    
    ax.set_xlim(box[:2])
    ax.set_ylim(box[2:])
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    set_font_size(ax, font_size)
    
    sca = ax.scatter(positions[0], positions[1], c=default_color, s=10, lw=0)
    
    # loop over frames
    if verbose:
        print('Generating and saving {} frames...'.format(len(ts)))
        
    save_files = []
    spike_offset_ctr = np.zeros(spikes.shape[1], dtype=int)
    
    for f_ctr, (t, sizes_, spikes_) in enumerate(zip(ts, sizes, spikes)):
        
        # set colors according to spiking
        spike_offset_ctr[spikes_] = frames_per_spike
        
        # set colors
        if not any(spike_offset_ctr):
            sca.set_color(default_color)
        else:
            colors = [spiking_color if s else default_color for s in spike_offset_ctr]
            sca.set_color(colors)
            
        # set sizes of non-spiking neurons
        sizes_[spike_offset_ctr > 0] = spiking_size
        sca.set_sizes(sizes_)
        
        if show_timestamp:
            if title:
                ax.set_title('{0}\nt = {1:.3f} s'.format(title, t), fontsize=font_size)
            else:
                ax.set_title('t = {0:.3f} s'.format(t), fontsize=font_size)
            
        plt.draw()
        
        spike_offset_ctr[spike_offset_ctr > 0] -= 1
        
        save_file = '{}_{}.png'.format(save_prefix, f_ctr+1)
        save_files.append(save_file)
        
        fig.savefig(save_file)
        
    plt.close()
    
    if verbose:
        print('Frames saved.')
        
    return save_files


def traj(
        save_prefix, time_file, traj_file, fps=30, decay=0.5,
        location_size=2000, path_size=200, location_color=(0, 0, 1, .3), path_color=(0, 0, 0),
        cov_cutoff=None, cov_color=(0, 1, 0, .3), cov_scale=3,
        box=None, title='', x_label='', y_label='', fig_size=(6.4, 4.8),
        show_timestamp=True, font_size=16, verbose=False):
    """
    Convert a time-series of positions into a series of still frames.
    
    :param save_prefix: prefix of frame files
    :param time_file: shelved file containing the following fields:
        'timestamps': 1-D array containing timestamps corresponding to neural activity
        'fs': scalar sampling frequency
    :param traj_file: shelved file containing the following fields:
        'xys': 2-D array containing (x, y) coordinates of each position over time; rows are
            time points, cols are x and y
        ['covs']: optional 3-D array containing uncertainty (covariance) matrix at each time
            point 
    :param fps: frame rate
    :param location_size: size of current location marker
    :param path_size: size of markers indicating recent path
    :param location_color: color of current location marker
    :param path_color: color of markers indicating recent path (note: rgb, not rgba)
    :param cov_cutoff: maximum covariance value allowed (in squared units) when showing
        uncertainty ellipse of position estimate; if None, uncertainty ellipse is not shown
    :param cov_color: rgba color of covariance ellipse
    :param cov_scale: size of ellipse relative to square root of covariance values
    :param box: bounding box to display path in
    :param show_timestamp: whether or not to show timestamp in figure title
    :param fig_size: size of figure to make
    :param verbose: whether or not to print progress details
    """
    
    # load time stamps
    ts, fs = load_time_file(time_file)

    # load trajectory data
    data_tr = shelve.open(traj_file)
    
    # make sure traj file contains required keys
    if 'xys' not in data_tr:
        raise KeyError('Item with key "xys" not found in file "{}".'.format(traj_file))
        
    if (cov_cutoff is not None) and ('covs' not in data_tr):
        
        raise KeyError(
            'When "cov_cutoff" is not None, key "covs" must be included '
            'in file "{}".'.format(traj_file))
        
    xys = data_tr['xys']
    covs = data_tr['covs'] if cov_cutoff is not None else None
        
    # downsample data if necessary
    if fps < fs:
        
        n_down = int(round((ts[-1] - ts[0]) * fps))
        ts = downsample_ma(ts, n_down)

        xys = downsample_ma(xys, n_down)
        
        if covs is not None:
            covs = downsample_ma(covs, n_down)
    
    # convert xys to two 1D arrays
    xs, ys = xys.T

    # automatically compute box size if not provided
    if box is None:
        x_min = xs.min()
        x_max = xs.max()
        x_r = x_max - x_min
        
        y_min = ys.min()
        y_max = ys.max()
        y_r = y_max - y_min
        
        box = [
            x_min - 0.1*x_r,
            x_max + 0.1*x_r,
            y_min - 0.1*y_r,
            y_max + 0.1*y_r,
        ]
    
    # automatically adjust box if either dimension is zero
    box = correct_box_dims(box)

    # make sure save directory exists
    save_dir = os.path.dirname(save_prefix)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
        
    # set up figure
    fig, ax = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
    
    ax.set_xlim(box[:2])
    ax.set_ylim(box[2:])
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    set_font_size(ax, font_size)
    
    # plot all points with zero opacity
    sca = ax.scatter(xs, ys, s=path_size, c=(path_color + (0,)), lw=0, zorder=0)
    
    # plot current position
    sca_2 = ax.scatter(xs[0], ys[0], s=location_size, c=location_color, lw=0, zorder=1)
    
    # plot uncertainty (covariance) ellipse
    if covs is not None:
        unc = ellipse_from_cov(xys[0], covs[0], cov_scale, cov_color)
        
        if unc is not None:
            ax.add_artist(unc)
            
            if np.trace(covs[0]) >= cov_cutoff:
                unc.set_alpha(0)
    
    # loop over frames
    save_files = []
    
    for f_ctr, (t, (x, y)) in enumerate(zip(ts, xys)):
        
        # get opacities of trailing path
        alphas = np.exp((ts - t)/decay)
        alphas[ts > t] = 0
        
        # plot trailing path
        colors = [path_color + (alpha,) for alpha in alphas]
            
        sca.set_color(colors)
        
        # plot current location
        sca_2.set_offsets([x, y])
        
        # plot covariance ellipse
        if covs is not None:
            if unc is not None:
                unc.remove()
                
            unc = ellipse_from_cov([x, y], covs[f_ctr], cov_scale, cov_color)
        
            if unc is not None:
                ax.add_artist(unc)

                if np.trace(covs[f_ctr]) >= cov_cutoff:
                    unc.set_alpha(0)
        
        if show_timestamp:
            if title:
                ax.set_title('{0}\nt = {1:.3f} s'.format(title, t), fontsize=font_size)
            else:
                ax.set_title('t = {0:.3f} s'.format(t), fontsize=font_size)
            
        plt.draw()
        
        save_file = '{}_{}.png'.format(save_prefix, f_ctr+1)
        save_files.append(save_file)
        
        fig.savefig(save_file)
        
    plt.close()
        
    return save_files


def correct_box_dims(box):
    """
    Check a box (left, right, bottom, top) for zero-valued dimensions and
    automatically replace them if found.
    :param box: box dimensions (left, right, bottom, top)
    :return: corrected box
    """
    box = copy(box)

    if all([b == 0 for b in box]):
        print('Zero-dimensioned box detected. Adjusting width and height to 1')
        box = [-0.5, 0.5, -0.5, 0.5]

    if box[0] == box[1]:
        print('Zero-width box detected. Adjusting to 1/5 of height.')
        temp = (box[3] - box[2]) / 5
        box[0] -= temp/2
        box[1] += temp/2

    if box[2] == box[3]:
        print('Zero-height box detected. Adjusting to 1/5 of width.')
        temp = (box[1] - box[0]) / 5
        box[2] -= temp/2
        box[3] += temp/2

    return box
                        

def ellipse_from_cov(xy, cov, scale, color):
    """
    Return an ellipse object from a covariance matrix. Add it to an axis
    using ax.add_artist(ell), and remove it using ell.remove().
    
    :param cov: 2x2 covariance matrix
    :param scale: how much to scale the covariance stds to get width and height
    :param color: rgba color of ellipse
    :return: ellipse object
    """
    if np.any(np.isnan(xy)) or np.any(np.isnan(cov)) or (np.trace(cov) == 0):
        return None
    
    evs, evecs = np.linalg.eig(cov)
        
    width, height = scale*np.sqrt(evs)
    angle = -np.arctan(evecs[0, 1]/evecs[0, 0]) * 180 / np.pi
        
    ell = Ellipse(xy, width=width, height=height, color=color)        
    
    return ell
