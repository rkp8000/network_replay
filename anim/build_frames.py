from copy import copy
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import os

from aux import load, save
from aux import load_time_file
from plot import set_font_size


def downsample_spks(spks, num):
    """
    Downsample a vector spk train to have num equally spaced samples.
    The downsampled spk value at a given time point is True if any spk occurred
    in the window surrounding that time point, and zero otherwise.
    
    :param spks: 2-D logical array indicating spk times (rows are times, cols
        are neurons)
    :param num: number of time points in the resampled signal
    """
    window = len(spks) / num
    
    if window < 1:
        err_msg = ('Provided "num" value must be less than len(spks); '
                   'upsampling is not supported')
        raise ValueError(err_msg)
    
    # just use a loop for now
    
    spks_down = np.zeros((num, spks.shape[1]), dtype=bool)
    
    for f_ctr in range(num):
        
        # get start and end of window for this downsampled time point
        start = int(round(window * f_ctr))
        end = int(round(window * (f_ctr + 1)))
        
        # assign the downsampled value to True for each neuron in which
        # any spk occurred in this time window
        spks_down[f_ctr] = np.any(spks[start:end], axis=0)
    
    return spks_down


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
        save_prefix, time_file, ntwk_file, fps=30, resting_size=50, spk_size=1000, amp=1,
        positions=None, cxn_color=(0, 0, 0), cxn_lw=1, cxn_zorder=-1,
        default_color=(0, 0, 0), spk_color=(1, 0, 0), frames_per_spk=5,
        box=None, title='', x_label='', y_label='', x_ticks=None, y_ticks=None,
        x_tick_labels=None, y_tick_labels=None, show_timestamp=True,
        fig_size=(6.4, 4.8), font_size=16, verbose=False):
    """
    Convert a time-series of membrane potentials and spks into viewable frames.
    
    :param save_prefix: prefix of frame files
    :param time_file: file with dict containing the following fields:
        'timestamps': 1-D array containing timestamps corresponding to neural activity
        'fs': scalar sampling frequency
    :param ntwk_file: file with dict containing the following fields:
        'vs': 2-D array containing membrane potential values (in V) where rows are time points
            and cols are neurons
        'spks': 2-D logical array indicating spk times of individual neurons
        'w': square matrix indicating recurrent connection weights among neurons
        'v_rest': resting membrane potential (in V)
        'v_th': threshold membrane potential (in V)
    :param fps: frame rate
    :param resting_size: size of neurons at rest
    :param spk_size: size of neurons when they've reached spking threshold
    :param amp: how much to polynomially amplify the visual difference between 
        different membrane voltages (keep at 1 for linear relationship between
        membrane voltage and circular area)
    :param cxn_color: color or dict of colors, e.g.,
        cxn_color=(0, 0, 0),
        cxn_color={('EX', 'EX'): (0, 0, 0), ('INH', 'EX'): (1, 0, 0)},
    :param default_color: neuron color
    :param spk_color: color of neuron when spking
    :param box: bounding box to display neurons in: (x_min, x_max, y_min, y_max)
    :param show_timestamp: whether or not to show timestamp in figure title
    :param fig_size: size of figure to make
    :param verbose: whether or not to print progress details
    """
    
    if verbose:
        print('Using timestamp file "{}" and activity file "{}".'.format(time_file, ntwk_file))
        print('Frames will be saved with prefix "{}".'.format(save_prefix))
        print('Loading timestamps and network activity data...')
        
    # load time stamps
    ts, fs = load_time_file(time_file)

    if fps > fs:
        err_msg = ('Provided "fps" value must be smaller than original sampling frequency; '
               'upsampling is not supported.')
        raise ValueError(err_msg)
    
    # load activity data
    data_a = load(ntwk_file)
    
    for key in ('vs', 'spks', 'v_rest', 'v_th'):
        if key not in data_a:
            raise KeyError('Item with key "{}" not found in file "{}".'.format(key, ntwk_file))
            
    vs = data_a['vs']
    spks = data_a['spks']
    v_rest = data_a['v_rest']
    v_th = data_a['v_th']
    cell_types = data_a['cell_types']
    
    # downsample data if necessary
    if fps < fs:
        if verbose:
            print('Downsampling data from {} Hz to {} fps...'.format(fs, fps))
        
        n_down = int(round((ts[-1] - ts[0]) * fps))
        ts = downsample_ma(ts, n_down)
        
        # membrane potential and spks
        vs = downsample_ma(vs, n_down)
        spks = downsample_spks(spks, n_down)
    
        if verbose:
            print('Data downsampled.')
    
    # let n be number of nrns
    n = vs.shape[1]
    
    # make sure timestamp vector is same length as activity vectors
    assert len(ts) == len(vs) == len(spks)
    
    # generate random positions if not provided
    if positions is None:
        positions = np.random.normal(0, 1, (2, n))
        
    if positions.shape != (2, n):
        raise ValueError('Arg "positions" must be a (2 x N) array.')
     
    # convert default_color into array of colors for each cell
    if cell_types is not None and isinstance(default_color, dict):
        # make sure cell types align with specified color cell types
        try:
            assert np.all([ct in default_color for ct in np.unique(cell_types)])
        except:
            raise KeyError('All cell types must have a default color.')
            
        default_color = np.array([default_color[ct] for ct in cell_types])
    else:
        default_color = np.array([default_color] * vs.shape[1])
    
    # get recurrent cxns
    ws_rcr = data_a['ws_rcr'] if 'ws_rcr' in data_a else None
    
    # check cxn visualization args
    if isinstance(cxn_color, dict) and not isinstance(cxn_lw, dict):
        cxn_lw = {k: cxn_lw for k in cxn_color}
    if isinstance(cxn_color, dict) and not isinstance(cxn_zorder, dict):
        cxn_zorder = {k: cxn_zorder for k in cxn_color}
    if isinstance(cxn_lw, dict) and not isinstance(cxn_color, dict):
        cxn_color = {k: cxn_color for k in cxn_lw}
    if isinstance(cxn_color, dict):
        if not np.all([isinstance(key, tuple) and len(key) == 2 for key in cxn_color]):
            raise TypeError('All keys in "cxn_color" must be tuples specifying (targ, src).')
    
    if verbose:
        print('Data loaded.')
    
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
    slope = (spk_size - resting_size) / ((v_th - v_rest)**amp)
    sizes = slope * ((vs - v_rest)**amp) + resting_size
    
    # make sure save directory exists
    save_dir = os.path.dirname(save_prefix)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # set up figure
    fig, ax = plt.subplots(1, 1, figsize=fig_size, tight_layout=True)
    
    ax.set_xlim(box[:2])
    ax.set_ylim(box[2:])
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels)
    if y_tick_labels is not None:
        ax.set_yticklabels(y_tick_labels)
        
    set_font_size(ax, font_size)
    
    # draw cxns if desired
    if np.any([np.any(w) for w in ws_rcr.values()]):
        
        # make all cxns same color if cxn_color is just a tuple
        if not isinstance(cxn_color, dict):
            
            for w in ws_rcr.values():
                
                line = w_to_line(w, positions)
                ax.plot(line[0], line[1], color=cxn_color, lw=cxn_lw, zorder=-1)
                
        else:
            
            # color cxns according to their src and targ cell types
            for targ, src in cxn_color:
                
                # get cxn mask
                targ_mask = np.array(cell_types) == targ
                src_mask = np.array(cell_types) == src
                targ_src_mask = np.outer(targ_mask, src_mask)
                
                # draw cxns
                lw = cxn_lw[(targ, src)]
                color = cxn_color[(targ, src)]
                
                for w in ws_rcr.values():
                    
                    # make new w with unmasked cxns set to zero
                    w_ = w.copy()
                    w_[~targ_src_mask] = 0
                    
                    # draw cxns
                    line = w_to_line(w_, positions)
                    ax.plot(line[0], line[1], color=color, lw=lw, zorder=cxn_zorder[(targ, src)])
                    
    # position neurons
    sca = ax.scatter(positions[0], positions[1], c=default_color, s=10, lw=0)
    
    # loop over frames
    if verbose:
        print('Generating and saving {} frames...'.format(len(ts)))
        
    save_files = []
    spk_offset_ctr = np.zeros(spks.shape[1], dtype=int)
    
    for f_ctr, (t, sizes_, spks_) in enumerate(zip(ts, sizes, spks)):
        
        # set colors according to spking
        spk_offset_ctr[spks_] = frames_per_spk
        
        # set colors
        if not any(spk_offset_ctr):
            sca.set_color(default_color)
        else:
            colors = [spk_color if s else dc for s, dc in zip(spk_offset_ctr, default_color)]
            sca.set_color(colors)
            
        # set sizes of non-spking neurons
        sizes_[spk_offset_ctr > 0] = spk_size
        sca.set_sizes(sizes_)
        
        if show_timestamp:
            if title:
                ax.set_title('{0}\nt = {1:.3f} s'.format(title, t), fontsize=font_size)
            else:
                ax.set_title('t = {0:.3f} s'.format(t), fontsize=font_size)
            
        plt.draw()
        
        spk_offset_ctr[spk_offset_ctr > 0] -= 1
        
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
    :param time_file: file with dict containing the following fields:
        'timestamps': 1-D array containing timestamps corresponding to neural activity
        'fs': scalar sampling frequency
    :param traj_file: file with dict containing the following fields:
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
    data_tr = load(traj_file)
    
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
        alphas = np.exp((ts[:f_ctr+1] - t)/decay)
        
        # plot trailing path
        colors = [path_color + (alpha,) for alpha in alphas]
        sca = ax.scatter(xs[:f_ctr+1], ys[:f_ctr+1], s=path_size, color=colors, lw=0, zorder=0)
            
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
        
        # remove trailing path so it doesn't interfere with next frame
        sca.remove()
        
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
    
    # get eigenvalues and vectors of covariance
    evs, evecs = np.linalg.eig(cov)
    
    # correct evs for small numerical errors
    evs = np.max([np.real(evs), [0, 0]], axis=0)
    
    # convert evs and evecs to ellipse parameters
    width, height = scale*np.sqrt(evs)
    angle = -np.arctan(evecs[0, 1]/evecs[0, 0]) * 180 / np.pi
        
    ell = Ellipse(xy, width=width, height=height, angle=angle, color=color)        
    
    return ell


def w_to_line(w, xys):
    """Convert a connection matrix and set of positions to a single plottable line.
    
    :param w: cxn matrix
    :param xys: neuron positions
    """
    if not (w.shape[0] == w.shape[1]):
        raise ValueError('Argument "w" must be a square array.')
    if not (xys.shape == (2, w.shape[1])):
        raise ValueError('Argument "xys" must be a (2 x N) array.')
    
    # get targ and src cell idxs for all nonzero weights
    idxs_targ, idxs_src = w.nonzero()
    
    # get positions corresponding to targs and src
    xs_targ, ys_targ = xys[:, idxs_targ]
    xs_src, ys_src = xys[:, idxs_src]
    
    # convert positions into pairs of coords separated by nans
    nans = np.nan * np.zeros(xs_targ.shape)
    
    xs_line = np.array([xs_src, xs_targ, nans]).T.flatten()
    ys_line = np.array([ys_src, ys_targ, nans]).T.flatten()
    
    return np.array([xs_line, ys_line])
