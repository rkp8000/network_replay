from copy import copy
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy.sparse import issparse
import time

from aux import load, save
from aux import load_time_file
from aux import downsample_spks, downsample_ma
from plot import set_font_size


def ntwk(
        frame_prfx, rsp, t_start=None, t_end=None,
        positions=None, box=None, fig_w=None, fig_h=None,
        resting_size=50, spk_size=1000, amp=1,
        non_spk_colors=None, spk_color='r',
        cxn_colors=None, cxn_lws=None, cxn_z_orders=None,
        x_ticks=None, y_ticks=None, x_tick_labels=None, y_tick_labels=None,
        x_label='', y_label='', title='', font_size=16,
        fps=30, frames_per_spk=5, show_timestamp=True,
        verbose=True, report_every=60):
    """
    Convert a time-series of membrane potentials and spks into viewable frames.
    
    :param frame_prfx: prefix of path to save frames at
    :param rsp: ntwk response object or path to ntwk response object
        rsp should have the following attributes:
            v_rest, v_th, ts, vs, spks, [cell_types, ws_rcr]
    :param t_start: start time of animation
    :param t_end: end time of animation
    :param positions: 2 x N array of cell positions
        if None, positions will be uniform in unit square
    :param box: bounding box for displaying cells
    :param fig_w: figure width
    :param fig_h: figure height
        at least fig_w or fig_h must be provided; the other can be
        calculated automatically from box
    :param resting_size: size of cell at resting potential
    :param spk_size: size of cell when spiking
    :param non_spk_color: dict of colors of non-spiking cells
    :param spk_color: color of spiking cells
    :param cxn_colors: dict of cxn colors, e.g.
        {('PC', 'PC'): 'k', ('INH', 'PC': 'b')}
    :param cxn_lws: dict of cxn line widths
    :param cxn_z_orders: dict of cxn z_orders
    :params x_ticks, y_ticks, x_tick_labels, y_tick_labels: as in matplotlib
    :params x_label, y_label, title: as in matplotlib
    :param font_size: font size
    :param fps: frame rate of animation (frames per second)
    :param frames_per_spk: how many frames to show each spk for
    :param show_timestamp: whether to append current timestamp below title
    :param verbose: whether to print out frame-building progress
    :param report_every: how often to report progress if verbose is True (s)
    
    :return: list of paths to frames, extra info about frames
    """
    alert = lambda m: print(m) if verbose else None
    alert('\n')
    
    # load response file
    if isinstance(rsp, str):
        alert('Loading activity file "{}"...'.format(rsp))
        rsp = load(rsp)
        alert('Loaded.\n')
    
    v_rest = rsp.v_rest
    v_th = rsp.v_th
    ts = rsp.ts
    vs = rsp.vs
    spks = rsp.spks
    
    n = vs.shape[1]
    dt = np.mean(np.diff(rsp.ts))
    fs = 1 / dt
    
    if hasattr(rsp, 'cell_types'):
        cell_types = rsp.cell_types
    else:
        cell_types = np.repeat('PC', n)
    
    ws_rcr = rsp.ws_rcr if hasattr(rsp, 'ws_rcr') else None
    
    if fps > fs:
        raise ValueError(
            'fps must be smaller than original sampling '
            'frequency (upsampling is not supported).')
        
    # make sure timestamp vector is same length as activity vectors
    assert len(ts) == len(vs) == len(spks)
    
    if t_start is None:
        t_start = ts[0] - dt
    if t_end is None:
        t_end = ts[-1] + dt
        
    # generate random positions if not provided
    if positions is None:
        positions = np.random.uniform(0, 1, (2, n))
        
    if positions.shape != (2, n):
        raise ValueError('Arg "positions" must be a (2 x N) array.')
     
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
    
    # convert non_spk_colors into array of colors for each cell
    if not non_spk_colors:
        non_spk_colors = {cell_type: 'k' for cell_type in set(cell_types)}
     
    assert set(rsp.cell_types) == set(non_spk_colors)

    non_spk_colors_ = np.repeat('', n)
    
    for cell_type in set(rsp.cell_types):
        non_spk_colors_[rsp.cell_types == cell_type] = non_spk_colors[cell_type]
    
    non_spk_colors = non_spk_colors_
   
    # make sure all cxn args are dicts
    if ws_rcr is not None:
        assert isinstance(cxn_colors, dict)
        assert isinstance(cxn_lws, dict)
        assert isinstance(cxn_z_orders, dict)
    
        if not set(cxn_colors) == set(cxn_lws) == set(cxn_z_orders):
            raise KeyError('All cxn* args must have same keys.')
    else:
        cxn_colors = {}
        
    assert (fig_w is not None) or (fig_h is not None)
   
    # select data only in desired time window
    t_mask = (t_start <= ts) & (ts < t_end)
    
    assert t_mask.sum() > 0
    
    ts = ts[t_mask]
    vs = vs[t_mask]
    spks = spks[t_mask]
    
    # downsample data if necessary
    if fps < fs:
        alert('Downsampling data from {} to {} fps...'.format(fs, fps))
        
        n_down = int(round((ts[-1] - ts[0]) * fps))
        ts = downsample_ma(ts, n_down)
        
        # membrane potential and spks
        vs = downsample_ma(vs, n_down)
        spks = downsample_spks(spks, n_down)
    
        alert('Downsampled.\n')
    
    # convert membrane potentials to scatter sizes
    slope = (spk_size - resting_size) / ((v_th - v_rest)**amp)
    sizes = slope * ((vs - v_rest)**amp) + resting_size
    
    # make sure save directory exists
    frame_dir = os.path.dirname(frame_prfx)
    if not os.path.exists(frame_dir): os.makedirs(frame_dir)
    
    # set up figure
    aspect = (box[3] - box[2]) / (box[1] - box[0])
    
    if fig_h is None:
        fig_h = fig_w * aspect
    elif fig_w is None:
        fig_w = fig_h / aspect
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), tight_layout=True)
    
    ax.set_xlim(box[:2])
    ax.set_ylim(box[2:])
    
    ax.autoscale(False)
    
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
    if cxn_colors:
        alert('Drawing cxns...')
        
    for key in cxn_colors:
        
        targ, src, syn = key
        
        # get cxn mask
        mask = np.outer(
            np.array(cell_types) == targ,
            np.array(cell_types) == src)

        # convert cxns to broken line for fast plotting
        line = w_to_line(ws_rcr[syn], mask, positions)
        
        ax.plot(
            *line,
            color=cxn_colors[key],
            lw=cxn_lws[key],
            zorder=cxn_z_orders[key])
    
    if cxn_colors:
        alert('Drawn.\n')

    # position neurons
    sca = ax.scatter(positions[0], positions[1], c=non_spk_colors, s=0, lw=0)
    
    # loop over frames
    alert(
        'Generating and saving {0} frames spanning times {1:.6f} to {2:.6f} s'
        '...'.format(len(ts), ts[0], ts[-1]))
        
    frames = []
    spk_offset_ctr = np.zeros(spks.shape[1], dtype=int)
    
    loop_start_time = time.time()
    last_update = time.time()
    
    for f_ctr, (t, sizes_, spks_) in enumerate(zip(ts, sizes, spks)):
        
        # get new spks
        spk_offset_ctr[spks_] = frames_per_spk
        
        # set colors according to spiking
        colors = non_spk_colors.copy()
        colors[spk_offset_ctr > 0] = spk_color
        
        sca.set_color(colors)
            
        # set sizes of non-spking neurons
        sizes_[spk_offset_ctr > 0] = spk_size
        sca.set_sizes(sizes_)
        
        if show_timestamp:
            title_ = '{0}\nt = {1:.3f} s'.format(title, t)
            ax.set_title(title_, fontsize=font_size)
            
        plt.draw()
        
        spk_offset_ctr[spk_offset_ctr > 0] -= 1
        
        frame = '{}_{}.png'.format(frame_prfx, f_ctr+1)
        frames.append(frame)
        
        fig.savefig(frame)
        
        if time.time() > last_update + report_every:
            
            alert('{0} frames completed after {1:.3f} s...'.format(
                f_ctr + 1, time.time() - loop_start_time)) 
            
            last_update = time.time()
        
    plt.close()
    
    alert('All frames written to disk.')
    
    extra = {
        'fig_size': (fig_w, fig_h),
    }
        
    return frames, extra


def traj(
        frame_prfx, traj, t_start, t_end, box, fig_w, fig_h,
        loc_size=1000, path_size=100,
        loc_color=(0, 0, 1, .3), path_color=(0, 0, 0), decay=0.5,
        x_ticks=None, y_ticks=None, x_tick_labels=None, y_tick_labels=None,
        x_label='', y_label='', title='', font_size=16,
        fps=30, show_timestamp=True, verbose=False, report_every=60):
    """
    Convert a time-series of positions into a series of still frames.
    
    :param frame_prfx: prfx of frame files
    :param traj: traj object with attributes ts, xys
    :param t_start: start time of animation
    :param t_end: end time of animation
    :param box: bounding box for displaying cells
    :param fig_w: figure width
    :param fig_h: figure height
        at least fig_w or fig_h must be provided; the other can be
        calculated automatically from box
    :param loc_size: size of current location indicator
    :param path_size: size of trailing path
    :param loc_color: color of current lcoation indicator
    :param path_color: color of trailing path
    :param decay: decay time constant of path (s)
    :params x_ticks, y_ticks, x_tick_labels, y_tick_labels: as in matplotlib
    :params x_label, y_label, title: as in matplotlib
    :param font_size: font size
    :param fps: frame rate of animation (frames per second)
    :param show_timestamp: whether to append current timestamp below title
    :param verbose: whether to print out frame-building progress
    :param report_every: how often to report progress if verbose is True (s)
    
    :return: list of paths to frames, extra info about frames
    """
    alert = lambda m: print(m) if verbose else None
    alert('\n')
    
    # get timestamps
    ts = traj.ts
    dt = np.mean(np.diff(ts))
    fs = 1/dt
    
    xys = traj.xys
    
    if t_start is None:
        t_start = ts[0] - dt
    if t_end is None:
        t_end = ts[-1] + dt

    # select data only in desired time window
    t_mask = (t_start <= ts) & (ts < t_end)
    
    assert t_mask.sum() > 0
    
    ts = ts[t_mask]
    xys = xys[t_mask]
    
    # downsample data if necessary
    if fps < fs:
        alert('Downsampling data from {} to {} fps...'.format(fs, fps))
        
        n_down = int(round((ts[-1] - ts[0]) * fps))
        
        ts = downsample_ma(ts, n_down)
        xys = downsample_ma(xys, n_down)
        
        alert('Downsampled.\n')
        
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
    save_dir = os.path.dirname(frame_prfx)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
        
    # set up figure
    aspect = (box[3] - box[2]) / (box[1] - box[0])
    
    if fig_h is None:
        fig_h = fig_w * aspect
    elif fig_w is None:
        fig_w = fig_h / aspect
        
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), tight_layout=True)
    
    ax.set_xlim(box[:2])
    ax.set_ylim(box[2:])
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    set_font_size(ax, font_size)
    
    # plot current position
    sca_2 = ax.scatter(
        xs[0], ys[0], s=loc_size, c=loc_color, lw=0, zorder=1)
    
    # loop over frames
    alert(
        'Generating and saving {0} frames spanning times {1:.6f} to {2:.6f} s'
        '...'.format(len(ts), ts[0], ts[-1]))
    
    frames = []
    
    loop_start_time = time.time()
    last_update = time.time()
    
    for f_ctr, (t, (x, y)) in enumerate(zip(ts, xys)):
        
        # get opacities of trailing path
        alphas = np.exp((ts[:f_ctr+1] - t)/decay)
        
        # plot trailing path
        colors = [path_color + (alpha,) for alpha in alphas]
        sca = ax.scatter(
            xs[:f_ctr+1], ys[:f_ctr+1], s=path_size, color=colors, lw=0, zorder=0)
            
        # plot current location
        sca_2.set_offsets([x, y])
        
        if show_timestamp:
            title_ = '{0}\nt = {1:.3f} s'.format(title, t)
            ax.set_title(title_, fontsize=font_size)
            
        plt.draw()
        
        frame = '{}_{}.png'.format(frame_prfx, f_ctr+1)
        frames.append(frame)
        
        fig.savefig(frame)
        
        if time.time() > last_update + report_every:
            
            alert('{0} frames completed after {1:.3f} s...'.format(
                f_ctr + 1, time.time() - loop_start_time)) 
            
            last_update = time.time()
        
        # remove trailing path so it doesn't interfere with next frame
        sca.remove()
    
    alert('All frames written to disk.')
    
    plt.close()
    extra = {}
    
    return frames, extra


def meta(
        frame_prfx, meta, t_start, t_end, box, fig_w, fig_h,
        text_xys, colors, title='', font_size=16, title_font_size=16,
        fps=30, show_timestamp=True, verbose=False, report_every=60):
    """
    Convert a time-series of positions into a series of still frames.
    
    :param frame_prfx: prfx of frame files
    :param traj: traj object with attributes ts, xys
    :param t_start: start time of animation
    :param t_end: end time of animation
    :param box: bounding box for displaying cells
    :param fig_w: figure width
    :param fig_h: figure height
        at least fig_w or fig_h must be provided; the other can be
        calculated automatically from box
    :param text_xys: dict of text x, y positions (keys should be same
        as meta.labels.keys(), meta.indicators.keys())
    :param colors: dict of colors with same keys as text_xys
    :param font_size: font size
    :param fps: frame rate of animation (frames per second)
    :param show_timestamp: whether to append current timestamp below title
    :param verbose: whether to print out frame-building progress
    :param report_every: how often to report progress if verbose is True (s)
    
    :return: list of paths to frames, extra info about frames
    """
    alert = lambda m: print(m) if verbose else None
    alert('\n')
    
    # get timestamps
    ts = meta.ts
    dt = np.mean(np.diff(ts))
    fs = 1/dt
    
    texts = meta.texts
    indicators = meta.indicators
    
    if t_start is None:
        t_start = ts[0] - dt
    if t_end is None:
        t_end = ts[-1] + dt

    # select data only in desired time window
    t_mask = (t_start <= ts) & (ts < t_end)
    
    assert t_mask.sum() > 0
    
    ts = ts[t_mask]
    indicators = {
        key: indicator[t_mask]
        for key, indicator in indicators.items()
    }
    
    # downsample data if necessary
    if fps < fs:
        alert('Downsampling data from {} to {} fps...'.format(fs, fps))
        
        n_down = int(round((ts[-1] - ts[0]) * fps))
        
        ts = downsample_ma(ts, n_down)
        
        indicators = {
            key: downsample_spks(indicator[:, None], n_down)[:, 0]
            for key, indicator in indicators.items()
        }
        
        alert('Downsampled.\n')
        
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
    save_dir = os.path.dirname(frame_prfx)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
        
    # set up figure
    aspect = (box[3] - box[2]) / (box[1] - box[0])
    
    if fig_h is None:
        fig_h = fig_w * aspect
    elif fig_w is None:
        fig_w = fig_h / aspect
        
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), tight_layout=True)
    
    ax.set_xlim(box[:2])
    ax.set_ylim(box[2:])
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    set_font_size(ax, title_font_size)
    
    # loop over frames
    alert(
        'Generating and saving {0} frames spanning times {1:.6f} to {2:.6f} s'
        '...'.format(len(ts), ts[0], ts[-1]))
    
    frames = []
    
    loop_start_time = time.time()
    last_update = time.time()
    
    # set initial text objects
    text_objs = {}
    for key, text in texts.items():
        
        x, y = text_xys[key]
        color = colors[key]
        
        text_ = ax.text(x, y, text, fontsize=font_size, color=color)
        text_.set_alpha(0)
        
        text_objs[key] = text_
    
    for f_ctr, t in enumerate(ts):
        
        # update texts
        for key, text_ in text_objs.items():
            
            if indicators[key][f_ctr]:
                text_.set_alpha(1)
            else:
                text_.set_alpha(0)
            
        if show_timestamp:
            title_ = '{0}\nt = {1:.3f} s'.format(title, t)
            ax.set_title(title_, fontsize=title_font_size)
            
        plt.draw()
        
        frame = '{}_{}.png'.format(frame_prfx, f_ctr+1)
        frames.append(frame)
        
        fig.savefig(frame)
        
        if time.time() > last_update + report_every:
            
            alert('{0} frames completed after {1:.3f} s...'.format(
                f_ctr + 1, time.time() - loop_start_time)) 
            
            last_update = time.time()
        
    alert('All frames written to disk.')
    
    plt.close()
    extra = {}
    
    return frames, extra


def merge(
        frame_sets, save_prefix, rects, size,
        size_rel_to=1, delete_originals=False, verbose=False):
    """
    Create merged frames from multiple sets of frames.
    
    :param frame_sets: list of sets of frames to merge
    :param save_prefix: filename prefix for each saved merged frame
    :param rects: list of rectangular locs [(left, upper, right, lower), ...] for
        sub-images, with (0, 0) in upper left (relative to the image size)
    :param size: size of final image (relative to size of first frame set)
    :param size_rel_to: change for "size" argument to be relative to a
        different frame set
    :param delete_originals: whether to delete files used to make original gif
    :param verbose: whether to print out progress
    """
    
    # check arguments
    if not all([len(fs) for fs in frame_sets]):
        raise ValueError('All frame sets must have same number of frames.')
    
    if len(frame_sets) != len(rects):
        raise ValueError(
            'Each frame set must have exactly one corresponding rect.')
    
    # convert rects and size to pixels
    imgs_0 = [Image.open(fs[0]) for fs in frame_sets]
    sizes_orig = [img.size for img in imgs_0]
    
    rects_px = []
    for rect, sub_size in zip(rects, sizes_orig):
        
        left = int(np.round(rect[0]*sub_size[0]))
        upper = int(np.round(rect[1]*sub_size[1]))
        right = int(np.round(rect[2]*sub_size[0]))
        lower = int(np.round(rect[3]*sub_size[1]))
        
        rects_px.append((left, upper, right, lower))
        
    size_px = (
        size[0]*sizes_orig[size_rel_to][0],
        size[1]*sizes_orig[size_rel_to][1]
    )
        
    for rect_px in rects_px:
        if not rect_px[0] < rect_px[2]:
            raise ValueError('Rect left must be less than rect right.')
        if not rect_px[1] < rect_px[3]:
            raise ValueError('Rect upper must be less than rect lower.')
    
    if (not size_px[0] >= np.max([rect_px[2] for rect_px in rects_px])) \
            or (not size_px[1] >= np.max([rect_px[3] for rect_px in rects_px])):
        raise ValueError('Merged frame size must be larger than sub-image size.')
        
    sub_sizes = [
        (rect_px[2]-rect_px[0], rect_px[3]-rect_px[1])
        for rect_px in rects_px
    ]
    
    # make sure save directory exists
    save_dir = os.path.dirname(save_prefix)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # loop over all frames in each frame set
    save_files = []
    
    for f_ctr, frames in enumerate(np.array(frame_sets).T):
        
        # make new image
        img = Image.new('RGBA', size_px)
        
        # loop over sub images
        for frame, rect_px, sub_size in zip(frames, rects_px, sub_sizes):
            
            # open and resize sub image
            sub_img = Image.open(frame)
            if sub_size != sub_img.size:
                sub_img = sub_img.resize(sub_size)
            
            # paste sub image into merged image
            img.paste(sub_img, rect_px)
        
        # save merged image
        save_file = '{}_{}.png'.format(save_prefix, f_ctr+1)
        save_files.append(save_file)
        
        img.save(save_file)
    
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
                        

def w_to_line(w, mask, xys):
    """
    Convert a cxn matrix, mask, and set of positions to single plottable line.
    
    :param w: cxn matrix
    :param mask: cell type mask (same size as cxn matrix)
    :param xys: neuron positions
    """
    if not (w.shape[0] == w.shape[1]):
        raise ValueError('Argument "w" must be a square array.')
    if not (xys.shape == (2, w.shape[1])):
        raise ValueError('Argument "xys" must be a (2 x N) array.')
        
    # get targ and src cell idxs for all nonzero weights within mask
    idxs_targ, idxs_src = (w > 0).multiply(mask).nonzero()
    
    # get positions corresponding to targs and src
    xs_targ, ys_targ = xys[:, idxs_targ]
    xs_src, ys_src = xys[:, idxs_src]
    
    # convert positions into pairs of coords separated by nans
    nans = np.nan * np.zeros(xs_targ.shape)
    
    xs_line = np.array([xs_src, xs_targ, nans]).T.flatten()
    ys_line = np.array([ys_src, ys_targ, nans]).T.flatten()
    
    return np.array([xs_line, ys_line])
