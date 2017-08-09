"""
Code for aspects of simulation not directly related to network dynamics.
"""
import numpy as np
import os
import shelve

from aux import load_time_file


class RandomTraj(object):

    """
    Random trajectory through space.
    :param ts: timestamp vector
    :param speed: STD of expected speed distribution (m/s)
    :param smoothness: timescale of velocity changes (s)
    :param xy_0: starting location (length-2 array) (m)
    :param v_0: starting velocity (length-2 array) (m/s)
    :param box: box boundaries (left, right, bottom, top) (m)
    :return: position sequence, velocity sequence
    """

    def __init__(self, ts, speed, smoothness, xy_0, v_0, box):

        # allocate space for positions and velocities
        xys = np.nan * np.zeros((len(ts), 2))
        vs = np.nan * np.zeros((len(ts), 2))

        # calculate noise scale yielding desired speed
        dt = np.mean(np.diff(ts))
        sig = np.sqrt(2*smoothness) * speed / np.sqrt(dt)

        # generate traj
        xys[0] = xy_0
        vs[0] = v_0

        for step in range(1, len(ts)):

            # calculate tentative change in velocity and position
            dv = (dt/smoothness) * (-vs[step-1] + sig*np.random.randn(2))
            v = vs[step-1] + dv

            dxy = v*dt
            xy = xys[step-1] + dxy

            # reflect position/velocity off walls if position out of bounds

            # horizontal
            if xy[0] < box[0]:
                v[0] *= -1
                xy[0] = 2*box[0] - xy[0]

            elif xy[0] >= box[1]:
                v[0] *= -1
                xy[0] = 2*box[1] - xy[0]

            # vertical
            if xy[1] < box[2]:
                v[1] *= -1
                xy[1] = 2*box[2] - xy[1]

            elif xy[1] >= box[3]:
                v[1] *= -1
                xy[1] = 2*box[3] - xy[1]

            # store final velocity and position
            xys[step] = xy.copy()
            vs[step] = v.copy()

        self.xys = xys
        self.vs = vs

    def save(self, save_file):
        """
        Save traj to a file.

        :param save_file: path to file to save
        """

        # make sure save directory exists
        save_dir = os.path.dirname(save_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # open file and save traj data
        data = shelve.open(save_file)
        data['xys'] = self.xys
        data.close()

        return save_file


def upstream_spikes_from_positions(ts, xys, centers, stds, max_rates):
    """
    Generate a set of "upstream" spikes from a trajectory sequence
    given the tuning curves of the cells.

    Tuning curves are assumed to be squared exponential in shape,
    specified by centers and widths (corresponding to the mean and
    std of a 2D Gaussian).

    :param ts: timestamp array
    :param xys: position array (T x 2)
    :param centers: tuning curve centers for each neuron (2 x N)
    :param stds: tuning curve widths for each neuron (N-length array)
    :param max_rates: spike rate for tuning curve max for each neuron
    :return: upstream spike trains (T x N)
    """

    n_steps = len(ts)
    n = centers.shape[1]

    if not len(xys) == len(ts):
        raise ValueError('Argument "xys" must have same length as "ts".')
    if not (stds.ndim == 1 and len(stds) == n):
        raise ValueError('Argument "stds" must be 1-D length-N array.')
    if not (max_rates.ndim == 1 and len(max_rates) == n):
        raise ValueError('Argument "max_rates" must be 1-D length-N array.')

    dt = np.mean(np.diff(ts))

    # get displacement of trajectory to each center over time
    dxs_1 = np.tile(xys[:, [0]], (1, n)) - np.tile(centers[[0], :], (n_steps, 1))
    dxs_2 = np.tile(xys[:, [1]], (1, n)) - np.tile(centers[[1], :], (n_steps, 1))

    # get firing rates
    stds = np.tile(stds[None, :], (n_steps, 1))
    max_rates = np.tile(max_rates[None, :], (n_steps, 1))

    spk_rates = max_rates * np.exp(-(1/(stds**2)) * (dxs_1**2 + dxs_2**2))
    mean_spk_cts = spk_rates * dt

    # randomly generate spikes
    spks = np.random.poisson(mean_spk_cts, mean_spk_cts.shape)

    return spks


class InferredTraj(object):
    """
    Trajectory inferred from network activity.
    
    :param ntwk_file: path to network activity file
    :param time_file: path to timestamp file
    :param window: time window for calculating positions in (s)
    """
    
    @staticmethod
    def infer_positions(ts, spks, window, place_field_centers):
        """
        Infer a sequence of positions and uncertainties from
        the spike trains of a set of place cells and their place field centers.
        
        :param ts: timestamp sequence
        :param spks: multi-cell spike train (rows are time points, cols are cells)
        :param window: length of window (s) to use to count spikes over
        :param place_field_centers: place-field centers of all cells
            (rows are x, y; cols are cells)
        """
        if not len(ts) == len(spks):
            raise ValueError('Spike array must be same length as timestamp array.')
        
        if not spks.shape[1] == place_field_centers.shape[1]:
            raise ValueError('Spike array must have same cols as place_field_centers.')
            
        if not place_field_centers.shape[0] == 2:
            raise ValueError('Place field centers must have 2 rows.')
        
        # loop over all windows
        xys = np.nan * np.zeros((len(ts), 2))
        covs = np.nan * np.zeros((len(ts), 2, 2))
        
        for start in np.arange(0, ts[-1], window):
            end = start + window
            mask = (ts >= start) & (ts < end)
            
            # get spike counts for this window
            spk_cts = spks[mask].sum(axis=0)
            
            # get position means and covariances, weighted by spike counts
            if np.any(spk_cts):
                xy = np.average(place_field_centers, axis=1, weights=spk_cts)
                cov = np.cov(place_field_centers, fweights=spk_cts)
            else:
                xy = np.nan * np.zeros(2)
                cov = np.nan * np.zeros((2, 2))
            
            # store results
            xys[mask] = xy
            covs[mask] = cov
            
        return xys, covs
        
    
    def __init__(self, ntwk_file, time_file, window):
        """Constructor."""
        self.ntwk_file = ntwk_file
        self.time_file = time_file
        self.window = window
        
        # extract timestamps
        ts = load_time_file(time_file)[0]
        
        # extract place-field centers and spike counts
        data = shelve.open(ntwk_file)
        
        if ('spikes' not in data) or ('place_field_centers' not in data):
            raise KeyError(
                'Network activity file must contain spiking activity '
                'and place field centers.'
            )
        
        xys, covs = self.infer_positions(
            ts, data['spikes'], window, data['place_field_centers'])
        
        data.close()
        
        self.xys = xys
        self.covs = covs
        
    def save(self, save_file):
        """
        Save inferred position sequence to file.
        
        :param save_file: path to save file
        """
        data = shelve.open(save_file)
        data['xys'] = self.xys
        data['covs'] = self.covs
        data.close()
        
        return save_file
