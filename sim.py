"""
Code for aspects of simulation not directly related to network dynamics.
"""
import numpy as np
import os
import shelve


class RandomTraj(object):

    """
    Random trajectory through space.
    :param ts: timestamp vector
    :param speed: STD of expected speed distribution (m/s)
    :param smoothness: timescale of velocity changes (s)
    :param x_0: starting location (length-2 array) (m)
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
