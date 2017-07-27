"""
Code for aspects of simulation not directly related to network dynamics.
"""
import numpy as np


def random_traj(ts, speed, smoothness, x_0, v_0, box):
    """
    Generate a random trajectory through space.
    :param ts: timestamp vector
    :param speed: STD of expected speed distribution (m/s)
    :param smoothness: timescale of velocity changes (s)
    :param x_0: starting location (length-2 array) (m)
    :param v_0: starting velocity (length-2 array) (m/s)
    :param box: box boundaries (left, right, bottom, top) (m)
    :return: position sequence, velocity sequence
    """

    # allocate space for positions and velocities
    xs = np.nan * np.zeros((len(ts), 2))
    vs = np.nan * np.zeros((len(ts), 2))

    # calculate noise scale yielding desired speed
    dt = np.mean(np.diff(ts))
    sig = np.sqrt(2*smoothness) * speed / np.sqrt(dt)

    # generate traj
    xs[0] = x_0
    vs[0] = v_0

    for step in range(1, len(ts)):

        # calculate tentative change in velocity and position
        dv = (dt/smoothness) * (-vs[step-1] + sig*np.random.randn(2))
        v = vs[step-1] + dv

        dx = v*dt
        x = xs[step-1] + dx

        # reflect position/velocity off walls if position out of bounds

        # horizontal
        if x[0] < box[0]:
            v[0] *= -1
            x[0] = 2*box[0] - x[0]

        elif x[0] >= box[1]:
            v[0] *= -1
            x[0] = 2*box[1] - x[0]

        # vertical
        if x[1] < box[2]:
            v[1] *= -1
            x[1] = 2*box[2] - x[1]

        elif x[1] >= box[3]:
            v[1] *= -1
            x[1] = 2*box[3] - x[1]

        # store final velocity and position
        xs[step] = x.copy()
        vs[step] = v.copy()

    return xs, vs


def upstream_spikes_from_positions(ts, xs, centers, stds, max_rates):
    """
    Generate a set of "upstream" spikes from a trajectory sequence
    given the tuning curves of the cells.

    Tuning curves are assumed to be squared exponential in shape,
    specified by centers and widths (corresponding to the mean and
    std of a 2D Gaussian).

    :param ts: timestamp array
    :param xs: position array (T x 2)
    :param centers: tuning curve centers for each neuron (2 x N)
    :param stds: tuning curve widths for each neuron (N-length array)
    :param max_rates: spike rate for tuning curve max for each neuron
    :return: upstream spike trains (T x N)
    """

    n_steps = len(ts)
    n = centers.shape[1]

    if not len(xs) == len(ts):
        raise ValueError('Argument "xs" must have same length as "ts".')
    if not (stds.ndim == 1 and len(stds) == n):
        raise ValueError('Argument "stds" must be 1-D length-N array.')
    if not (max_rates.ndim == 1 and len(max_rates) == n):
        raise ValueError('Argument "max_rates" must be 1-D length-N array.')

    dt = np.mean(np.diff(ts))

    # get displacement of trajectory to each center over time
    dxs_1 = np.tile(xs[:, [0]], (1, n)) - np.tile(centers[[0], :], (n_steps, 1))
    dxs_2 = np.tile(xs[:, [1]], (1, n)) - np.tile(centers[[1], :], (n_steps, 1))

    # get firing rates
    stds = np.tile(stds[None, :], (n_steps, 1))
    max_rates = np.tile(max_rates[None, :], (n_steps, 1))

    spk_rates = max_rates * np.exp(-(1/(stds**2)) * (dxs_1**2 + dxs_2**2))
    mean_spk_cts = spk_rates * dt

    # randomly generate spikes
    spks = np.random.poisson(mean_spk_cts, mean_spk_cts.shape)

    return spks
