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
