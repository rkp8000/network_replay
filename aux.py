import numpy as np
import os
import shelve


def save_time_file(save_file, ts):
    """
    Save a file containing a set of time stamps. Sampling frequency is computed
    from the mean time interval in ts.

    :param save_file: path of file to save (do not include .db extension)
    :param ts: 1D timestamp array
    """
    # make sure save directory exists
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # delete file if it already exists
    if os.path.exists(save_file + '.db'):
        os.remove(save_file + '.db')
    
    data = shelve.open(save_file)
    data['timestamps'] = ts
    data['fs'] = 1 / np.mean(np.diff(ts))

    data.close()

    return save_file


def load_time_file(time_file):
    """
    Return the timestamp array and sampling frequency from a timestamp file.
    :param time_file: path to file containing timestamp array
    :return: timestamp array, sampling frequency
    """
    data_t = shelve.open(time_file)

    for key in ('timestamps', 'fs'):
        if key not in data_t:
            raise KeyError('Item with key "{}" not found in file "{}".'.format(key, time_file))

    return data_t['timestamps'], data_t['fs']

