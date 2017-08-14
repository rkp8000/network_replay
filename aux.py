import numpy as np
import os


def save(save_file, obj):
    """
    Save a python object to a file using np.save.
    
    :param save_file: path to save file (should have .npy extension)
    :param obj: python object to save
    :return: path to saved file
    """
    if len(save_file) < 4 or save_file[-4:].lower() != '.npy':
        raise ValueError('Saved file must end with ".npy" extension.')
        
    # make sure save directory exists
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
 
    # delete file if it already exists
    # if os.path.exists(save_file):
    #     os.remove(save_file)
    
    np.save(save_file, np.array([obj]))
    return save_file


def load(load_file):
    """
    Load a python object using np.load.
    
    :param load_file: path to file containing object
    :return: loaded python object
    """
    if load_file[-4:].lower() != '.npy':
        raise ValueError('Load file must end with ".npy"')
        
    return np.load(load_file)[0]


def save_time_file(save_file, ts):
    """
    Save a file containing a set of time stamps. Sampling frequency is computed
    from the mean time interval in ts.

    :param save_file: path of file to save
    :param ts: 1D timestamp array
    :return: path to saved file
    """
    data = {'timestamps': ts, 'fs': 1 / np.mean(np.diff(ts))}
    return save(save_file, data)


def load_time_file(time_file):
    """
    Return the timestamp array and sampling frequency from a timestamp file.
    :param time_file: path to file containing timestamp array
    :return: timestamp array, sampling frequency
    """
    data_t = load(time_file)

    for key in ('timestamps', 'fs'):
        if key not in data_t:
            raise KeyError('Item with key "{}" not found in file "{}".'.format(key, time_file))

    return data_t['timestamps'], data_t['fs']
