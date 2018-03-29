import imageio
import numpy as np
import os
from PIL import Image


def create_mp4(
        files, save_file, playback_fps, delete_originals=False, verbose=False):
    """
    Create an MP4 movie from a sequence of image files.
    
    :param files: ordered list of paths to files to be merged into MP4
    :param save_file: prefix (without ".mp4") of file to save MP4 to
    :param playback_fps: fps of MP4 given files (max 30 Hz)
    :param delete_originals: whether to delete files used to make original MP4
    :param verbose: whether to print out progress
    """
    
    if playback_fps > 30:
        print('Warning: playback rates > 30 Hz are usually rendered incorrectly')
        
    # make sure save directory exists
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
        
    if verbose:
        print('Loading source images...')
        
    images = [imageio.imread(file) for file in files]
    
    if verbose:
        print('Source images loaded.')
        print('Generating movie and saving at "{}"...'.format(save_file + '.mp4'))
        
    imageio.mimsave(save_file + '.mp4', images, fps=playback_fps)
    
    if verbose:
        print('Movie generated.')
    
    if delete_originals:
        if verbose:
            print('Deleting originals...')
            
        for file in files:
            os.remove(file)
            
        if verbose:
            print('Originals deleted.')
            
    return save_file + '.mp4'


# AUXILIARY FUNCTIONS
def random_oval(cent, rad, n):
    """
    Generate n points uniformly distributed within an oval.
    
    :param cent: (center x, center y)
    :param rad: (radius x, radius y)
    :param n: number of points to sample
    """
    t = 2 * np.pi * np.random.rand(n)
    u = np.random.rand(n) + np.random.rand(n)
    r = np.nan * np.zeros(n)
    r[u < 1] = u[u < 1]
    r[u >= 1] = 2 - u[u >= 1]
    
    x = cent[0] + (rad[0] * r * np.cos(t))
    y = cent[1] + (rad[1] * r * np.sin(t))
    
    return np.array([x, y]).T
