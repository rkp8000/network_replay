import imageio
import numpy as np
import os
from PIL import Image


def merge_frames(frame_sets, save_prefix, rects, size, delete_originals=False, verbose=False):
    """
    Create merged frames from multiple sets of frames.
    
    :param frame_sets: list of sets of frames to merge
    :param save_prefix: filename prefix for each saved merged frame
    :param rects: list of rectangular locations [(left, upper, right, lower), ...] for
        sub-images, with (0, 0) in upper left
    :param size: size of final image
    :param delete_originals: whether to delete files used to make original gif
    :param verbose: whether to print out progress
    """
    
    # check validity of arguments
    if not all([len(fs) for fs in frame_sets]):
        raise ValueError('All frame sets must have same number of frames.')
    
    if len(rects) != len(frame_sets):
        raise ValueError('Each frame set must have exactly one corresponding rect.')
        
    for rect in rects:
        if not rect[0] < rect[2]:
            raise ValueError('Rect left must be less than rect right.')
        if not rect[1] < rect[3]:
            raise ValueError('Rect upper must be less than rect lower.')
    
    if (not size[0] >= np.max([rect[2] for rect in rects])) \
            or (not size[1] >= np.max([rect[3] for rect in rects])):
        raise ValueError('Merged frame size must be larger than sub-image size.')
        
    sub_sizes = [(rect[2]-rect[0], rect[3]-rect[1]) for rect in rects]
    
    # make sure save directory exists
    save_dir = os.path.dirname(save_prefix)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # loop over all frames in each frame set
    save_files = []
    
    for f_ctr, frames in enumerate(np.array(frame_sets).T):
        
        # make new image
        img = Image.new('RGBA', size)
        
        # loop over sub images
        for frame, rect, sub_size in zip(frames, rects, sub_sizes):
            
            # open and resize sub image
            sub_img = Image.open(frame)
            if sub_size != sub_img.size:
                sub_img = sub_img.resize(sub_size)
            
            # paste sub image into merged image
            img.paste(sub_img, rect)
        
        # save merged image
        save_file = '{}_{}.png'.format(save_prefix, f_ctr+1)
        save_files.append(save_file)
        
        img.save(save_file)
    
    return save_files
    

def create_gif(files, save_file, playback_fps, delete_originals=False, verbose=False):
    """
    Create a gif from a sequence of image files.
    
    :param files: ordered list of paths to files to be merged into gif
    :param save_file: prefix (without ".gif") of file to save gif to
    :param playback_fps: fps of gif given files (max 30 Hz)
    :param delete_originals: whether to delete files used to make original gif
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
        print('Generating gif...')
        
    imageio.mimsave(save_file + '.gif', images, fps=playback_fps)
    
    if verbose:
        print('Gif generated.')
    
    if delete_originals:
        if verbose:
            print('Deleting originals...')
            
        for file in files:
            os.remove(file)
            
        if verbose:
            print('Originals deleted.')
            
    return save_file + '.gif'
