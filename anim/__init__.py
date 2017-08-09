import imageio
import numpy as np
import os
from PIL import Image


def merge_frames(frame_sets, save_prefix, rects, size, size_rel_to=1, delete_originals=False, verbose=False):
    """
    Create merged frames from multiple sets of frames.
    
    :param frame_sets: list of sets of frames to merge
    :param save_prefix: filename prefix for each saved merged frame
    :param rects: list of rectangular locations [(left, upper, right, lower), ...] for
        sub-images, with (0, 0) in upper left (relative to the image size)
    :param size: size of final image (relative to size of first frame set)
    :param size_rel_to: change for "size" argument to be relative to a different frame set
    :param delete_originals: whether to delete files used to make original gif
    :param verbose: whether to print out progress
    """
    
    # check arguments
    if not all([len(fs) for fs in frame_sets]):
        raise ValueError('All frame sets must have same number of frames.')
    
    if len(frame_sets) != len(rects):
        raise ValueError('Each frame set must have exactly one corresponding rect.')
    
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
        
    size_px = (size[0]*sizes_orig[size_rel_to][0], size[1]*sizes_orig[size_rel_to][1])
        
    for rect_px in rects_px:
        if not rect_px[0] < rect_px[2]:
            raise ValueError('Rect left must be less than rect right.')
        if not rect_px[1] < rect_px[3]:
            raise ValueError('Rect upper must be less than rect lower.')
    
    if (not size_px[0] >= np.max([rect_px[2] for rect_px in rects_px])) \
            or (not size_px[1] >= np.max([rect_px[3] for rect_px in rects_px])):
        raise ValueError('Merged frame size must be larger than sub-image size.')
        
    sub_sizes = [(rect_px[2]-rect_px[0], rect_px[3]-rect_px[1]) for rect_px in rects_px]
    
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
    

def create_movie(files, save_file, playback_fps, delete_originals=False, verbose=False):
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
