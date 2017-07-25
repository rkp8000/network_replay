import imageio
import os


def create_gif(files, save_file, playback_fps, delete_originals=False, verbose=False):
    """
    Create a gif from a sequence of image files.
    
    :param files: ordered list of paths to files to be merged into gif
    :param save_file: prefix (without ".gif") of file to save gif to
    :param playback_fps: fps of gif given files (max 30 Hz)
    :param delete_originals: set to True to delete files used to make original gif
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
