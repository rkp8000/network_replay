import imageio
import os


def create_gif(files, save_prefix, fps):
    """
    Create a gif from a sequence of image files.
    """
    
    # make sure save directory exists
    save_dir = os.path.dirname(save_prefix)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
        
    images = [imageio.imread(file) for file in files]
    imageio.mimsave(save_prefix + '.gif', images, fps=fps)
