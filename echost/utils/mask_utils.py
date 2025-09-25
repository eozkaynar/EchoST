import numpy as np

# Dictionary mapping for camus dataset
DEFAULT_RGB_TO_CLASS = {
    (0,   0,   0): 0,  # background
    (255, 0,   0): 1,  # RED - LV endocardium
    (0,   0, 255): 2,  # GREEN - LV cavity
    (0, 255,   0): 3,  # BLUE - myocardium 
}

def rgb2class(rgb_mask,rgb_to_class=None):
    """
    Convert RGB mask image to a class-indexed mask.

    Args:
        rgb_mask (np.ndarray): RGB mask image of shape (H, W, 3)
        rgb_to_class (dict): Optional mapping from RGB tuples to class ids

    Returns:
        np.ndarray: Class-indexed mask of shape (H, W)
    """
    if rgb_to_class is None:
        rgb_to_class = DEFAULT_RGB_TO_CLASS

    height, width   = rgb_mask.shape[:2]
    class_mask      = np.zeros((height, width), dtype=np.uint8)

    for rgb, class_id in rgb_to_class.items():
        match = np.all(rgb_mask == rgb, axis=-1) # Boolean mask where all pixels match the current rgb color
        # Assign the class ID where the match is true
        class_mask[match] = class_id

    return class_mask