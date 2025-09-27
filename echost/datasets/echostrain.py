import os
import skimage.draw
import torchvision
import collections
import torch
import cv2
import imageio
import PIL
import numpy    as np
import pandas   as pd

class EchoStrain(torchvision.datasets.VisionDataset):

    def __init__(self, root=None, split="train", mean=0., std=1.):
        super().__init__(root)

        self.split      = split.upper()
        self.mean       = mean
        self.std        = std

        # Initialize attributes
        self.images     = []
        self.masks      = []
        self.header     = []

        # Automatically set paths based on split
        self.imgs_root  = os.path.join(root, self.split, "imgs", f"CAMUS_{self.split.upper()}")
        self.masks_root = os.path.join(root, self.split, "masks", f"CAMUS_{self.split.upper()}")


        self.images = sorted([
                f for f in os.listdir(self.imgs_root)
                if f.lower().endswith(".png")
            ])
        
        self.masks = sorted([
                f for f in os.listdir(self.masks_root)
                if f.lower().endswith(".png")
            ])

    def __getitem__(self, index):
        # images and masks have same name
        image_path  = os.path.join(self.imgs_root, self.images[index]) 
        mask_path   = os.path.join(self.masks_root, self.images[index])

        img         = PIL.Image.open(image_path).convert('L')   # grayscale
        img         = np.array(img, dtype=np.float32) / 255.0
        img         = np.expand_dims(img, 0)  # (1,H,W)
    
        mask        = PIL.Image.open(mask_path)

        # Masks, classes are represented by integers
        mask = np.array(mask, dtype=np.uint8)  # (H,W)
        img  = self.preprocess(img)

        return {
            "image": torch.from_numpy(img).float(),     # (1,H,W)
            "mask" : torch.from_numpy(mask).long()      # (H,W)
        }
    
    def __len__(self):
        return len(self.images)

    def load_video_labels(self):
        pass
    
    def preprocess(self, img):

        if self.mean != 0. and self.std != 1.:
            img = (img - self.mean) / self.std 
        
        return img




def loadvideo(filename:str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError
    
    # Open video
    capture         = cv2.VideoCapture(filename)

    frame_count     = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width     = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height    = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v               = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)# (F ,H, W, C) 


    # Read video frame by frame  
    for count in range(frame_count):
        ret, frame  = capture.read()# If ret is True, reading is succesful 
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame           = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB 
        v[count, :, :]  = frame

    v = v.transpose((3, 0, 1, 2)) # (C, F, H, W)   

    return v