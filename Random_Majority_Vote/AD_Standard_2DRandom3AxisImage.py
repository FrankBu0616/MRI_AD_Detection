import nibabel as nib
import os
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize 
from PIL import Image
import random


AX_INDEX = 78
COR_INDEX = 79
SAG_INDEX = 57
AX_SCETION = "[:, :, slice_i]"
COR_SCETION = "[:, slice_i, :]"
SAG_SCETION = "[slice_i, :, :]"


class AD_Standard_2DRandom3AxisImage(Dataset):
    """labeled Faces in the Wild dataset."""
    
    def __init__(self, root_dir, data_file, transform=None, slice = slice):
        """
        Args:
            root_dir (string): Directory of all the images.
            data_file (string): File name of the train/test split file.
            transform (callable, optional): Optional transform to be applied on a sample.
            data_augmentation (boolean): Optional data augmentation.
        """
        self.root_dir = root_dir
        self.data_file = data_file
        self.transform = transform
    
    def __len__(self):
        return sum(1 for line in open(self.data_file))
    
    def __getitem__(self, idx):
        df = open(self.data_file)
        lines = df.readlines()
        lst = lines[idx].split()
        img_name = lst[0]
        img_label = lst[1]
        image_path = os.path.join(self.root_dir, img_name)
        image = nib.load(image_path)
        samples = []
        if img_label == 'Normal':
            label = 0
        elif img_label == 'AD':
            label = 1
        elif img_label == 'MCI':
            label = 2

        AXimageChann = axRandomChannel(image)
        CORimageChann = corRandomChannel(image)
        SAGimageChann = sagRandomChannel(image)

        AXimageChann_std = resize(AXimageChann, (224, 224), mode='reflect', preserve_range=True, anti_aliasing=False)
        AXimageChann_std = resize(CORimageChann, (224, 224), mode='reflect', preserve_range=True, anti_aliasing=False)
        AXimageChann_std = resize(SAGimageChann, (224, 224), mode='reflect', preserve_range=True, anti_aliasing=False)
        
        image2D = np.stack((AXimageChann_std, AXimageChann_std, AXimageChann_std), axis = 2)

        if self.transform:
            image2D = self.transform(image2D)

        sample = {'image': image2D, 'label': label}
        samples.append(sample)
        return samples


def getRandomChannel(image_array, keyIndex, section):
    slice_i = keyIndex
    randomShift = random.randint(-9, 9)
    slice_i = slice_i + randomShift
    slice_select = eval("image_array" + section)
    return slice_select

def axRandomChannel(image):
    image_array = np.array(image.get_data())
    return getRandomChannel(image_array, AX_INDEX, AX_SCETION)


def corRandomChannel(image):
    image_array = np.array(image.get_data())
    return getRandomChannel(image_array, COR_INDEX, COR_SCETION)


def sagRandomChannel(image):
    image_array = np.array(image.get_data())
    return getRandomChannel(image_array, SAG_INDEX, SAG_SCETION)




