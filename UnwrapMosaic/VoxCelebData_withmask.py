
# Load the dataset
import torch.utils.data as data
import numpy as np
from torchvision.transforms import ToTensor, Scale, Compose, Pad, RandomHorizontalFlip, CenterCrop, RandomCrop, Resize
from PIL import Image

import torch
import os

def load_img(file_path):
    img = Image.open(file_path).convert('RGB')
    return img

def load_img_mask(file_path):
        img = Image.open(file_path).convert('RGB')
        return img


class VoxCeleb(data.Dataset):
        def __init__(self, num_views, random_seed, dataset):
                super(VoxCeleb, self).__init__()
                self.rng = np.random.RandomState(random_seed)

                # Update the npz files with the name that you downloaded it to from the website
                assert(os.path.exists('/scratch/local/ssd/ow/faces/datasets/voxceleb/landmarks_samevideoimg_%d25thframe_5imgs_%d.npz' % (dataset, num_views)))

                files = np.load('/scratch/local/ssd/ow/faces/datasets/voxceleb/landmarks_samevideoimg_%d25thframe_5imgs_%d.npz' % (dataset, num_views))
                self.image_names = files['image_names']
                self.input_indices = files['input_indices']
                self.landmarks = files['landmarks']
                self.num_views = num_views
                self.transform = Compose([Scale((256,256)), ToTensor()])

        def __len__(self):
                return self.image_names.shape[0] - 1

        def __getitem__(self, index):
                return self.get_blw_item(index)


        def get_blw_item(self, index):
                # Load the images
                imgs = [0] * (self.num_views)

                for i in range(0, self.num_views):
                        img_index = int(self.input_indices[index,i]) - 1
                        img_name = self.image_names[img_index][0].astype(np.str)
                        img_name = img_name.replace('koepke/voxceleb/faces/', 'ow/voxceleb/faces/faces/')
                        imgs[i] = Image.open(img_name)
                        imgs[i] = self.transform(imgs[i])


                return imgs, []
