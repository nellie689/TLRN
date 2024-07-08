import random
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy
import torch.nn.functional as F
import torchvision.transforms as transforms
 
class RandomRotation90:
    def __call__(self, image):
        angle = random.choice([0, 90, 180, 270])
        return transforms.functional.rotate(image, angle)
class DENSE(Dataset):
    def __init__(self, path, split='train', trainlenscale=1, testlen=150, cur_img_size=64, resmode="pair", series_len=9, basesize=64, pathmask=None): #resmode="series"
        self.resmode = resmode
        if "lemniscate" in "path":
            self.data_transform = None
        else:
            if split == "test":
                self.data_transform = None
            else:
                self.data_transform = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    RandomRotation90(),
                    # transforms.ToTensor()
                ])

        self.series_len = series_len
        self.split = split
        self.cur_img_size = cur_img_size
        factor = cur_img_size/basesize
        self.trainlenscale = trainlenscale
        
        data = scipy.io.loadmat(path)
        data = data['data']   #(178, 9, 64, 64)   (795, 12, 64, 64)
        l,c,w,h = data.shape

        data = torch.from_numpy(data.astype(np.float32))  #(178, 9, 64, 64)
        self.data = F.interpolate(data, scale_factor=factor, mode='bilinear', align_corners=False)
        
        if pathmask is not None:  ## load mask
            mask = scipy.io.loadmat(pathmask)
            mask = mask['data']
            self.mask = torch.from_numpy(mask.astype(np.float32))  #torch.Size([492, 12, 128, 128])
        else:
            self.mask = None

        self.trainlen = int(100/self.trainlenscale)
        self.testlen = testlen

        if l == 491:
            self.train_data_len = 388
        elif l<200:
            self.train_data_len = 150
        elif l<700:
            self.train_data_len = 442
        elif l<800:
            self.train_data_len = 700
        elif l<1000:
            self.train_data_len = 800
        else:
            self.train_data_len = 1000

        self.test_data_len = l - self.train_data_len
        self.len = self.__len__()

    def __len__(self):
        return self.trainlen if self.split == "train" else min(self.testlen, self.test_data_len)

    def __getitem__(self, idx):
        if self.resmode == "pair" or self.resmode == "pair-no-skip" or self.resmode == 'pair-skip':
            indices = np.random.randint(self.train_data_len, size=1) if self.split == "train" else np.random.randint(self.test_data_len, size=1)+self.train_data_len
        elif self.resmode != "pair" and self.split == "train":
            indices = np.random.randint(self.train_data_len, size=1) if self.split == "train" else np.random.randint(self.test_data_len, size=1)+self.train_data_len
        elif self.resmode != "pair" and self.split == "test":
            indices = np.array([idx]) % self.test_data_len   +self.train_data_len

        slices = self.data[indices]
        if not(self.mask is None):
            masks = self.mask[indices] 
        else:
            masks = None

        if self.resmode == "voxelmorph":
            if self.data_transform:
                slices = self.data_transform(slices)

            src = slices[:,0:1,...] 
            src = src.repeat(1, self.series_len-1, 1, 1) 
            tar = slices[:,1:self.series_len,...]  
            
            if not(self.mask is None):
                if self.data_transform:
                    masks = self.data_transform(masks)
                mask_src = masks[:,0:1,...]
                mask_src = mask_src.repeat(1, self.series_len-1, 1, 1)
                mask_tar = masks[:,1:self.series_len,...]
                sample = {'src': src[0], 'tar': tar[0], 'mask_src': mask_src[0], 'mask_tar': mask_tar[0]}
            else:
                sample = {'src': src[0], 'tar': tar[0]}
        else:
            slices = slices[:, : self.series_len, ]
            if self.data_transform:
                slices = self.data_transform(slices)
            if not(self.mask is None):
                masks = masks[:, : self.series_len, ]
                if self.data_transform:
                    masks = self.data_transform(masks)
                sample = {'series': slices, 'masks': masks}
            else:
                sample = {'series': slices}
        return sample
    

