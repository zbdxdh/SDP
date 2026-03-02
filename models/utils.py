from __future__ import print_function
from PIL import Image
import numpy as np
from glob import glob
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Data_load(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform,
                 use_dynamic_mask=False, noise_range=(0.05, 0.35),
                 use_multimodal_data=True, dop_root='./Dataset/DOP'):
        super(Data_load, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.use_dynamic_mask = use_dynamic_mask
        self.noise_range = noise_range
        self.use_multimodal_data = use_multimodal_data
        self.dop_root = dop_root  

        self.paths = glob(f"{img_root}/*", recursive=True)
        
        if use_multimodal_data:
            self.dop_paths = glob(f"{dop_root}/*", recursive=True)
            
            self.paths = sorted(self.paths)
            self.dop_paths = sorted(self.dop_paths)
            assert len(self.paths) == len(self.dop_paths), "AOP和DOP数据集文件数量不一致"

        if not use_dynamic_mask:
            self.mask_paths = glob('{:s}/*.png'.format(mask_root))
            self.N_mask = len(self.mask_paths)

    def __getitem__(self, index):
        aop_img = Image.open(self.paths[index])
        orig_size = aop_img.size
        
        if self.use_multimodal_data:
            dop_img = Image.open(self.dop_paths[index])
            
            aop_tensor = self.img_transform(aop_img.convert('RGB'))
            dop_tensor = self.img_transform(dop_img.convert('RGB'))
            
            gt_img = torch.cat([aop_tensor, dop_tensor], dim=0)
        else:
            gt_img = self.img_transform(aop_img.convert('RGB'))

        if not self.use_dynamic_mask:
            mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
            mask = self.mask_transform(mask.convert('L'))
        else:
            mask = self._generate_random_mask(orig_size)
            if self.mask_transform:
                mask = self.mask_transform(mask)

        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        return gt_img, mask

    def _generate_random_mask(self, size):
        width, height = size

        threshold = random.uniform(self.noise_range[0], self.noise_range[1])

        noise = np.random.rand(height, width)
        mask_array = np.where(noise < threshold, 0, 1).astype(np.float32)

        mask = Image.fromarray((mask_array * 255).astype(np.uint8))

        return mask

    def __len__(self):
        return len(self.paths)
