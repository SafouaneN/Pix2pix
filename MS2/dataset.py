import albumentations 
import os
import pytorch_lightning as pl 
import numpy as np

from PIL import Image
from torch._C import _test_only_remove_upgraders
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2

#import config

class transformation:
    def __call__(self,image): # +hparams
        
        data_transform = albumentations.Compose([
        albumentations.Resize(256, 512),
        albumentations.CenterCrop(256, 512)
        ])
        
        sat_transform = albumentations.Compose([
        # albumentations.HorizontalFlip(p=0.2), 
        albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        albumentations.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), 
        albumentations.pytorch.ToTensorV2()
        ])
        
        map_transform = albumentations.Compose([
        albumentations.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        albumentations.pytorch.ToTensorV2()
        ])
        
        
        flip = albumentations.Compose([
            albumentations.HorizontalFlip(p=1.0)
        ])
        

        transformed = data_transform(image=image)
        image = transformed["image"]
        sat = image[:,:256,:]
        map = image[:,256:,:]

        if np.random.rand()>0.5:
            sat = flip( image = sat)["image"]
            map = flip( image = map)["image"]
        
        transformed_sat = sat_transform( image = sat)
        transformed_map = map_transform( image = map)
        
        return  transformed_sat["image"], transformed_map["image"]



class myDataset(Dataset):
    def __init__(self, root_dir, transformation = None):
        self.transformation = transformation
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        
    def __len__(self):
        return len(self.list_files)  
    
    def __getitem__(self, index): 
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        
        image = np.array(Image.open(img_path)) 
        
        if self.transformation:
            image = self.transformation(image=image)  # +hparams          
        return image




class myDataModule(pl.LightningDataModule):
    def __init__(self,train_dir=None,val_dir=None,test_dir=None,train_batch_size=16,val_batch_size=4,test_batch_size=4): 
        super().__init__()

        self.train_dir = test_dir
        self.val_dir = test_dir
        self.test_dir = test_dir

        self.train_batch_size = train_batch_size # didnt think of this, had to check documentation, I was thinking of rewriting config file inside program and reloading modules.. I thought I cant change batch size with arguments
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
    
    def setup(self, stage=None):   
        self.transformation = transformation()
        self.train_dataset = myDataset(root_dir= self.test_dir, transformation = self.transformation)
        self.val_dataset = myDataset(root_dir= self.test_dir, transformation = self.transformation)
        self.test_dataset =  myDataset(root_dir= self.test_dir, transformation = self.transformation)
        return 0

    def train_dataloader(self):   
        return DataLoader(
            self.train_dataset,
            batch_size= self.train_batch_size,
            shuffle=True,
            num_workers= 4
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size= self.val_batch_size ,  
            shuffle=False, 
            num_workers=4
            )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size= self.test_batch_size ,
            shuffle=False,
            num_workers=4
            )
