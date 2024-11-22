import os
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn 
import torch.optim as optim
import pytorch_lightning as pl 
import torchvision
import albumentations 
import matplotlib
matplotlib.use('Agg')

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2


# global variables
TRAIN_DIR = "/home/space/datasets/pix2pix2/pix2pix-dataset/maps/maps/train"
VAL_DIR = "/home/space/datasets/pix2pix2/pix2pix-dataset/maps/maps/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500


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
            image = self.transformation(image=image)            
        return image
    
    
class transformation:
    def __call__(self,image):
        
        data_transform = albumentations.Compose([
        albumentations.Resize(256, 512),
        albumentations.CenterCrop(256, 512)
        ])
        
        sat_transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.2), 
        albumentations.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        albumentations.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), 
        albumentations.pytorch.ToTensorV2()
        ])
        
        
        map_transform = albumentations.Compose([
        albumentations.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        albumentations.pytorch.ToTensorV2()
        ])
        
        transformed = data_transform(image=image)
        image = transformed["image"]
        sat = image[:,:256,:]
        map = image[:,256:,:]
        transformed_sat = sat_transform( image = sat)
        transformed_map = map_transform( image = map)
        
        return  transformed_sat["image"], transformed_map["image"]

class myDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    
    def setup(self, stage=None):   
        self.transformation = transformation()
        self.train_dataset = myDataset(root_dir= TRAIN_DIR, transformation = self.transformation)
        self.val_dataset = myDataset(root_dir= VAL_DIR, transformation = self.transformation)
        self.test_dataset =  myDataset(root_dir= VAL_DIR, transformation = self.transformation)
        return 0

    def train_dataloader(self):   
        return DataLoader(
            self.train_dataset,
            batch_size= BATCH_SIZE,
            shuffle=True,
            num_workers= 4
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=4,  
            shuffle=True, 
            num_workers=4
            )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4
            )

class genBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, down=True, activation="relu", use_dropout=False):
        super(genBlock, self).__init__()
        
        if down:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False, padding_mode = "reflect")
        else:
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias=False)#,
        
        if activation == "relu":
            act = nn.ReLU()
        else:
            act = nn.LeakyReLU(0.2)
        
        self.conv = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            act,
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)

        if self.use_dropout:
            return self.dropout(x) 
        else:
            return x


class Generator(pl.LightningModule): 

    def __init__(self,in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size = 4, stride = 2, padding = 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2), 
        )
        self.down1 = genBlock(features*1, features*2, down=True, activation="leaky", use_dropout=False)
        self.down2 = genBlock(features*2, features*4, down=True, activation="leaky", use_dropout=False)
        self.down3 = genBlock(features*4, features*8, down=True, activation="leaky", use_dropout=False)
        self.down4 = genBlock(features*8, features*8, down=True, activation="leaky", use_dropout=False)
        self.down5 = genBlock(features*8, features*8, down=True, activation="leaky", use_dropout=False)
        self.down6 = genBlock(features*8, features*8, down=True, activation="leaky", use_dropout=False)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(features* 8, features*8, kernel_size = 4, stride = 2, padding = 1), nn.ReLU())

        self.up1 = genBlock(features*8*1, features*8, down=False, activation="relu", use_dropout=True)
        self.up2 = genBlock(features*8*2, features*8, down=False, activation="relu", use_dropout=True)
        self.up3 = genBlock(features*8*2, features*8, down=False, activation="relu", use_dropout=True)
        self.up4 = genBlock(features*8*2, features*8, down=False, activation="relu", use_dropout=False)
        self.up5 = genBlock(features*8*2, features*4, down=False, activation="relu", use_dropout=False)
        self.up6 = genBlock(features*4*2, features*2, down=False, activation="relu", use_dropout=False)
        self.up7 = genBlock(features*2*2, features*1, down=False, activation="relu", use_dropout=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))

class CNNBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = stride, padding = 1, bias = False, padding_mode = "reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(pl.LightningModule):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size = 4, stride = 2, padding = 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append( CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2), )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = "reflect"),
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x

class Pix2Pix(LightningModule):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels=in_channels

        self.BCE = nn.BCEWithLogitsLoss()
        self.L1 = nn.L1Loss()

        self.gen  = Generator(in_channels=3, features=64)
        self.disc = Discriminator(in_channels=3)

    def forward(self, input_image): 
        return self.gen(input_image)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x,y = batch 
       
        # train generator
        if optimizer_idx == 0:

            y_fake = self.gen(x) 
            D_fake = self.disc(x, y_fake)

            G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake)) 
            L1 = self.L1(y_fake, y) * L1_LAMBDA 
            G_loss = G_fake_loss + L1

            self.log("G_loss", G_loss, prog_bar=True)
            return G_loss

        # train discriminator
        if optimizer_idx == 1:
            y_fake = self.gen(x) 
            D_real = self.disc(x, y)
            D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
            D_fake = self.disc(x, y_fake.detach()) 
            D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            self.log("D_loss", D_loss, prog_bar=True)
            return D_loss

    def configure_optimizers(self):
        LR = LEARNING_RATE
        B1 = 0.5
        B2 = 0.999

        opt_disc = optim.Adam(self.disc.parameters(), lr=LEARNING_RATE, betas=(B1, B2),)
        opt_gen = optim.Adam(self.gen.parameters(), lr=LEARNING_RATE, betas=(B1, B2))

        return [opt_gen, opt_disc], []

    def validation_step(self, batch, batch_idx):
        x,y = batch        

        if batch_idx == 0:
          self.sample = x,y 

        if True:
            y_fake = self.gen(x)
            D_fake = self.disc(x, y_fake)

            G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake)) 
            L1 = self.L1(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1
            
            self.log("G_val_loss", G_loss, prog_bar=True)
            self.log("G_val_l1", L1, prog_bar=True)

        if True:
            y_fake = self.gen(x) 
            D_real = self.disc(x, y)
            D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
            D_fake = self.disc(x, y_fake.detach()) 
            D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            self.log("D_val_loss", D_loss, prog_bar=True)


    def on_validation_epoch_end(self):
        satellite_images = self.sample[0][:] 
        true_map_images = self.sample[1][:]
        generated_map_images = self.gen(satellite_images) 

        for i in range(len(generated_map_images)):
            generated_maps_grid = torchvision.utils.make_grid(generated_map_images)
            true_maps_grid = torchvision.utils.make_grid(true_map_images)
            satellites_grid = torchvision.utils.make_grid(satellite_images)

            torchvision.utils.save_image(satellite_images[i]* 0.5 + 0.5,"evaluation/"+str(self.current_epoch)+"_satellite_"+str(i)+".jpg")
            torchvision.utils.save_image(true_map_images[i],"evaluation/"+str(self.current_epoch)+"_true_map_"+str(i)+".jpg")
            torchvision.utils.save_image(generated_map_images[i],"evaluation/"+str(self.current_epoch)+"_gen_map"+str(i)+".jpg")
            
            self.logger.experiment.add_image("generated_map_images", generated_maps_grid, self.current_epoch)
            self.logger.experiment.add_image("true_map_images", true_maps_grid, self.current_epoch)
            self.logger.experiment.add_image("satellite_images", satellites_grid, self.current_epoch)

from pytorch_lightning import loggers as pl_loggers
tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs1/")

data_module = myDataModule()
data_module.setup()

model = Pix2Pix()
trainer = pl.Trainer(fast_dev_run=True,logger=tb_logger, max_epochs = NUM_EPOCHS,accelerator="auto", devices=1 if torch.cuda.is_available() else None) 
trainer.fit(model,data_module)













###### To plot images in Notebook/Colab
#for batch in data_module.train_dataloader():
#  # you get a batch of input images and a batch of target images
#  print(len(batch[0]),len(batch[1]))#.shape)
#  print(batch[0].shape,batch[1].shape)#.shape)
#  first_batch = batch
#  break
#for i in range(1,10):
#    torchvision.utils.save_image(first_batch[0][i] * 0.5 + 0.5,str(i)+".jpg")

import warnings
import matplotlib.pyplot as plt

def show_batch(x,save_file=None,figsize = (30,60),nrow=None):
    plt.figure(figsize = figsize) 
    batch = (torch.cat(x,axis=1)* 0.5 + 0.5)
    batch_len,channels,rows,cols = batch.shape
    batch = batch.reshape(batch_len*2,channels//2,rows,cols)
    print(batch.shape)
    if not nrow:
      nrow= batch_len
    if not batch.shape[0] % nrow ==0:
      warnings.warn("Number of rows must be divisible by the number of rows. Setting number of rows to batch size.")
      nrow= batch_len

    nrow = batch.shape[0]//nrow 
    grid_img = torchvision.utils.make_grid(batch,nrow=nrow)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    if save_file:
      torchvision.utils.save_image(grid_img,save_file)

def show_image(image,save_file=None,figsize = (5,5)):
    plt.figure(figsize = figsize)

    plt.imshow((image * 0.5 + 0.5).permute(1, 2, 0))
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    if save_file:
      torchvision.utils.save_image(image * 0.5 + 0.5,save_file)

def show_pair(batch,index,save_file=None,figsize = (5,5)):
    batch = [batch[0][[index]],batch[1][[index]]]
    show_batch(batch,save_file=save_file,nrow=3)

#show_batch(first_batch)
#show_pair(first_batch,1) 
#show_image(first_batch[0][1])
