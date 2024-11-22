#!/usr/bin/env python3
#$ -N pix2pix_cv # name of the experiment
#$ -l cuda=1 # remove this line when no GPU is needed!
#$ -q all.q # do not fill the qlogin queue
#$ -wd /home/pml_18/MS2 # cwd start processes in current directory
#$ -V # provide environment variables


import os
#print(os.getcwd())
#os.chdir("/home/pml_18/MS2")
#print(os.getcwd())

import pickle
import torchvision
import json
import torch
import torch.nn as nn 
import torch.optim as optim #adamw
import pytorch_lightning as pl 
from sklearn.model_selection import ParameterGrid
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning import loggers as pl_loggers
from torchvision.utils import save_image
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs_clean/")



# TODO Solve: import works with "qlogin->python3 pix2pix.py" but not with "qsub pix2pix.py"
#from utils import config
#from utils import dataset
#from utils import models 
#import config # this works if
#import dataset
#import models 

################################## config.py ##################################
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def setup():
    print("installing torch-fidelity...")
    install("torch-fidelity")
    print("done")

###############################################################################
#config.setup()
setup()

################################## dataset.py ##################################
import albumentations 
import os
import pytorch_lightning as pl 
import numpy as np
from PIL import Image
from torch._C import _test_only_remove_upgraders
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2

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
###############################################################################


################################## models.py ##################################
import os
import pickle
import torchvision
import torch
import torch.nn as nn 
import torch.optim as optim #adamw
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from torchvision.utils import save_image
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


class genBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, down=True, activation="relu", use_dropout=False, kernel_size = 4, stride = 2):
        super(genBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
        if down:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = self.kernel_size, stride = self.stride, padding = 1, bias = False, padding_mode = "reflect")
        else:
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = self.kernel_size, stride = self.stride, padding = 1, bias=False)#,
        
        if activation == "relu":
            act = nn.ReLU()
        else:
            act = nn.LeakyReLU(0.2)
        
        self.conv = nn.Sequential(
            conv,
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine=True),
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

    def __init__(self,in_channels=3, features=64, kernel_size = 4, stride = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size = self.kernel_size, stride = self.stride, padding = 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2), 
        )
        self.down1 = genBlock(features*1, features*2, down=True, activation="leaky", use_dropout=False, kernel_size = self.kernel_size, stride = self.stride)
        self.down2 = genBlock(features*2, features*4, down=True, activation="leaky", use_dropout=False, kernel_size = self.kernel_size, stride = self.stride)
        self.down3 = genBlock(features*4, features*8, down=True, activation="leaky", use_dropout=False, kernel_size = self.kernel_size, stride = self.stride)
        self.down4 = genBlock(features*8, features*8, down=True, activation="leaky", use_dropout=False, kernel_size = self.kernel_size, stride = self.stride)
        self.down5 = genBlock(features*8, features*8, down=True, activation="leaky", use_dropout=False, kernel_size = self.kernel_size, stride = self.stride)
        self.down6 = genBlock(features*8, features*8, down=True, activation="leaky", use_dropout=False, kernel_size = self.kernel_size, stride = self.stride)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(features* 8, features*8, kernel_size = self.kernel_size, stride = self.stride, padding = 1), nn.ReLU())

        self.up1 = genBlock(features*8*1, features*8, down=False, activation="relu", use_dropout=True , kernel_size = self.kernel_size, stride = self.stride)
        self.up2 = genBlock(features*8*2, features*8, down=False, activation="relu", use_dropout=True , kernel_size = self.kernel_size, stride = self.stride)
        self.up3 = genBlock(features*8*2, features*8, down=False, activation="relu", use_dropout=True , kernel_size = self.kernel_size, stride = self.stride)
        self.up4 = genBlock(features*8*2, features*8, down=False, activation="relu", use_dropout=False , kernel_size = self.kernel_size, stride = self.stride)
        self.up5 = genBlock(features*8*2, features*4, down=False, activation="relu", use_dropout=False , kernel_size = self.kernel_size, stride = self.stride)
        self.up6 = genBlock(features*4*2, features*2, down=False, activation="relu", use_dropout=False , kernel_size = self.kernel_size, stride = self.stride)
        self.up7 = genBlock(features*2*2, features*1, down=False, activation="relu", use_dropout=False , kernel_size = self.kernel_size, stride = self.stride)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size = self.kernel_size, stride = self.stride, padding=1),
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
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = stride, padding = 1, bias = False, padding_mode = "reflect"),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(pl.LightningModule):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512],kernel_size = 4, stride=2): # some strides are still hard coded 
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size = kernel_size, stride = stride, padding = 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append( CNNBlock(in_channels, feature, kernel_size = kernel_size, stride=1 if feature == features[-1] else stride), )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size = kernel_size, stride = 1, padding = 1, padding_mode = "reflect"),
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x

class Pix2Pix(LightningModule):
    def __init__(self, in_channels = 3, gen_kernel_size = 4, gen_stride = 2, disc_kernel_size = 4, disc_stride = 2, gen_features = 64, disc_features = [64, 128, 256, 512], LR = 2e-4, B1 = 0.5, B2 = 0.999, L1_LAMBDA = 100, output_folder = "test_images/", loss_function = "BCE"): 
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels

        self.loss_function_name = loss_function
        # loss functions
        if loss_function == "BCE":
            self.loss_function = nn.BCEWithLogitsLoss()
        else: # "MSE"
            self.loss_function = nn.MSELoss()

        self.L1 = nn.L1Loss()

        #self.fid = FrechetInceptionDistance(feature=64, normalize=True) # TODO how many features ? Can be one of the following: 64, 192, 768, 2048
        # default normalize=False in documentation
        # didnt work but checked source code and you can *255.byte()
        self.fid = FrechetInceptionDistance(feature=64, normalize=False) 

        self.inception = InceptionScore()

        
        # networks
        self.gen  = Generator(in_channels=in_channels, features = gen_features, kernel_size = gen_kernel_size, stride = gen_stride)
        self.disc = Discriminator(in_channels=in_channels, features = disc_features, kernel_size = disc_kernel_size, stride = disc_stride)

        self.LR = LR
        self.B1 = B1 # 0.5
        self.B2 = B2 # 0.999
        self.L1_LAMBDA = L1_LAMBDA

        self.output_folder = output_folder


    def forward(self, input_image): # 
        return self.gen(input_image)


    def training_step(self, batch, batch_idx, optimizer_idx):
        x,y = batch #real_input,real_output
       
        # train generator
        if optimizer_idx == 0:

            y_fake = self.gen(x) # y_fake or y_gen 
            D_fake = self.disc(x, y_fake)

            G_fake_loss = self.loss_function(D_fake, torch.ones_like(D_fake)) # D_fake should be 1(30x30) which means the discriminator thinks the y_gen is real 
            #G_fake_loss = self.MSE(D_fake, torch.ones_like(D_fake)) # D_fake should be 1(30x30) which means the discriminator thinks the y_gen is real 
            
            L1 = self.L1(y_fake, y) * self.L1_LAMBDA # images should be similar (low L1 distance)
            G_loss = G_fake_loss + L1

            self.log("G_loss", G_loss, on_step=True, on_epoch=True, prog_bar=True)
            return G_loss

        # train discriminator
        if optimizer_idx == 1:
            y_fake = self.gen(x) 
            D_real = self.disc(x, y)
            D_real_loss = self.loss_function(D_real, torch.ones_like(D_real))
            #D_real_loss = self.MSE(D_real, torch.ones_like(D_real))
            D_fake = self.disc(x, y_fake.detach()) # or loss.backward(retain_graph =True)
            D_fake_loss = self.loss_function(D_fake, torch.zeros_like(D_fake))
            #D_fake_loss = self.MSE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            self.log("D_loss", D_loss, on_step=True, on_epoch=True, prog_bar=True)
            return D_loss

    def configure_optimizers(self):
        opt_disc = optim.Adam(self.disc.parameters(), lr=self.LR, betas=(self.B1, self.B2))
        opt_gen = optim.Adam(self.gen.parameters(), lr=self.LR, betas=(self.B1, self.B2))
        # TODO grad scaler and autocast in second milestone lets get an executable code now

        return [opt_gen, opt_disc], []

    def validation_step(self, batch, batch_idx):
        x,y = batch  #real_input,real_output         

        # Save sample to check results
        if batch_idx == 0:
          self.sample = x,y

        # train generator
        if True:

            y_fake = self.gen(x) # y_fake or y_gen
            D_fake = self.disc(x, y_fake)

            G_fake_loss = self.loss_function(D_fake, torch.ones_like(D_fake)) # D_fake should be 1(30x30) which means the discriminator thinks the y_gen is real 
            L1 = self.L1(y_fake, y) * self.L1_LAMBDA # images should be similar (low L1 distance)
            G_loss = G_fake_loss + L1
            
            self.log("G_val_loss", G_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("G_val_l1", L1, on_step=True, on_epoch=True, prog_bar=True)
            #return G_loss

          # train discriminator
        if True:
            y_fake = self.gen(x) # DONE:it does #TODO check if gen accepts batch data should be fine because pl lightning does it
            D_real = self.disc(x, y)
            D_real_loss = self.loss_function(D_real, torch.ones_like(D_real))
            D_fake = self.disc(x, y_fake.detach()) # or loss.backward(retain_graph =True)
            D_fake_loss = self.loss_function(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            self.log("D_val_loss", D_loss, on_step=True, on_epoch=True, prog_bar=True)
            #return D_loss

        self.fid.update((y* 255).byte() , real=True)
        self.fid.update((y_fake* 255).byte() , real=False) # normalize didnt work dunno if its version or colab or etc. problem but I checked actual source code and they do         imgs = (imgs * 255).byte() 
        fid = self.fid.compute()
        self.log("Val_fid", fid, on_step=True, on_epoch=True, prog_bar=True)
        self.fid.reset()

        self.inception.update((y_fake* 255).byte())
        inception = self.inception.compute()
        self.log("Val_inception_0", inception[0], on_step=True, on_epoch=True, prog_bar=True)
        self.log("Val_inception_1", inception[1], on_step=True, on_epoch=True, prog_bar=True)
        self.inception.reset()

    def on_validation_epoch_end(self):
        # TODO more images and information # I only get latest image. #Save every image under path
        satellite_images = self.sample[0][:] 
        true_map_images = self.sample[1][:]
        generated_map_images = self.gen(satellite_images) 
        generated_maps_grid = torchvision.utils.make_grid(generated_map_images)
        true_maps_grid = torchvision.utils.make_grid(true_map_images)
        satellites_grid = torchvision.utils.make_grid(satellite_images)

                    
        self.logger.experiment.add_image("generated_map_images", generated_maps_grid, self.current_epoch)
        self.logger.experiment.add_image("true_map_images", true_maps_grid, self.current_epoch)
        self.logger.experiment.add_image("satellite_images", satellites_grid, self.current_epoch)


        if not os.path.exists(self.output_folder+ "evaluation_"+self.loss_function_name):
            os.makedirs(self.output_folder+ "evaluation_"+self.loss_function_name)

        for i in range(len(generated_map_images)):
            torchvision.utils.save_image(satellite_images[i]* 0.5 + 0.5,"evaluation_"+self.loss_function_name+"/"+str(self.current_epoch)+"_satellite_"+str(i)+".jpg")
            torchvision.utils.save_image(true_map_images[i],"evaluation_"+self.loss_function_name+"/"+str(self.current_epoch)+"_true_map_"+str(i)+".jpg")
            torchvision.utils.save_image(generated_map_images[i],"evaluation_"+self.loss_function_name+"/"+str(self.current_epoch)+"_gen_map_"+str(i)+".jpg")



    def test_step(self, batch, batch_idx):
        x,y = batch  #real_input,real_output         

        # Save sample to check results
        if batch_idx == 0:
          self.sample = x,y 

        # train generator
        if True:

            y_fake = self.gen(x) # y_fake or y_gen
            D_fake = self.disc(x, y_fake)

            G_fake_loss = self.loss_function(D_fake, torch.ones_like(D_fake)) # D_fake should be 1(30x30) which means the discriminator thinks the y_gen is real 
            L1 = self.L1(y_fake, y) * self.L1_LAMBDA # images should be similar (low L1 distance)
            G_loss = G_fake_loss + L1
            
            self.log("G_test_loss", G_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("G_test_l1", L1, on_step=True, on_epoch=True, prog_bar=True)
            #return G_loss

          # train discriminator
        if True:
            y_fake = self.gen(x) # DONE:it does #TODO check if gen accepts batch data should be fine because pl lightning does it
            D_real = self.disc(x, y)
            D_real_loss = self.loss_function(D_real, torch.ones_like(D_real))
            D_fake = self.disc(x, y_fake.detach()) # or loss.backward(retain_graph =True)
            D_fake_loss = self.loss_function(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

            self.log("D_test_loss", D_loss, on_step=True, on_epoch=True, prog_bar=True)
            #return D_loss

        satellite_images = x 
        true_map_images = y
        generated_map_images = self.gen(satellite_images) 

        #if not os.path.exists("test_images"):
        #    os.makedirs("test_images")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        generated_maps_grid = torchvision.utils.make_grid(generated_map_images)
        true_maps_grid = torchvision.utils.make_grid(true_map_images)
        satellites_grid = torchvision.utils.make_grid(satellite_images)

        self.logger.experiment.add_image("generated_map_images", generated_maps_grid, batch_idx)
        self.logger.experiment.add_image("true_map_images", true_maps_grid, batch_idx)
        self.logger.experiment.add_image("satellite_images", satellites_grid, batch_idx)
        #self.logger.experiment.add_image("satellite_images", satellites_grid, self.global_step) # apparently global step not affected by test loops

        for i in range(len(generated_map_images)):

            torchvision.utils.save_image(satellite_images[i]* 0.5 + 0.5,self.output_folder+str(batch_idx)+"_satellite_"+str(i)+".jpg")
            torchvision.utils.save_image(true_map_images[i],self.output_folder+str(batch_idx)+"_true_map_"+str(i)+".jpg")
            torchvision.utils.save_image(generated_map_images[i],self.output_folder+str(batch_idx)+"_gen_map"+str(i)+".jpg")
            
            l1 = self.L1(generated_map_images[i:i+1], true_map_images[i:i+1])
            with open(self.output_folder+str(batch_idx)+"_"+str(i)+".txt","w") as f:
              print("l1 = " + str(l1.item()), file = f)


        self.fid.update((y * 255).byte() , real=True)
        self.fid.update((y_fake * 255).byte() , real=False) # normalize didnt work dunno if its version or colab or etc. problem but I checked actual source code and they do         imgs = (imgs * 255).byte() 

        fid = self.fid.compute()
        self.log("Test_fid", fid, on_step=True, on_epoch=True, prog_bar=True)
        self.fid.reset()
        #return {"satellite_images":satellite_images,"true_map_images":true_map_images,"generated_map_images":generated_map_images}  
        #return {"true_map_images":true_map_images,"generated_map_images":generated_map_images}  
    
        self.inception.update((y_fake* 255).byte())
        inception = self.inception.compute()
        self.log("Test_inception_0", inception[0], on_step=True, on_epoch=True, prog_bar=True)
        self.log("Test_inception_1", inception[1], on_step=True, on_epoch=True, prog_bar=True)
        self.inception.reset()
###############################################################################


def train_apply(method = "pix2pix", dataset_name = "sat2map", train_batch_size=16, val_batch_size=4,test_batch_size=4,in_channels = 3, gen_kernel_size = 4, gen_stride = 2, disc_kernel_size = 4, disc_stride = 2, gen_features = 64, disc_features = [64, 128, 256, 512], LR = 2e-4, B1 = 0.5, B2 = 0.999, L1_LAMBDA = 100, NUM_EPOCHS = 500, output_folder = "test_images/",loss_function = "BCE" ):
    passed_hyper_parameters = locals()
    print(passed_hyper_parameters)
    if dataset_name == "sat2map":
        train_dir = "/home/space/datasets/pix2pix2/pix2pix-dataset/maps/maps/train"
        val_dir = "/home/space/datasets/pix2pix2/pix2pix-dataset/maps/maps/val"
        test_dir = "/home/space/datasets/pix2pix2/pix2pix-dataset/maps/maps/val"

    else:
        print("The required dataset is not available.")
        exit(1)
    
    #data_module = dataset.myDataModule() # TODO
    data_module = myDataModule(train_dir,val_dir,test_dir, train_batch_size, val_batch_size, test_batch_size)
    data_module.setup()

    if method =="pix2pix":
        #model = models.Pix2Pix() # TODO
        model = Pix2Pix(in_channels, gen_kernel_size, gen_stride, disc_kernel_size, disc_stride, gen_features, disc_features, LR, B1, B2, L1_LAMBDA, output_folder, loss_function)
    else:
        print("The required model is not available.")
        exit(1)
        
    trainer = pl.Trainer(fast_dev_run=True,logger=tb_logger, max_epochs = NUM_EPOCHS, check_val_every_n_epoch=5, accelerator="auto", devices=1 if torch.cuda.is_available() else None) #,plugins=DDPPlugin(find_unused_parameters=True))

    trainer.fit(model,data_module)
    test_results = trainer.test(model,[data_module.test_dataloader()])

    return test_results#, passed_hyper_parameters

if False:
    hyper_parameters = dict()
    hyper_parameters["method"]  = "pix2pix"
    hyper_parameters["dataset_name"] = "sat2map"
    hyper_parameters["train_batch_size"] = 16
    hyper_parameters["val_batch_size"]   = 16
    hyper_parameters["test_batch_size"]  = 16
    hyper_parameters["in_channels"]      = 3
    hyper_parameters["gen_kernel_size"]  = 4
    hyper_parameters["gen_stride"]       = 2
    hyper_parameters["disc_kernel_size"] = 4
    hyper_parameters["disc_stride"]      = 2
    hyper_parameters["gen_features"]     = 64
    hyper_parameters["disc_features"]    = [64,128,256,512]
    hyper_parameters["LR"] = 2e-4
    hyper_parameters["B1"] = 0.5
    hyper_parameters["B2"] = 0.999
    hyper_parameters["L1_LAMBDA"] = 100
    hyper_parameters["NUM_EPOCHS"] = 500
    hyper_parameters["output_folder"] = "test_images/"
    hyper_parameters["loss_function"] = "BCE" # MSE

    output = train_apply(**hyper_parameters)


grid = [ {  
"method" :  ["pix2pix"],
"dataset_name" :  ["sat2map"],
"train_batch_size" :  [8,16,32],
"in_channels" :  [3],
"gen_kernel_size" : [4],# [2,3,4],
"gen_stride" :  [2], #[1,2],
"disc_kernel_size" : [4], # [2,3,4],
"disc_stride" :  [2], # [1,2],
"gen_features" :  [64], # [32,64,128],
"disc_features" :  [[64,128,256,512]], #[[32,64,128,256,512],[64,128,256,512],[128,256,512]],
"LR" :  [2e-4],# [2e-3,2e-4,2e-5],
"B1" :  [0.5],#,0.8],
"B2" :  [0,999],#,0.99,0.9],
"L1_LAMBDA" :  [100],
"NUM_EPOCHS" :  [100,250,500,1000],
"loss_function" :  ["BCE", "MSE"]
}]


all_results_folder = "all_results_folder/"
if not os.path.exists(all_results_folder):
    os.makedirs(all_results_folder)


experiment_id = 0 
for hyper_parameters in ParameterGrid(grid):
    output_folder = all_results_folder + str(experiment_id) +"/"
    hyper_parameters["output_folder"] = output_folder + "images/"
    print(hyper_parameters)

    test_results = train_apply(**hyper_parameters)
    print(test_results)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_folder+"results.json", "w") as f:
        json.dump(test_results, f)
    
    with open(output_folder+"hyper_parameters.json", "w") as f:
        json.dump(hyper_parameters, f)
    
    experiment_id += 1
    break
