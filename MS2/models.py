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
        # default normalize=False thank god I read documentation now we dont have to change anything else
        #didnt work but checked source code and you can *255.byte()
        self.fid = FrechetInceptionDistance(feature=64, normalize=False) # TODO how many features ? Can be one of the following: 64, 192, 768, 2048

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


        if not os.path.exists("evaluation_"+self.loss_function_name):
            os.makedirs("evaluation_"+self.loss_function_name)

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

        if not os.path.exists("test_images"):
            os.makedirs("test_images")
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
