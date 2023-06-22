'''
---------------------------------------------------------------------------------------------------------------------------------
                                    Pytorch Lightning wrapper class for diffusion model
---------------------------------------------------------------------------------------------------------------------------------
'''

import torch
import pytorch_lightning as pl
from Diffusion import Unet, GaussianDiffusion


class LightningDiffusion(pl.LightningModule):
    
    def __init__(self, train_config):
        super().__init__()
        
        self.train_config = train_config
        
        self.score_model = Unet(
            dim = train_config["hidden_dim"],
            dim_mults = (1, 2, 4, 8),
            flash_attn = True,
            channels=1,
        )
        
        self.diffusion_model = GaussianDiffusion(
            self.score_model,
            image_size = train_config["map_size"],          # 196
            timesteps = train_config["epochs"],           # number of steps 1000
            sampling_timesteps = train_config["sampling_timesteps"]    # 250 number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        )
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_config["learning_rate"])
        return optimizer
        
    def forward(self, x):
        return self.diffusion_model(x)
        
    def sample(self, batch_size):
        return self.diffusion_model.sample(batch_size=batch_size)
    
    def training_step(self, train_batch, batch_idx):
        _, loss = self.diffusion_model(train_batch)
        self.log("train_loss",loss, on_step=False, on_epoch=True)
        return loss
    
    def training_step(self, train_batch, batch_idx):
        _, loss = self.diffusion_model(train_batch)
        self.log("train_loss",loss, on_step=False, on_epoch=True)
        return loss
    
    def val_step(self, val_batch, batch_idx):
        _, loss = self.diffusion_model(val_batch)
        self.log("val_loss",loss, on_step=False, on_epoch=True)
        
        