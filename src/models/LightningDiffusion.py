'''
---------------------------------------------------------------------------------------------------------------------------------
                                    Pytorch Lightning wrapper class for diffusion model
---------------------------------------------------------------------------------------------------------------------------------
'''

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb
from .Diffusion import Unet, GaussianDiffusion


class LightningDiffusion(pl.LightningModule):
    
    def __init__(self, train_config):
        super().__init__()
        
        self.train_config = train_config
        
        self.score_model = Unet(
            dim = train_config["map_size"],
            dim_mults=(1, 2, 4, 8),
            channels=1
        )
        
        self.diffusion_model = GaussianDiffusion(
            self.score_model,
            objective=train_config["prediction_type"],
            image_size = train_config["map_size"],          # 200
            timesteps = train_config["timesteps"],           # number of steps 1000
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
        out, loss = self.diffusion_model(train_batch.unsqueeze(1))
        
        if batch_idx == 0:
            og = wandb.Image(train_batch[0].squeeze().cpu(), caption="original contact map")
            recon = wandb.Image(out[0].squeeze().cpu(), caption="reconstructed contact map")
            self.logger.experiment.log({"train_original_contact_map":og,"train_reconstructed_contact_map":recon})
            
        self.log("train_loss",loss.detach(), on_step=True, on_epoch=True)
        return loss
    
    # def on_train_epoch_end(self):
    #     img = wandb.Image(
    #         self.training_reconstructed_image , 
    #         caption="Training reconstructed image"
    #     )
    #     self.log_image(images=img)
    #     self.training_step_outputs.clear()  # free memory
    
    def validation_step(self, val_batch, batch_idx):
        _, loss = self.diffusion_model(val_batch.unsqueeze(1))
        self.log("val_loss",loss.detach(), on_step=True, on_epoch=True)
        
        
# class LogSampleContactMaps(Callback):
    
#      def on_validation_batch_end(self, trainer, daifuku_model, outputs, batch, batch_idx, dataloader_idx):
#         """Called when the validation batch ends."""
#         img, loss = daifuku_model(batch.unsqueeze(1)[0])
#         
#         trainer.log()

 
