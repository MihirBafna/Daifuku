'''
---------------------------------------------------------------------------------------------------------------------------------
                                    Pytorch Lightning wrapper class for diffusion model
---------------------------------------------------------------------------------------------------------------------------------
'''

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb
from .Diffusion import Unet, Unet_conditional, GaussianDiffusion


class LightningDiffusion(pl.LightningModule):
    
    def __init__(self, train_config):
        super().__init__()
        
        self.train_config = train_config
        self.conditional = True if train_config["conditional"] == "true" else False     # whether or not Unet model should be conditional
                
        if self.conditional:
            print("Conditioning has been set to true. Using conditional Unet as score model.")
            self.score_model = Unet_conditional(
                dim = train_config["hidden_dim"],
                dim_mults=(1, 2, 4),
                channels=1,
            )
        else:
            print("Conditioning has been set to false. Using regular Unet as score model.")
            self.score_model = Unet(
                dim = train_config["hidden_dim"],
                dim_mults=(1, 2, 4, 8),
                channels=1,
            )
        
        self.diffusion_model = GaussianDiffusion(
            self.score_model,
            auto_normalize = True,
            objective=train_config["prediction_type"],
            image_size = train_config["map_size"],          # 200
            timesteps = train_config["timesteps"],           # number of steps 1000
            sampling_timesteps = train_config["sampling_timesteps"],    # 250 number of sampling timesteps (using ddim for faster inference [see citation for ddim paper]),
            conditional = self.conditional
            
        )
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_config["learning_rate"])
        return optimizer
        
    def forward(self, x, condition = None):
        return self.diffusion_model(x, condition=condition)
        
    def sample(self, batch_size, input_image=None, return_all_timesteps=False, condition=None):
        return self.diffusion_model.sample(batch_size=batch_size, input_image=input_image, return_all_timesteps=return_all_timesteps, condition=condition)
    
    def training_step(self, train_batch, batch_idx):
        if self.conditional:
            # define condition
            bulk,sc = train_batch.chunk(2, dim=1)
            out, loss = self.diffusion_model(bulk, condition = sc)
        else:
            out, loss = self.diffusion_model(train_batch.unsqueeze(1), condition = None)
            
            
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
        if self.conditional:
            bulk,sc = val_batch.chunk(2, dim=1)
            _, loss = self.diffusion_model(bulk, condition = sc)
        else:
            _, loss = self.diffusion_model(val_batch.unsqueeze(1))
            
        self.log("val_loss",loss.detach(), on_step=True, on_epoch=True)
        
        
# class LogSampleContactMaps(Callback):
    
#      def on_validation_batch_end(self, trainer, daifuku_model, outputs, batch, batch_idx, dataloader_idx):
#         """Called when the validation batch ends."""
#         img, loss = daifuku_model(batch.unsqueeze(1)[0])
#         
#         trainer.log()

 
