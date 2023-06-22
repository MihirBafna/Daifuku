
import argparse
import os
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

import preprocessing as pre
import models.LightningDiffusion as ld

def parse_args():
    parser = argparse.ArgumentParser(description="Daifuku Help")
    parser.add_argument('-m', '--mode', type=str, default="preprocess,train", help="Daifuku Pipeline Mode (options are preprocess,train)")
    parser.add_argument('-s', '--studyname', type=str, default="default", help="Name to be used for saved dataloaders, models, and visualizations (for organization)")
    parser.add_argument('-c', '--config', type=str, default="./config.json", help="Path to configuration file")
    parser.add_argument('-t', '--train_config', type=str, default="./config.json", help="Path to training configuration file")
    parser.add_argument('--debug', action="store_true", help="For debugging purposes only")
    return parser.parse_args()


def main():
    args = parse_args()
    config = pre.get_config(args.config)
    train_config = pre.get_config(args.train_config)

    
    preprocess_output_path = os.path.join(config["output_dir"], "preprocessed")
    training_output_path = os.path.join(config["output_dir"], "train")
    
    if "preprocess" in args.mode:
        print("\n#------------------------------ Preprocessing ----------------------------#\n")
        
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)
            
        if not args.debug:
            pre.higashi_preprocess(config)
        
        train_loader, test_loader = pre.construct_dataloaders(config=config, batch_size=64, train_size=0.8, type="bulk")
        
        torch.save(train_loader, os.path.join(preprocess_output_path,f'train_dataloader_{args.studyname}.pth'))
        torch.save(test_loader, os.path.join(preprocess_output_path,f'test_dataloader_{args.studyname}.pth'))
        
        
    if "train" in args.mode:
        print("\n#-------------------------------- Training -------------------------------#\n")
        
        if not "preprocess" in args.mode:
            train_loader = torch.load(os.path.join(preprocess_output_path,f'train_dataloader_{args.studyname}.pth'))
            test_loader = torch.load(os.path.join(preprocess_output_path,f'test_dataloader_{args.studyname}.pth'))
            
        daifuku = ld.LightningDiffusion(train_config=train_config)
        
        wandb_logger = WandbLogger()
        trainer = Trainer(
                    logger=wandb_logger,
                    max_epochs=train_config["epochs"],
                )
        
        daifuku.fit()
        


if __name__ == "__main__":
    main()