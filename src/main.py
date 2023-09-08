
import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

import preprocessing as pre
import models.LightningDiffusion as ld
import models.SimpleDiffusion as sd


def parse_args():
    parser = argparse.ArgumentParser(description="Daifuku Help")
    parser.add_argument('-m', '--mode', type=str, default="preprocess,train", help="Daifuku Pipeline Mode (options are preprocess,train)")
    parser.add_argument('-s', '--studyname', type=str, default="default", help="Name to be used for saved dataloaders, models, and visualizations (for organization)")
    parser.add_argument('-c', '--config', type=str, default="./config.json", help="Path to configuration file")
    parser.add_argument('-t', '--train_config', type=str, default="./config.json", help="Path to training configuration file")
    parser.add_argument('--checkpoint', type=str, default="None", help="Path to model checkpoint")
    parser.add_argument('--debug', action="store_true", help="For debugging purposes only")
    return parser.parse_args()


def main():
    args = parse_args()
    config = pre.get_config(args.config)
    train_config = pre.get_config(args.train_config)
    config["train_config"] = train_config
    
    preprocess_output_path = os.path.join(config["output_dir"], "preprocessed")
    training_output_path = os.path.join(config["output_dir"], "trained")
    
    if "preprocess" in args.mode:
        print("\n#------------------------------ Preprocessing ----------------------------#\n")
        
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)
            
        if not args.debug:
            pre.higashi_preprocess(config)
        
        train_loader, test_loader = pre.construct_dataloaders(config=config)
        
        if train_config["loading"] != "all_at_once":
            torch.save(train_loader, os.path.join(preprocess_output_path,f'train_dataloader_{args.studyname}.pth'))
            torch.save(test_loader, os.path.join(preprocess_output_path,f'test_dataloader_{args.studyname}.pth'))
        
        print("\n#------------------------------ Done ----------------------------#\n")

    if "train" in args.mode:
        print("\n#-------------------------------- Training -------------------------------#\n")
        
        pl.seed_everything(1)

        if not "preprocess" in args.mode:
            train_loader = torch.load(os.path.join(preprocess_output_path,f'train_dataloader_{args.studyname}.pth'))
            test_loader = torch.load(os.path.join(preprocess_output_path,f'test_dataloader_{args.studyname}.pth'))
            

        daifuku = ld.LightningDiffusion(train_config=train_config)
        
        
        wandb_logger = WandbLogger(project="Daifuku")
        trainer = Trainer(
                    devices=train_config["num_devices"],
                    logger=wandb_logger,
                    max_epochs=train_config["epochs"],
                    accelerator=train_config["accelerator"],
                    # profile=args.debug
                )
        
        if os.path.exists(args.checkpoint):
            print("Loading from checkpoint ...")
            trainer.fit(daifuku, train_loader, test_loader, ckpt_path=args.checkpoint)
        else:
            trainer.fit(daifuku, train_loader, test_loader)

        trainer.save_checkpoint(os.path.join(training_output_path,f'daifuku_{args.studyname}_{train_config["epochs"]}epochs.ckpt'))
        print("\n#------------------------------ Done ----------------------------#\n")



if __name__ == "__main__":
    main()