
import argparse
import preprocessing as pre


def parse_args():
    parser = argparse.ArgumentParser(description="Daifuku Help")
    parser.add_argument('-m', '--mode', type=str, default="preprocess,train", help="Daifuku Pipeline Mode (options are preprocess,train)")
    parser.add_argument('-s', '--studyname', type=str, default="default", help="Name to be used for saved dataloaders, models, and visualizations (for organization)")
    parser.add_argument('-c', '--config', type=str, default="./config.json")
    return parser.parse_args()


def main():
    args = parse_args()
    config = pre.get_config(args.config)
    if "preprocess" in args.mode:
        print("\n#------------------------------ Preprocessing ----------------------------#\n")
        
        pre.higashi_preprocess(config)

    if "train" in args.mode:
        print("\n#-------------------------------- Training -------------------------------#\n")
        
if __name__ == "__main__":
    main()