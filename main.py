import json
import torch
# from tensorboardX import SummaryWriter
from model_train import train
import warnings

warnings.filterwarnings("ignore")

def main():
    config_path = "config.json"
    with open(config_path, "r", encoding="utf8") as f:
        config = json.load(f)

    # writer = SummaryWriter()
    writer = None
    gpu = config["gpu"] and torch.cuda.is_available()
    device = torch.device("cuda" if gpu else "cpu")

    train(config, writer, device)

if __name__ == "__main__":
    main()