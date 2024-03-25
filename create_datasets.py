import json
from utils import *
import random

if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_path = config["dataset_split_filepath"]
    with open(dataset_path, "r") as f:
        data = f.read()

    image2caption = load_captions(data)
    image2caption = caption_cleaning(image2caption)
    vocab = create_vocabulary(image2caption)
    extract_GloVE_embeddings(config, vocab)

    random.seed(config["seed"])
    with open(dataset_path, "r") as f:
        data = f.readlines()
    random.shuffle(data)
    train_data = data[:6000]
    val_data = data[6000:7000]
    test_data = data[7000:]
    write_image_names(train_data, config["split_images_filepath"]["train"])
    write_image_names(val_data, config["split_images_filepath"]["validation"])
    write_image_names(test_data, config["split_images_filepath"]["test"])

    split_images_paths = list(config["split_images_filepath"].values())
    split_save_paths = list(config["split_labels_filepath"].values())
    split_dataset(image2caption, split_images_paths, split_save_paths)