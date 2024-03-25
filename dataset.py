import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict

class Flickr8K(Dataset):
    def __init__(self, config, filepath, training=True):
        with open(filepath, "r") as f:
            self.data = [line.replace("\n", "") for line in f.readlines()]

        with open(config["word2vec_filepath"], "r", encoding="utf8") as f:
            self.word2vec = json.load(f)
        self.vec2word = {str(idx): word for word, idx in self.word2vec.items()}

        self.training = training
        self.inference_captions = self.group_captions(self.data)
        self.pad_index = config["PAD_index"]
        self.start_index = config["START_index"]
        self.end_index = config["END_index"]
        self.unk_index = config["UNK_index"]
        self.PAD_token = config["PAD_token"]
        self.START_token = config["START_token"]
        self.END_token = config["END_token"]
        self.UNK_token = config["UNK_token"]
        self.caption_max_length = config["caption_max_length"]
        self.image_properties = config["image_properties"]
        self.image_transform = self.transform_images(self.image_properties["image_size"])
        self.image_folder = self.image_properties["image_folder"]
        self.data = self.image_caption_mapping(self.data)
        self.dataset_size = len(self.data) if self.training else 0

    def transform_images(self, image_size):
        preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return preprocessing

    def load_and_process_images(self, image_folder, image_names):
        image_paths = [os.path.join(image_folder, filename) for filename in image_names]
        images = [Image.open(path) for path in image_paths]
        image_tensors = [self.image_transform(image) for image in images]
        images_processed = {image_name: image_tensor for image_name, image_tensor in zip(image_names, image_tensors)}

        return images_processed

    def group_captions(self, data):
        grouped_captions = defaultdict(list)

        for line in data:
            caption_data = line.split()
            image_name = caption_data[0].split("#")[0]
            image_caption = caption_data[1:]
            grouped_captions[image_name].append(image_caption)

        return grouped_captions

    def image_caption_mapping(self, data):
        mapping = []
        for line in data:
            tokens = line.split()
            image_name = tokens[0].split("#")[0]
            caption_words = tokens[1:]
            mapping.append((image_name, caption_words))

        return mapping

    def preprocess_image(self, image_name):
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        return image_tensor

    def inference_batch(self, batch_size):
        inference_caption_items = list(self.inference_captions.items())
        # random.shuffle(inference_caption_items)
        num_batches = len(inference_caption_items) // batch_size

        for idx in range(num_batches):
            caption_samples = inference_caption_items[idx * batch_size: (idx + 1) * batch_size]
            idx += batch_size
            batch_images = []
            batch_captions = []
            
            for image_name, captions in caption_samples:
                batch_captions.append(captions)
                batch_images.append(self.preprocess_image(image_name))

            batch_images = torch.stack(batch_images, dim=0)
            if batch_size == 1:
                batch_images = batch_images.unsqueeze(0)

            yield batch_images, batch_captions

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        image, tokens = self.data[index]
        image_tensor = self.preprocess_image(image)
        tokens = tokens[:self.caption_max_length]
        tokens = [token.strip().lower() for token in tokens]
        tokens = [self.START_token] + tokens + [self.END_token]
        input_tokens = tokens[:-1].copy()
        target_tokens = tokens[1:].copy()
        sample_size = len(input_tokens)
        padding_size = self.caption_max_length - sample_size

        if padding_size > 0:
            padding_vector = [self.PAD_token]*padding_size
            input_tokens += padding_vector.copy()
            target_tokens += padding_vector.copy()

        input_tokens = [self.word2vec.get(token, self.unk_index) for token in input_tokens]
        target_tokens = [self.word2vec.get(token, self.unk_index) for token in target_tokens]
        input_tokens = torch.Tensor(input_tokens).long()
        target_tokens = torch.Tensor(target_tokens).long()

        target_padding_mask = torch.ones([self.caption_max_length, ])
        target_padding_mask[:sample_size] = 0.0
        target_padding_mask = target_padding_mask.bool()

        return image_tensor, input_tokens, target_tokens, target_padding_mask