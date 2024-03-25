import os
import torch
import numpy as np
import json
import string
from typing import Counter

def write_image_names(data, filename):
    with open(filename, "w") as file:
        for line in data:
            image_name = line.split(",")[0]
            file.write(image_name + "\n")

def caption_preprocessing(caption):
    punctuation = str.maketrans("", "", string.punctuation)
    caption = caption.split()
    caption = [word.lower() for word in caption]
    caption = [word.translate(punctuation) for word in caption]
    caption = [word for word in caption if len(word) > 1]
    caption = [word for word in caption if word.isalpha()]

    return " ".join(caption)

def save_captions(image2caption, image_subset, filepath):
    captions = []
    for image in image_subset:
        image_id = os.path.splitext(image)[0]
        if image_id in image2caption:
            for caption in image2caption[image_id]:
                captions.append(f"{image} {caption}\n")

    with open(filepath, "w") as f:
        f.writelines(captions)

def split_dataset(image2caption, split_filepaths, save_filepaths):
    for split_filepath, save_filepath in zip(split_filepaths, save_filepaths):
        with open(split_filepath, "r") as f:
            subset_imgs = [filename.replace("\n", "") for filename in f.readlines()]
        save_captions(image2caption, subset_imgs, save_filepath)

def extract_GloVE_embeddings(config, vocabulary):
    np.random.seed(config["seed"])
    embedding_configuration = config["embeddings"]
    embedding_filepath = embedding_configuration["path"]
    embedding_dimension = embedding_configuration["size"]

    punctuation = str.maketrans("", "", string.punctuation)

    vectors = []
    tokens = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    ct = len(tokens)

    embedding_filename = "glove.6B.{}d.txt".format(embedding_dimension)
    embedding_path = os.path.join(config["glove_folder"], embedding_filename)
    with open(embedding_path, "rb") as f:
        for line in f:
            line = line.decode().split()
            word = line[0]
            word = word.strip().lower()
            word = word.translate(punctuation)
            if (word in vocabulary) and (word not in tokens):
                embedding_vector = np.array(line[1:], dtype="float")
                vectors += [embedding_vector]
                tokens[word] = ct
                ct += 1

    with open(config["word2vec_filepath"], "w", encoding="utf8") as f:
        json.dump(tokens, f)

    vectors = np.array(vectors)
    pad_embedding = np.zeros((embedding_dimension,))
    start_embedding = np.random.normal(size=(embedding_dimension,))
    end_embedding = np.random.normal(size=(embedding_dimension,))
    unk_embedding =  np.random.normal(size=(embedding_dimension,))

    assert not np.allclose(start_embedding, end_embedding), "Start and end embeddings are too close."

    for embedding_vectors in vectors:
        assert not np.allclose(start_embedding, embedding_vectors), "Start and other embeddings are too close."
        assert not np.allclose(end_embedding, embedding_vectors), "End and other embeddings are too close."

    vectors = np.vstack([pad_embedding, start_embedding, end_embedding, unk_embedding, vectors])
    np.savetxt(embedding_filepath, vectors)

    print("\nGloVe embeddings extracted.") 
    print("Size of embedding vectors:", embedding_dimension)
    print("Size of vocabulary:", len(tokens))

def create_vocabulary(image2caption):
    tokens = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    words = set()

    for captions in image2caption.values():
        current_words = [word for caption in captions for word in caption.split()]
        words.update(current_words)

    starting_len = len(tokens)
    words = list(words)
    tokens.update({word: (idx + starting_len) for idx, word in enumerate(words)})

    return tokens

def caption_cleaning(image2caption):
    image2caption_clean = image2caption.copy()
    for image, captions in image2caption.items():
        for i in range(len(captions)):
            caption = captions[i]
            clean_caption = caption_preprocessing(caption)
            image2caption_clean[image][i] =  clean_caption

    return image2caption_clean

def load_captions(data):
    image2caption = dict()
    for sample in data.split("\n"):
        if len(sample) < 2:
            continue

        tokens = sample.split()
        image, caption = tokens[0], tokens[1:]

        image_id = image.split(".")[0]
        caption = " ".join(caption)
        
        if image_id not in image2caption:
            image2caption[image_id] = list()
        image2caption[image_id].append(caption)

    return image2caption

def create_traingular_mask(sequence_length, device):
    temp_tensor = torch.ones(sequence_length, sequence_length)
    triangle = torch.triu(temp_tensor)
    mask = triangle.transpose(0, 1)
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.to(device)
    mask.requires_grad = False

    return mask

def store_log_gradients(model, writer, iteration, mode, norm=2):
    norm_value = 0

    for param in model.parameters():
        if param.requires_grad:
            param_norm = param.grad.data.norm(norm)
            norm_value += param_norm.item() ** 2

    norm_value = norm_value ** (0.5)
    writer.add_scalar(f"Gradient/{mode}", norm_value, iteration)    #Check

def add_checkpoint(model, optimizer, start, epoch):
    save_folder = os.path.join("Checkpoint", str(start))
    os.makedirs(save_folder, exist_ok=True)
    model_filename = os.path.join(save_folder, f"model_{epoch}.pth")
    optimizer_filename = os.path.join(save_folder, f"optimizer_{epoch}.pth")
    torch.save(model.state_dict(), model_filename)
    torch.save(optimizer.state_dict(), optimizer_filename)
    print(f"Checkpoint added at epoch={epoch}")

def generate_captions(model, image_features, start_id, end_id, pad_id, vec2word, caption_max_length, device):
    batch_size = image_features.size(0)

    x_words = torch.Tensor([start_id] + [pad_id] * (caption_max_length - 1))
    x_words = x_words.to(device)
    x_words = x_words.long()
    x_words = x_words.repeat(batch_size, 1)

    padding_mask = torch.Tensor([True] * caption_max_length)
    padding_mask = padding_mask.to(device)
    padding_mask = padding_mask.bool()
    padding_mask = padding_mask.repeat(batch_size, 1)

    is_decoded = [False] * batch_size

    generated_captions = []
    for _ in range(batch_size):
        generated_captions.append([])

    for i in range(caption_max_length - 1):
        padding_mask[:, i] = False

        y_predicted_probabilities = model(x_words, image_features, padding_mask)
        y_predicted_probabilities = y_predicted_probabilities[torch.arange(batch_size), [i] * batch_size].clone()
        y_predicted = y_predicted_probabilities.argmax(-1)
        
        for batch_index in range(batch_size):
            if is_decoded[batch_index]:
                continue
            generated_captions[batch_index].append(vec2word[str(y_predicted[batch_index].item())])
            if y_predicted[batch_index] == end_id:
                is_decoded[batch_index] = True
            
        if np.all(is_decoded):
            break

        if i < (caption_max_length - 1):
            x_words[torch.arange(batch_size), [i+1] * batch_size] = y_predicted.view(-1)

    for batch_index in range(batch_size):
        if not is_decoded[batch_index]:
            generated_captions[batch_index].append(vec2word[str(end_id)])

    for caption in generated_captions:
        caption.remove("<END>")

    return generated_captions