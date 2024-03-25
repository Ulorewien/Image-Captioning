import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from nltk.translate.bleu_score import corpus_bleu

from dataset import Flickr8K
from transformer import CaptionDecoder
from utils import *

def evaluate(subset, encoder, decoder, config, device):
    batch_size = config["batch_size"]
    caption_max_length = config["caption_max_length"]
    bleu_w = config["bleu_weights"]
    vec2word = subset.vec2word
    pad_id = subset.pad_index
    start_id = subset.start_index
    end_id = subset.end_index
    references_total = []
    predictions_total = []

    print("Evaluating the model...")
    for x_image, y_caption in subset.inference_batch(batch_size):
        x_image = x_image.to(device)

        image_features = encoder(x_image)
        image_features = image_features.view(image_features.size(0), image_features.size(1), -1)
        image_features = image_features.permute(0, 2, 1)
        image_features = image_features.detach()

        predictions = generate_captions(decoder, image_features, start_id, end_id, pad_id, vec2word, caption_max_length, device)
        references_total += y_caption
        predictions_total += predictions

    bleu_1 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu_1"]) * 100
    bleu_2 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu_2"]) * 100
    bleu_3 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu_3"]) * 100
    bleu_4 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu_4"]) * 100
    bleu = [bleu_1, bleu_2, bleu_3, bleu_4]

    return bleu

def train(config, writer, device):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    train_hyperparams = {
        "batch_size": config["batch_size"],
        "shuffle": True,
        "num_workers": 1,
        "drop_last": True
    }

    train_set = Flickr8K(config, config["split_labels_filepath"]["train"], training=True)
    valid_set = Flickr8K(config, config["split_labels_filepath"]["validation"], training=False)
    train_loader = DataLoader(train_set, **train_hyperparams)

    encoder = models.resnet50(pretrained=True)
    encoder = torch.nn.Sequential(*(list(encoder.children())[:-2]))
    encoder = encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    decoder = CaptionDecoder(config)
    decoder = decoder.to(device)
    if config["checkpoint"]["load"]:
        checkpoint_path = config["checkpoint"]["path"]
        decoder.load_state_dict(torch.load(checkpoint_path))
    decoder.train()

    causal_mask = create_traingular_mask(config["caption_max_length"], device)

    training_configuration = config["training_configuration"]
    learning_rate = training_configuration["learning_rate"]
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=training_configuration["learning_rate"],
        weight_decay=training_configuration["l2_penalty"]
    )
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_time = time.strftime("%b-%d_%H-%M-%S")
    train_step = 0
    for epoch in range(training_configuration["num_of_epochs"]):
        print("Epoch:", epoch)
        decoder.train()

        for x_image, x_words, y, target_padding_mask in train_loader:
            optimizer.zero_grad()
            train_step += 1

            x_image, x_words = x_image.to(device), x_words.to(device)
            y = y.to(device)
            target_padding_mask = target_padding_mask.to(device)

            with torch.no_grad():
                image_features = encoder(x_image)
                image_features = image_features.view(image_features.size(0), image_features.size(1), -1)
                image_features = image_features.permute(0, 2, 1)
                image_features = image_features.detach()

            y_pred = decoder(x_words, image_features, target_padding_mask, causal_mask)
            target_padding_mask = torch.logical_not(target_padding_mask)
            y_pred = y_pred[target_padding_mask]
            y = y[target_padding_mask]

            loss = loss_function(y_pred, y.long())
            loss.backward()
            # store_log_gradients(decoder, writer, train_step, "Before")
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), training_configuration["gradient_clipping"])
            # store_log_gradients(decoder, writer, train_step, "After")
            optimizer.step()
            # writer.add_scalar("Train/Step-Loss", loss.item(), train_step)
            # writer.add_scalar("Train/Learning-Rate", learning_rate, train_step)

        add_checkpoint(decoder, optimizer, start_time, epoch)
        if (epoch + 1) % training_configuration["eval_period"] == 0:
            with torch.no_grad():
                encoder.eval()
                decoder.eval()

                train_bleu = evaluate(train_set, encoder, decoder, config, device)
                valid_bleu = evaluate(valid_set, encoder, decoder, config, device)
                for i, t_b in enumerate(train_bleu):
                    print(f"Train/BLEU-{i+1}", t_b, epoch)
                #     writer.add_scalar(f"Train/BLEU-{i+1}", t_b, epoch)
                for i, v_b in enumerate(valid_bleu):
                    print(f"Valid/BLEU-{i+1}", v_b, epoch)
                #     writer.add_scalar(f"Valid/BLEU-{i+1}", v_b, epoch)

                decoder.train()