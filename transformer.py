import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.residual_block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        return x + self.residual_block(x)

class Normalization(nn.Module):
    def __init__(self, eps=1e-5):
        super(Normalization, self).__init__()
        self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, dim=-1):
        norm = x.norm(2, dim=dim).unsqueeze(-1)
        x = self.eps * (x / norm)
        return x

class PositionalEncodings(nn.Module):
    def __init__(self, sequence_length, decoder_model, dropout_value):
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=sequence_length).view(-1, 1)
        dimension_positions = torch.arange(start=0, end=decoder_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dimension_positions) / decoder_model))
        encodings = torch.zeros(1, sequence_length, decoder_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = x + self.positional_encodings
        x = self.dropout(x)
        return x

class CaptionDecoder(nn.Module):
    def __init__(self, config):
        super(CaptionDecoder, self).__init__()
        model_configuration = config["model_configuration"]
        decoder_layers = model_configuration["decoder_layers"]
        decoder_model = model_configuration["decoder_model"]
        feed_forward_dimension = model_configuration["feed_forward_dimension"]
        attention_heads = model_configuration["attention_heads"]
        dropout = model_configuration["dropout_value"]
        embedding_dimension = config["embeddings"]["size"]
        vocabulary_size = config["vocabulary"]
        image_feature_channels = config["image_properties"]["image_feature_channels"]

        word_embeddings = torch.Tensor(np.loadtxt(config["embeddings"]["path"]))
        self.embedding_layer = nn.Embedding.from_pretrained(
            word_embeddings,
            freeze=True,
            padding_idx=config["PAD_index"]
        )

        self.entry_mapping_words = nn.Linear(embedding_dimension, decoder_model)
        self.entry_mapping_image = nn.Linear(image_feature_channels, decoder_model)

        self.residual_block = ResidualBlock(decoder_model)
        self.positional_encodings = PositionalEncodings(config["caption_max_length"], decoder_model, dropout)
        transformer_decoder_layer = TransformerDecoderLayer(
            d_model=decoder_model,
            nhead=attention_heads,
            dim_feedforward=feed_forward_dimension,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(transformer_decoder_layer, decoder_layers)
        self.classifier = nn.Linear(decoder_model, vocabulary_size)

    def forward(self, x, image_features, target_padding_mask=None, target_mask=None):
        image_features = self.entry_mapping_image(image_features)
        image_features = image_features.permute(1, 0, 2)
        image_features = F.leaky_relu(image_features)

        x = self.embedding_layer(x)
        x = self.entry_mapping_words(x)
        x = F.leaky_relu(x)
        x = self.residual_block(x)
        x = F.leaky_relu(x)
        x = self.positional_encodings(x)
        x = x.permute(1, 0, 2)
        x = self.decoder(
            tgt=x,
            memory=image_features,
            tgt_key_padding_mask=target_padding_mask,
            tgt_mask=target_mask
        )
        x = x.permute(1, 0, 2)
        x = self.classifier(x)
        
        return x