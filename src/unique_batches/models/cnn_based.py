# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from dataclasses import dataclass

import torch
import torch.nn as nn

from unique_batches.models.factory import register
from unique_batches.models.layers import TimeDistributed
from unique_batches.models.model import Model, ModelConfig


@dataclass
class CNNBasedConfig(ModelConfig):
    vocab_size: int = None
    embed_dim: int = None
    first_conv_dim: int = None
    second_conv_dim: int = None

    def __init__(self, **kwargs):
        super(CNNBasedConfig, self).__init__(**kwargs)


@register("cnn_based")
class CNNBased(Model):
    def __init__(self, model_config: CNNBasedConfig, **kwargs):

        super().__init__(model_config)

        self.embed_dim = model_config.embed_dim
        self.hidden_size = model_config.hidden_size

        self.embedding = nn.Embedding(model_config.vocab_size, model_config.embed_dim)

        # first convolutional layer
        self.cnn1 = TimeDistributed(
            nn.Conv1d(
                model_config.embed_dim, model_config.first_conv_dim, kernel_size=(3,)
            )
        )

        self.max_pool = TimeDistributed(nn.AdaptiveMaxPool1d(1))

        # second set of convolutions
        self.cnn2_3 = TimeDistributed(
            nn.Conv1d(1, model_config.second_conv_dim, kernel_size=(3,))
        )
        self.cnn2_4 = TimeDistributed(
            nn.Conv1d(1, model_config.second_conv_dim, kernel_size=(4,))
        )
        self.cnn2_5 = TimeDistributed(
            nn.Conv1d(1, model_config.second_conv_dim, kernel_size=(5,))
        )

        self.bilstm = nn.LSTM(
            bidirectional=True,
            dropout=0.5,
            input_size=model_config.second_conv_dim * 3,
            hidden_size=512,
            batch_first=True,
        )

        self.fc = TimeDistributed(
            nn.Linear(
                in_features=self.hidden_size * 2, out_features=model_config.num_tags
            )
        )

    def forward(self, utterances: torch.tensor, **kwargs) -> torch.tensor:
        """
        Args:
            encoded_utterances: input tensor (batch_size, sentence_len, word_len)
        Returns:
            output: tensor of logits (batch_size, sentence_len, num_classes)
        """

        # shape (batch_size, sentence_len, word_len, emb_dim)
        emb_char = self.embedding(utterances)

        # invert channel dim with sequence len, as convolution expects something of shape (N, C, L_in)
        # (batch_size, utt_len, word_len, emb_dim) -->  (batch_size, utt_len, emb_dim, word_len)
        emb_char = emb_char.transpose(3, 2)

        # 1st CNN
        relu = nn.ReLU()
        conv1_output = self.cnn1(relu(emb_char))

        conv1_output = conv1_output.transpose(3, 2)
        max1 = self.max_pool(conv1_output).transpose(2, 3)

        # 2nd CNN
        cnn2_3_output = self.cnn2_3(relu(max1))
        cnn2_4_output = self.cnn2_4(relu(max1))
        cnn2_5_output = self.cnn2_5(relu(max1))

        # max pooling
        max2_3 = self.max_pool(cnn2_3_output)
        max2_4 = self.max_pool(cnn2_4_output)
        max2_5 = self.max_pool(cnn2_5_output)

        max_all = torch.cat((max2_3, max2_4, max2_5), dim=2).squeeze()

        lstm_output, (_, _) = self.bilstm(max_all)

        output = self.fc(lstm_output).squeeze()

        return output
