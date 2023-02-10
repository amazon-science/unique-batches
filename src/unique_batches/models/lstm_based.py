# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from unique_batches.models.factory import register
from unique_batches.models.model import Model, ModelConfig


@dataclass
class LSTMbasedConfig(ModelConfig):
    vocab_size: int = -1
    dropout_chance: float = 0.2
    embed_dim: int = 64
    hidden_size: int = 512
    num_layers: int = 2

    def __init__(self, **kwargs):
        super(LSTMbasedConfig, self).__init__(**kwargs)


@register("lstm_based")
class LSTMBased(Model):
    def __init__(self, model_config: LSTMbasedConfig, **kwargs):

        super().__init__(model_config)

        self.embedding = nn.Embedding(model_config.vocab_size, model_config.embed_dim)

        self.bilstm = nn.LSTM(
            bidirectional=True,
            dropout=model_config.dropout_chance,
            input_size=model_config.embed_dim,
            hidden_size=model_config.hidden_size,
            batch_first=True,
            num_layers=model_config.num_layers,
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=2 * self.hidden_size, out_features=self.hidden_size),
            nn.Dropout(model_config.dropout_chance),
            nn.ReLU(),
        )

        self.fc_2 = nn.Linear(
            in_features=self.hidden_size, out_features=model_config.num_tags
        )

        self.save_hyperparameters(logger=False)

    def forward(self, utterances, **kwargs) -> torch.tensor:

        # (batch_size, max_utt_len, emb_dim)
        embedded_utterances = self.embedding(utterances)

        # (batch_size, max_utt_len, 2 * hidden_size)
        lstm_output, (_, _) = self.bilstm(embedded_utterances)

        # (batch_size, max_utt_len, hidden_size)
        hiddens = self.fc(lstm_output).squeeze()

        # (batch_size, max_utt_len, num_labels)
        logits = self.fc_2(hiddens).squeeze()

        return logits
