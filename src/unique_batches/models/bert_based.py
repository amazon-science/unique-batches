# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_scatter import scatter_mean
from transformers.models.distilbert import DistilBertModel

from unique_batches.models.factory import register
from unique_batches.models.model import Model, ModelConfig
from unique_batches.utils.framework import pad_sequence, scatter_mean


@dataclass
class BertBasedConfig(ModelConfig):
    dropout_chance: float = None
    embedder_checkpoint: str = None
    embedding_mode: str = None
    max_utt_len: int = None
    use_bilstm: bool = False

    def __init__(self, **kwargs):
        super(BertBasedConfig, self).__init__(**kwargs)


@register("bert_based")
class BertBased(Model):
    """
    BERT based model, expects utterances tokenized and encoded from a WordPieceEncoder
    each word is embedded by Bert, fed to a bi-LSTM and finally to a fully connected layer to map to the tag space
    """

    def __init__(self, model_config: BertBasedConfig, **kwargs):

        super().__init__(model_config)

        self.save_hyperparameters(logger=False)

        self.embedding_mode = model_config.embedding_mode
        self.embedder_checkpoint = Path(model_config.embedder_checkpoint).resolve()
        self.max_utt_len = model_config.max_utt_len
        self.use_bilstm = model_config.use_bilstm

        self._setup_embedder()
        self._freeze_embedder()

        self.embed_dim = self._get_embedding_dim()
        self.dropout_chance = model_config.dropout_chance

        fc_in_features = self.embed_dim
        if self.use_bilstm:
            self.bilstm = nn.LSTM(
                bidirectional=True,
                # dropout=self.dropout_chance,
                input_size=self.embed_dim,
                hidden_size=self.hidden_size,
                batch_first=True,
            )

            fc_in_features = 2 * self.hidden_size

        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_in_features, out_features=self.hidden_size),
            nn.Dropout(self.dropout_chance),
            nn.ReLU(),
        )

        self.fc_2 = nn.Linear(
            in_features=self.hidden_size, out_features=model_config.num_tags
        )

    def _freeze_embedder(self):
        """
        Freeze the Bert model by disabling the gradient requirement of the parameters
        """
        self.embedder = self.embedder.eval()
        for param in self.embedder.parameters():
            param.requires_grad = False

    def _setup_embedder(self):
        """
        Load the pretrained BERT model from path
        """
        checkpoint = self.embedder_checkpoint
        print(f"Using checkpoint: {checkpoint}")
        self.embedder: DistilBertModel = DistilBertModel.from_pretrained(checkpoint)

    def forward(
        self, utterances: torch.FloatTensor, word_ids: torch.LongTensor, **kwargs
    ) -> torch.tensor:
        """
        Args:
            encoded_utterances: tensor ~ (batch_size, max_bert_len, emb_dim) where max_bert_len is the length
                                 of the longest utterance after the bert tokenization,
                                 i.e. with SEP, CLS and words split in wordpieces
            word_ids:      list of lists ~ (batch_size, max_bert_len), each list contains the mapping from tokens
                                to original words in the utterance, e.g. the utterance
                                 'presumo tu lo sappia' would result in
                                 [ 'pres##', '##umo', 'tu', 'lo', 'sap##', '##ia']
                                and therefore in the following subword mask [0, 0, 1, 2, 3, 3]
        Returns:
            logits: tensor of logits (batch_size, max_utt_len, num_classes)
        """
        assert utterances.shape[0] == word_ids.shape[0]

        # (batch_size, max_utt_len, emb_dim)
        embedded_sentences = self._embed(utterances, word_ids)

        if self.use_bilstm:
            # (batch_size, max_utt_len, 2*hidden_size)
            embedded_sentences, (_, _) = self.bilstm(embedded_sentences)

        # (batch_size, max_utt_len, num_lang_labels)
        hiddens = self.fc(embedded_sentences)

        # (batch_size, max_utt_len, num_tags)
        logits = self.fc_2(hiddens).squeeze()

        return logits

    def _embed(self, utterances: torch.FloatTensor, word_ids: torch.LongTensor):
        """
        Takes utterances encoded by a WordPieceEncoder and feeds them to Bert,
        obtaining an embedding for each utterance; each embedded utterance is (max_bert_len, emb_dim),
        as each subword is embedded separately. The embeddings corresponding to the original words
        are then obtained by aggregating the subword embeddings.
        Args:
            encoded_utterances: tensor ~ (batch_size, max_bert_len) where max_bert_len is the length
                                of the longest utterance after the bert tokenization,
                                i.e. with SEP, CLS and words split in wordpieces
            word_ids:      list of lists ~ (batch_size, max_bert_len), each list contains
                                the mapping from tokens to original words in the utterance,
                                e.g. the utterance 'presumo tu lo sappia' would result in
                                [ 'pres##', '##umo', 'tu', 'lo', 'sap##', '##ia']
                                and therefore in the following subword mask [0, 0, 1, 2, 3, 3]
        Returns:
            embedded_utterances_merged: tensor ~ (batch_size, max_utt_len, emb_dim) of utterances embedded at word level
        """

        # (batch_size, bert_utt_len, emb_dim)
        embedded_utterances = self._get_bert_output(utterances)

        # (batch_size, max_utt_len, emb_dim)
        embedded_utterances = self._merge_subword_embeddings(
            embedded_utterances, word_ids
        )

        return embedded_utterances

    def _get_bert_output(self, utterances: torch.FloatTensor):

        bert_output = self.embedder(utterances, output_hidden_states=True)

        if self.embedding_mode == "last_hidden_state":
            k = 1
        elif self.embedding_mode == "last_2_hidden_states":
            k = 2
        elif self.embedding_mode == "last_3_hidden_states":
            k = 3
        else:
            raise NotImplementedError

        # (batch_size, max_utt_len, num_hidden_states, embedding_dim)
        last_k_hidden_states = torch.stack(bert_output.hidden_states[-k:], dim=2)

        # (batch_size, max_utt_len, embedding_dim)
        embedded_utterances = last_k_hidden_states.sum(dim=2)

        return embedded_utterances

    def _merge_subword_embeddings(
        self, embedded_utterances: torch.FloatTensor, word_ids: torch.LongTensor
    ):
        """
        Each input embedded utterance is (max_bert_len, emb_dim), as each subword is embedded separately,
        so the embeddings for the original words are obtained by aggregating the subword embeddings.
        Args:
            embedded_utterances: (batch_size, max_bert_len, emb_dim)
            word_ids: (batch_size, max_bert_len)
        Returns:
            merged_embeddings: (batch_size, utt_len, emb_dim)
        """
        assert embedded_utterances.shape[0] == len(word_ids)

        word_ids = torch.where(word_ids == -100, self.max_utt_len - 1, word_ids)
        merged_embeddings = scatter_mean(embedded_utterances, word_ids, dim=1)

        merged_embeddings = pad_sequence(
            sequences=merged_embeddings,
            max_len=self.max_utt_len,
            padding_value=-100,
        )
        return merged_embeddings

    def _get_embedding_dim(self):
        """
        Obtain the dimension of the embedding from the configuration of BertModel
        """
        return self.embedder.config.dim
