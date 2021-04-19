from typing import Optional

import torch
from torch import Tensor, nn
from yukarin_sa.config import NetworkConfig
from yukarin_sa.network.encoder import EncoderType, create_encoder


class Predictor(nn.Module):
    def __init__(
        self,
        phoneme_size: int,
        phoneme_embedding_size: int,
        phoneme_encoder_type: EncoderType,
        phoneme_encoder_hidden_size: int,
        phoneme_encoder_kernel_size: int,
        phoneme_encoder_layer_num: int,
        accent_encoder_type: EncoderType,
        accent_encoder_hidden_size: int,
        accent_encoder_kernel_size: int,
        accent_encoder_layer_num: int,
        speaker_size: int,
        speaker_embedding_size: int,
    ):
        super().__init__()

        self.phoneme_embedder = nn.Embedding(
            num_embeddings=phoneme_size + 1,  # empty consonant
            embedding_dim=phoneme_embedding_size,
            padding_idx=0,
        )
        self.speaker_embedder = (
            nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
            if speaker_size > 0
            else None
        )

        self.phoneme_encoder = create_encoder(
            type=phoneme_encoder_type,
            input_size=phoneme_embedding_size + speaker_embedding_size,
            hidden_size=phoneme_encoder_hidden_size,
            kernel_size=phoneme_encoder_kernel_size,
            layer_num=phoneme_encoder_layer_num,
        )

        self.accent_encoder = create_encoder(
            type=accent_encoder_type,
            input_size=4 + speaker_embedding_size,
            hidden_size=accent_encoder_hidden_size,
            kernel_size=accent_encoder_kernel_size,
            layer_num=accent_encoder_layer_num,
        )

        input_size = (
            self.phoneme_encoder.output_hidden_size
            + self.accent_encoder.output_hidden_size
        )
        self.post = nn.Conv1d(input_size, 2, kernel_size=1)

    def forward(
        self,
        phoneme_list: Tensor,  # (batch_size, length)
        consonant_phoneme_list: Optional[Tensor],  # (batch_size, length)
        start_accent_list: Tensor,  # (batch_size, length)
        end_accent_list: Tensor,  # (batch_size, length)
        start_accent_phrase_list: Tensor,  # (batch_size, length)
        end_accent_phrase_list: Tensor,  # (batch_size, length)
        speaker_id: Optional[Tensor],  # (batch_size, )
    ):
        ph = self.phoneme_embedder(phoneme_list + 1)  # (batch_size, length, ?)
        if consonant_phoneme_list is not None:
            ph = ph + self.phoneme_embedder(consonant_phoneme_list + 1)
        ph = ph.transpose(1, 2)  # (batch_size, ?, length)

        ah = torch.stack(
            [
                start_accent_list,
                end_accent_list,
                start_accent_phrase_list,
                end_accent_phrase_list,
            ],
            dim=1,
        ).to(
            ph.dtype
        )  # (batch, ?, length)

        if self.speaker_embedder is not None and speaker_id is not None:
            speaker_id = self.speaker_embedder(speaker_id)  # (batch_size, ?)
            speaker_id = speaker_id.unsqueeze(2)  # (batch_size, ?, 1)
            speaker = speaker_id.expand(
                speaker_id.shape[0], speaker_id.shape[1], ph.shape[2]
            )  # (batch_size, ?, length)
            ph = torch.cat((ph, speaker), dim=1)  # (batch_size, ?, length)
            ah = torch.cat((ah, speaker), dim=1)  # (batch_size, ?, length)

        ph = self.phoneme_encoder(ph)  # (batch_size, ?, length)
        ah = self.accent_encoder(ah)  # (batch_size, ?, length)
        h = torch.cat((ph, ah), dim=1)  # (batch_size, ?, length)
        h = self.post(h)  # (batch_size, ?, length)

        phoneme_length = h[:, 0, :]  # (batch_size, length)
        f0 = h[:, 1, :]  # (batch_size, length)
        return phoneme_length, f0


def create_predictor(config: NetworkConfig):
    return Predictor(
        phoneme_size=config.phoneme_size,
        phoneme_embedding_size=config.phoneme_embedding_size,
        phoneme_encoder_type=EncoderType(config.phoneme_encoder_type),
        phoneme_encoder_hidden_size=config.phoneme_encoder_hidden_size,
        phoneme_encoder_kernel_size=config.phoneme_encoder_kernel_size,
        phoneme_encoder_layer_num=config.phoneme_encoder_layer_num,
        accent_encoder_type=EncoderType(config.accent_encoder_type),
        accent_encoder_hidden_size=config.accent_encoder_hidden_size,
        accent_encoder_kernel_size=config.accent_encoder_kernel_size,
        accent_encoder_layer_num=config.accent_encoder_layer_num,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
    )
