#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import logging
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

from funasr.build_utils.build_model_from_file import build_model_from_file
from funasr.torch_utils.device_funcs import to_device
from funasr.utils.misc import statistic_model_parameters


class Speech2Xvector:
    """Speech2Xvector class

    Examples:
        >>> import librosa
        >>> speech2xvector = Speech2Xvector("sv_config.yml", "sv.pb")
        >>> audio, rate = librosa.load("speech.wav")
        >>> speech2xvector(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
            self,
            sv_train_config: Union[Path, str] = None,
            sv_model_file: Union[Path, str] = None,
            device: str = "cpu",
            batch_size: int = 1,
            dtype: str = "float32",
            streaming: bool = False,
            embedding_node: str = "resnet1_dense",
    ):

        # TODO: 1. Build SV model
        sv_model, sv_train_args = build_model_from_file(
            config_file=sv_train_config,
            model_file=sv_model_file,
            cmvn_file=None,
            device=device,
            task_name="sv",
            mode="sv",
        )
        logging.info("sv_model: {}".format(sv_model))
        logging.info("model parameter number: {}".format(statistic_model_parameters(sv_model)))
        logging.info("sv_train_args: {}".format(sv_train_args))
        sv_model.to(dtype=getattr(torch, dtype)).eval()

        self.sv_model = sv_model
        self.sv_train_args = sv_train_args
        self.device = device
        self.dtype = dtype
        self.embedding_node = embedding_node

    @torch.no_grad()
    def calculate_embedding(self, speech: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        speech = speech.unsqueeze(0).to(getattr(torch, self.dtype))
        # lengths: (1,)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}

        # a. To device
        batch = to_device(batch, device=self.device)

        # b. Forward Encoder
        enc, ilens = self.sv_model.encode(**batch)

        # c. Forward Pooling
        pooling = self.sv_model.pooling_layer(enc)

        # d. Forward Decoder
        outputs, embeddings = self.sv_model.decoder(pooling)

        if self.embedding_node not in embeddings:
            raise ValueError("Required embedding node {} not in {}".format(
                self.embedding_node, embeddings.keys()))

        return embeddings[self.embedding_node]

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray],
            ref_speech: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """Inference

        Args:
            speech: Input speech data
            ref_speech: Reference speech to compare
        Returns:
            embedding, ref_embedding, similarity_score

        """
        self.sv_model.eval()
        embedding = self.calculate_embedding(speech)
        ref_emb, score = None, None
        if ref_speech is not None:
            ref_emb = self.calculate_embedding(ref_speech)
            score = torch.cosine_similarity(embedding, ref_emb)

        results = (embedding, ref_emb, score)
        return results
