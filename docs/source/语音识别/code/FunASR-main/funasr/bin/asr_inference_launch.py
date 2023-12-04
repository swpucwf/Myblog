#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import argparse
import logging
from optparse import Option
import os
import sys
import time
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torchaudio
# import librosa
import librosa
import yaml

from funasr.bin.asr_infer import Speech2Text
from funasr.bin.asr_infer import Speech2TextMFCCA
from funasr.bin.asr_infer import Speech2TextParaformer, Speech2TextParaformerOnline
from funasr.bin.asr_infer import Speech2TextSAASR
from funasr.bin.asr_infer import Speech2TextTransducer
from funasr.bin.asr_infer import Speech2TextUniASR
from funasr.bin.asr_infer import Speech2TextWhisper
from funasr.bin.punc_infer import Text2Punc
from funasr.bin.tp_infer import Speech2Timestamp
from funasr.bin.vad_infer import Speech2VadSegment
from funasr.build_utils.build_streaming_iterator import build_streaming_iterator
from funasr.fileio.datadir_writer import DatadirWriter
from funasr.modules.beam_search.beam_search import Hypothesis
from funasr.modules.subsampling import TooShortUttError
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.set_all_random_seed import set_all_random_seed
from funasr.utils import asr_utils, postprocess_utils
from funasr.utils import config_argparse
from funasr.utils.cli_utils import get_commandline_args
from funasr.utils.timestamp_tools import time_stamp_sentence, ts_prediction_lfr6_standard
from funasr.utils.types import str2bool
from funasr.utils.types import str2triple_str
from funasr.utils.types import str_or_none
from funasr.utils.vad_utils import slice_padding_fbank
from funasr.utils.speaker_utils import (check_audio_list, 
                                        sv_preprocess, 
                                        sv_chunk, 
                                        CAMPPlus, 
                                        extract_feature, 
                                        postprocess,
                                        distribute_spk)
from funasr.build_utils.build_model_from_file import build_model_from_file
from funasr.utils.cluster_backend import ClusterBackend
from funasr.utils.modelscope_utils import get_cache_dir
from tqdm import tqdm

def inference_asr(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        beam_size: int,
        ngpu: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        log_level: Union[int, str],
        # data_path_and_name_and_type,
        asr_train_config: Optional[str],
        asr_model_file: Optional[str],
        cmvn_file: Optional[str] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        token_type: Optional[str] = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        bpemodel: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        streaming: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        num_workers: int = 1,
        mc: bool = False,
        param_dict: dict = None,
        **kwargs,
):
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
    )
    logging.info("speech2text_kwargs: {}".format(speech2text_kwargs))
    speech2text = Speech2Text(**speech2text_kwargs)

    def _forward(data_path_and_name_and_type,
                 raw_inputs: Union[np.ndarray, torch.Tensor] = None,
                 output_dir_v2: Optional[str] = None,
                 fs: dict = None,
                 param_dict: dict = None,
                 **kwargs,
                 ):
        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = build_streaming_iterator(
            task_name="asr",
            preprocess_args=speech2text.asr_train_args,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            fs=fs,
            mc=mc,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
        )

        finish_count = 0
        file_count = 1
        # 7 .Start for-loop
        # FIXME(kamo): The output format should be discussed about
        asr_result_list = []
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            writer = DatadirWriter(output_path)
        else:
            writer = None

        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            # batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, hyp_object)
            try:
                results = speech2text(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["sil"], [2], hyp]] * nbest

            # Only supporting batch_size==1
            key = keys[0]
            for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                if writer is not None:
                    ibest_writer = writer[f"{n}best_recog"]

                    # Write the result to each file
                    ibest_writer["token"][key] = " ".join(token)
                    ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["score"][key] = str(hyp.score)

                if text is not None:
                    text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
                    item = {'key': key, 'value': text_postprocessed}
                    asr_result_list.append(item)
                    finish_count += 1
                    asr_utils.print_progress(finish_count / file_count)
                    if writer is not None:
                        ibest_writer["text"][key] = text

                logging.info("uttid: {}".format(key))
                logging.info("text predictions: {}\n".format(text))
        return asr_result_list

    return _forward


def inference_paraformer(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        beam_size: int,
        ngpu: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        log_level: Union[int, str],
        # data_path_and_name_and_type,
        asr_train_config: Optional[str],
        asr_model_file: Optional[str],
        cmvn_file: Optional[str] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        token_type: Optional[str] = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        bpemodel: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        dtype: str = "float32",
        seed: int = 0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        num_workers: int = 1,
        output_dir: Optional[str] = None,
        timestamp_infer_config: Union[Path, str] = None,
        timestamp_model_file: Union[Path, str] = None,
        param_dict: dict = None,
        decoding_ind: int = 0,
        **kwargs,
):
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)

    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    export_mode = False
    if param_dict is not None:
        hotword_list_or_file = param_dict.get('hotword')
        export_mode = param_dict.get("export_mode", False)
        clas_scale = param_dict.get('clas_scale', 1.0)
    else:
        hotword_list_or_file = None
        clas_scale = 1.0

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size = 1

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        hotword_list_or_file=hotword_list_or_file,
        clas_scale=clas_scale,
        decoding_ind=decoding_ind,
    )

    speech2text = Speech2TextParaformer(**speech2text_kwargs)

    if timestamp_model_file is not None:
        speechtext2timestamp = Speech2Timestamp(
            timestamp_cmvn_file=cmvn_file,
            timestamp_model_file=timestamp_model_file,
            timestamp_infer_config=timestamp_infer_config,
        )
    else:
        speechtext2timestamp = None

    def _forward(
            data_path_and_name_and_type,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            output_dir_v2: Optional[str] = None,
            fs: dict = None,
            param_dict: dict = None,
            **kwargs,
    ):

        decoding_ind = None
        hotword_list_or_file = None
        if param_dict is not None:
            hotword_list_or_file = param_dict.get('hotword')
        if 'hotword' in kwargs and kwargs['hotword'] is not None:
            hotword_list_or_file = kwargs['hotword']
        if hotword_list_or_file is not None or 'hotword' in kwargs:
            speech2text.hotword_list = speech2text.generate_hotwords_list(hotword_list_or_file)
        if param_dict is not None and "decoding_ind" in param_dict:
            decoding_ind = param_dict["decoding_ind"]

        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = build_streaming_iterator(
            task_name="asr",
            preprocess_args=speech2text.asr_train_args,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            fs=fs,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
        )

        if param_dict is not None:
            use_timestamp = param_dict.get('use_timestamp', True)
        else:
            use_timestamp = True

        forward_time_total = 0.0
        length_total = 0.0
        finish_count = 0
        file_count = 1
        # 7 .Start for-loop
        # FIXME(kamo): The output format should be discussed about
        asr_result_list = []
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            writer = DatadirWriter(output_path)
        else:
            writer = None

        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            # batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}

            logging.info("decoding, utt_id: {}".format(keys))
            # N-best list of (text, token, token_int, hyp_object)

            time_beg = time.time()
            batch["decoding_ind"] = decoding_ind
            results = speech2text(**batch)
            if len(results) < 1:
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["sil"], [2], hyp, 10, 6, []]] * nbest
            time_end = time.time()
            forward_time = time_end - time_beg
            lfr_factor = results[0][-1]
            length = results[0][-2]
            forward_time_total += forward_time
            length_total += length
            rtf_cur = "decoding, feature length: {}, forward_time: {:.4f}, rtf: {:.4f}".format(length, forward_time,
                                                                                               100 * forward_time / (
                                                                                                       length * lfr_factor))
            logging.info(rtf_cur)

            for batch_id in range(_bs):
                result = [results[batch_id][:-2]]

                key = keys[batch_id]
                for n, result in zip(range(1, nbest + 1), result):
                    text, token, token_int, hyp = result[0], result[1], result[2], result[3]
                    timestamp = result[4] if len(result[4]) > 0 else None
                    # conduct timestamp prediction here
                    # timestamp inference requires token length
                    # thus following inference cannot be conducted in batch
                    if timestamp is None and speechtext2timestamp:
                        ts_batch = {}
                        ts_batch['speech'] = batch['speech'][batch_id].unsqueeze(0)
                        ts_batch['speech_lengths'] = torch.tensor([batch['speech_lengths'][batch_id]])
                        ts_batch['text_lengths'] = torch.tensor([len(token)])
                        us_alphas, us_peaks = speechtext2timestamp(**ts_batch)
                        ts_str, timestamp = ts_prediction_lfr6_standard(us_alphas[0], us_peaks[0], token,
                                                                        force_time_shift=-3.0)
                    # Create a directory: outdir/{n}best_recog
                    if writer is not None:
                        ibest_writer = writer[f"{n}best_recog"]

                        # Write the result to each file
                        ibest_writer["token"][key] = " ".join(token)
                        # ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                        ibest_writer["score"][key] = str(hyp.score)
                        ibest_writer["rtf"][key] = rtf_cur

                    if text is not None:
                        if use_timestamp and timestamp is not None and len(timestamp):
                            postprocessed_result = postprocess_utils.sentence_postprocess(token, timestamp)
                        else:
                            postprocessed_result = postprocess_utils.sentence_postprocess(token)
                        timestamp_postprocessed = ""
                        if len(postprocessed_result) == 3:
                            text_postprocessed, timestamp_postprocessed, word_lists = postprocessed_result[0], \
                                                                                      postprocessed_result[1], \
                                                                                      postprocessed_result[2]
                        else:
                            text_postprocessed, word_lists = postprocessed_result[0], postprocessed_result[1]
                        item = {'key': key, 'value': text_postprocessed}
                        if timestamp_postprocessed != "":
                            item['timestamp'] = timestamp_postprocessed
                        asr_result_list.append(item)
                        finish_count += 1
                        # asr_utils.print_progress(finish_count / file_count)
                        if writer is not None:
                            ibest_writer["text"][key] = " ".join(word_lists)

                    logging.info("decoding, utt: {}, predictions: {}".format(key, text))
        rtf_avg = "decoding, feature length total: {}, forward_time total: {:.4f}, rtf avg: {:.4f}".format(length_total,
                                                                                                           forward_time_total,
                                                                                                           100 * forward_time_total / (
                                                                                                                   length_total * lfr_factor))
        logging.info(rtf_avg)
        if writer is not None:
            ibest_writer["rtf"]["rtf_avf"] = rtf_avg
        torch.cuda.empty_cache()
        return asr_result_list

    return _forward


def inference_paraformer_vad_punc(
        maxlenratio: float=0.0,
        minlenratio: float=0.0,
        batch_size: int=1,
        beam_size: int=1,
        ngpu: int=1,
        ctc_weight: float=0.0,
        lm_weight: float=0.0,
        penalty: float=0.0,
        log_level: Union[int, str]=logging.ERROR,
        # data_path_and_name_and_type,
        asr_train_config: Optional[str]=None,
        asr_model_file: Optional[str]=None,
        cmvn_file: Optional[str] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        token_type: Optional[str] = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        bpemodel: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        num_workers: int = 0,
        vad_infer_config: Optional[str] = None,
        vad_model_file: Optional[str] = None,
        vad_cmvn_file: Optional[str] = None,
        time_stamp_writer: bool = True,
        punc_infer_config: Optional[str] = None,
        punc_model_file: Optional[str] = None,
        outputs_dict: Optional[bool] = True,
        param_dict: dict = None,
        **kwargs,
):
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)
    language = kwargs.get("model_lang", None)

    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if param_dict is not None:
        hotword_list_or_file = param_dict.get('hotword')
    else:
        hotword_list_or_file = None

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2vadsegment
    speech2vadsegment_kwargs = dict(
        vad_infer_config=vad_infer_config,
        vad_model_file=vad_model_file,
        vad_cmvn_file=vad_cmvn_file,
        device=device,
        dtype=dtype,
    )
    # logging.info("speech2vadsegment_kwargs: {}".format(speech2vadsegment_kwargs))
    speech2vadsegment = Speech2VadSegment(**speech2vadsegment_kwargs)

    # 3. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        hotword_list_or_file=hotword_list_or_file,
    )
    speech2text = Speech2TextParaformer(**speech2text_kwargs)
    text2punc = None
    if punc_model_file is not None:
        text2punc = Text2Punc(punc_infer_config, punc_model_file, device=device, dtype=dtype)

    if output_dir is not None:
        writer = DatadirWriter(output_dir)
        ibest_writer = writer[f"1best_recog"]
        ibest_writer["token_list"][""] = " ".join(speech2text.asr_train_args.token_list)

    def _forward(data_path_and_name_and_type,
                 raw_inputs: Union[np.ndarray, torch.Tensor] = None,
                 output_dir_v2: Optional[str] = None,
                 fs: dict = None,
                 param_dict: dict = None,
                 **kwargs,
                 ):

        hotword_list_or_file = None
        if param_dict is not None:
            hotword_list_or_file = param_dict.get('hotword')

        if 'hotword' in kwargs:
            hotword_list_or_file = kwargs['hotword']

        speech2vadsegment.vad_model.vad_opts.max_single_segment_time = kwargs.get("max_single_segment_time", 60000)
        batch_size_token_threshold_s = kwargs.get("batch_size_token_threshold_s", int(speech2vadsegment.vad_model.vad_opts.max_single_segment_time*0.67/1000)) * 1000
        batch_size_token = kwargs.get("batch_size_token", 6000)
        print("batch_size_token: ", batch_size_token)

        if speech2text.hotword_list is None:
            speech2text.hotword_list = speech2text.generate_hotwords_list(hotword_list_or_file)

        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = build_streaming_iterator(
            task_name="asr",
            preprocess_args=None,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            fs=fs,
            batch_size=1,
            key_file=key_file,
            num_workers=num_workers,
        )

        if param_dict is not None:
            use_timestamp = param_dict.get('use_timestamp', True)
        else:
            use_timestamp = True

        finish_count = 0
        file_count = 1
        lfr_factor = 6
        # 7 .Start for-loop
        asr_result_list = []
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        writer = None
        if output_path is not None:
            writer = DatadirWriter(output_path)
            ibest_writer = writer[f"1best_recog"]

        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            beg_vad = time.time()
            vad_results = speech2vadsegment(**batch)
            end_vad = time.time()
            print("time cost vad: ", end_vad - beg_vad)
            _, vadsegments = vad_results[0], vad_results[1][0]

            speech, speech_lengths = batch["speech"], batch["speech_lengths"]

            n = len(vadsegments)
            data_with_index = [(vadsegments[i], i) for i in range(n)]
            sorted_data = sorted(data_with_index, key=lambda x: x[0][1] - x[0][0])
            results_sorted = []
            
            if not len(sorted_data):
                key = keys[0]
                # no active segments after VAD
                if writer is not None:
                    # Write empty results
                    ibest_writer["token"][key] = ""
                    ibest_writer["token_int"][key] = ""
                    ibest_writer["vad"][key] = ""
                    ibest_writer["text"][key] = ""
                    ibest_writer["text_with_punc"][key] = ""
                    if use_timestamp:
                        ibest_writer["time_stamp"][key] = ""

                logging.info("decoding, utt: {}, empty speech".format(key))
                continue

            batch_size_token_ms = batch_size_token*60
            if speech2text.device == "cpu":
                batch_size_token_ms = 0
            if len(sorted_data) > 0 and len(sorted_data[0]) > 0:
                batch_size_token_ms = max(batch_size_token_ms, sorted_data[0][0][1] - sorted_data[0][0][0])
            
            batch_size_token_ms_cum = 0
            beg_idx = 0
            beg_asr_total = time.time()
            for j, _ in enumerate(tqdm(range(0, n))):
                batch_size_token_ms_cum += (sorted_data[j][0][1] - sorted_data[j][0][0])
                if j < n - 1 and (batch_size_token_ms_cum + sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size_token_ms and (sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size_token_threshold_s:
                    continue
                batch_size_token_ms_cum = 0
                end_idx = j + 1
                speech_j, speech_lengths_j = slice_padding_fbank(speech, speech_lengths, sorted_data[beg_idx:end_idx])
                beg_idx = end_idx
                batch = {"speech": speech_j, "speech_lengths": speech_lengths_j}
                batch = to_device(batch, device=device)

                beg_asr = time.time()
                results = speech2text(**batch)
                end_asr = time.time()
                if speech2text.device != "cpu":
                    print("batch: ", speech_j.shape[0])
                    print("time cost asr: ", end_asr - beg_asr)

                if len(results) < 1:
                    results = [["", [], [], [], [], [], []]]
                results_sorted.extend(results)
            end_asr_total = time.time()
            print("total time cost asr: ", end_asr_total-beg_asr_total)
            restored_data = [0] * n
            for j in range(n):
                index = sorted_data[j][1]
                restored_data[index] = results_sorted[j]
            result = ["", [], [], [], [], [], []]
            for j in range(n):
                result[0] += restored_data[j][0]
                result[1] += restored_data[j][1]
                result[2] += restored_data[j][2]
                if len(restored_data[j][4]) > 0:
                    for t in restored_data[j][4]:
                        t[0] += vadsegments[j][0]
                        t[1] += vadsegments[j][0]
                    result[4] += restored_data[j][4]
                # result = [result[k]+restored_data[j][k] for k in range(len(result[:-2]))]

            key = keys[0]
            # result = result_segments[0]
            text, token, token_int = result[0], result[1], result[2]
            time_stamp = result[4] if len(result[4]) > 0 else None

            if language == "en-bpe":
                postprocessed_result = postprocess_utils.sentence_postprocess_sentencepiece(token)
            else:
                if use_timestamp and time_stamp is not None and len(time_stamp):
                    postprocessed_result = postprocess_utils.sentence_postprocess(token, time_stamp)
                else:
                    postprocessed_result = postprocess_utils.sentence_postprocess(token)
            text_postprocessed = ""
            time_stamp_postprocessed = ""
            text_postprocessed_punc = postprocessed_result
            if len(postprocessed_result) == 3:
                text_postprocessed, time_stamp_postprocessed, word_lists = postprocessed_result[0], \
                                                                           postprocessed_result[1], \
                                                                           postprocessed_result[2]
            else:
                text_postprocessed, word_lists = postprocessed_result[0], postprocessed_result[1]

            text_postprocessed_punc = text_postprocessed
            punc_id_list = []
            if len(word_lists) > 0 and text2punc is not None:
                beg_punc = time.time()
                text_postprocessed_punc, punc_id_list = text2punc(word_lists, 20)
                end_punc = time.time()
                print("time cost punc: ", end_punc - beg_punc)

            item = {'key': key, 'value': text_postprocessed_punc}
            if text_postprocessed != "":
                item['text_postprocessed'] = text_postprocessed
            if time_stamp_postprocessed != "":
                item['time_stamp'] = time_stamp_postprocessed

            item['sentences'] = time_stamp_sentence(punc_id_list, time_stamp_postprocessed, text_postprocessed)

            asr_result_list.append(item)
            finish_count += 1
            # asr_utils.print_progress(finish_count / file_count)
            if writer is not None:
                # Write the result to each file
                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["vad"][key] = "{}".format(vadsegments)
                ibest_writer["text"][key] = " ".join(word_lists)
                ibest_writer["text_with_punc"][key] = text_postprocessed_punc
                if time_stamp_postprocessed is not None:
                    ibest_writer["time_stamp"][key] = "{}".format(time_stamp_postprocessed)

            logging.info("decoding, utt: {}, predictions: {}".format(key, text_postprocessed_punc))
        torch.cuda.empty_cache()
        return asr_result_list

    return _forward


def inference_paraformer_vad_speaker(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        beam_size: int,
        ngpu: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        log_level: Union[int, str],
        # data_path_and_name_and_type,
        asr_train_config: Optional[str],
        asr_model_file: Optional[str],
        cmvn_file: Optional[str] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        token_type: Optional[str] = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        bpemodel: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        num_workers: int = 1,
        vad_infer_config: Optional[str] = None,
        vad_model_file: Optional[str] = None,
        vad_cmvn_file: Optional[str] = None,
        time_stamp_writer: bool = True,
        punc_infer_config: Optional[str] = None,
        punc_model_file: Optional[str] = None,
        sv_model_file: Optional[str] = None, 
        streaming: bool = False,
        embedding_node: str = "resnet1_dense",
        sv_threshold: float = 0.9465,
        outputs_dict: Optional[bool] = True,
        param_dict: dict = None,

        **kwargs,
):
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)

    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    sv_model_file = asr_model_file.replace("model.pb", "campplus_cn_common.bin")

    if param_dict is not None:
        hotword_list_or_file = param_dict.get('hotword')
    else:
        hotword_list_or_file = None

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2vadsegment
    speech2vadsegment_kwargs = dict(
        vad_infer_config=vad_infer_config,
        vad_model_file=vad_model_file,
        vad_cmvn_file=vad_cmvn_file,
        device=device,
        dtype=dtype,
    )
    # logging.info("speech2vadsegment_kwargs: {}".format(speech2vadsegment_kwargs))
    speech2vadsegment = Speech2VadSegment(**speech2vadsegment_kwargs)

    # 3. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        hotword_list_or_file=hotword_list_or_file,
    )
    speech2text = Speech2TextParaformer(**speech2text_kwargs)
    text2punc = None
    if punc_model_file is not None:
        text2punc = Text2Punc(punc_infer_config, punc_model_file, device=device, dtype=dtype)

    if output_dir is not None:
        writer = DatadirWriter(output_dir)
        ibest_writer = writer[f"1best_recog"]
        ibest_writer["token_list"][""] = " ".join(speech2text.asr_train_args.token_list)

    def _forward(data_path_and_name_and_type,
                 raw_inputs: Union[np.ndarray, torch.Tensor] = None,
                 output_dir_v2: Optional[str] = None,
                 fs: dict = None,
                 param_dict: dict = None,
                 **kwargs,
                 ):

        hotword_list_or_file = None
        if param_dict is not None:
            hotword_list_or_file = param_dict.get('hotword')

        if 'hotword' in kwargs:
            hotword_list_or_file = kwargs['hotword']

        speech2vadsegment.vad_model.vad_opts.max_single_segment_time = kwargs.get("max_single_segment_time", 60000)
        batch_size_token_threshold_s = kwargs.get("batch_size_token_threshold_s", int(speech2vadsegment.vad_model.vad_opts.max_single_segment_time*0.67/1000)) * 1000
        batch_size_token = kwargs.get("batch_size_token", 6000)
        print("batch_size_token: ", batch_size_token)

        if speech2text.hotword_list is None:
            speech2text.hotword_list = speech2text.generate_hotwords_list(hotword_list_or_file)

        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = build_streaming_iterator(
            task_name="asr",
            preprocess_args=None,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            fs=fs,
            batch_size=1,
            key_file=key_file,
            num_workers=num_workers,
        )

        if param_dict is not None:
            use_timestamp = param_dict.get('use_timestamp', True)
        else:
            use_timestamp = True

        finish_count = 0
        file_count = 1
        lfr_factor = 6
        # 7 .Start for-loop
        asr_result_list = []
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        writer = None
        if output_path is not None:
            writer = DatadirWriter(output_path)
            ibest_writer = writer[f"1best_recog"]

        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            beg_vad = time.time()
            vad_results = speech2vadsegment(**batch)
            end_vad = time.time()
            print("time cost vad: ", end_vad - beg_vad)
            _, vadsegments = vad_results[0], vad_results[1][0]
            ##################################
            #####  speaker_verification  #####
            ##################################
            # load sv model
            sv_model_dict = torch.load(sv_model_file, map_location=torch.device('cpu'))
            sv_model = CAMPPlus()
            sv_model.load_state_dict(sv_model_dict)
            sv_model.eval()
            cb_model = ClusterBackend()
            vad_segments = []
            audio = batch['speech'].numpy().reshape(-1)
            for vadsegment in vadsegments:
                st = int(vadsegment[0]) / 1000
                ed = int(vadsegment[1]) / 1000
                vad_segments.append(
                    [st, ed, audio[int(st * 16000):int(ed * 16000)]])
            audio_dur = check_audio_list(vad_segments)
            if audio_dur > 5:
                # sv pipeline
                segments = sv_chunk(vad_segments)
                embeddings = []
                for s in segments:
                    #_, embs = self.sv_pipeline([s[2]], output_emb=True)
                    # embeddings.append(embs)
                    wavs = sv_preprocess([s[2]])
                    # embs = self.forward(wavs)
                    embs = []
                    for x in wavs:
                        x = extract_feature([x])
                        embs.append(sv_model(x))
                    embs = torch.cat(embs)
                    embeddings.append(embs.detach().numpy())
                embeddings = np.concatenate(embeddings)
                labels = cb_model(embeddings)
                sv_output = postprocess(segments, vad_segments, labels, embeddings)
            else:
                # fake speaker res for too shot utterance
                sv_output = [[0.0, vadsegments[-1][-1]/1000.0, 0]]
                logging.warning("Too short utterence found: {}, return default speaker results.".format(keys))

            speech, speech_lengths = batch["speech"], batch["speech_lengths"]

            n = len(vadsegments)
            data_with_index = [(vadsegments[i], i) for i in range(n)]
            sorted_data = sorted(data_with_index, key=lambda x: x[0][1] - x[0][0])
            results_sorted = []
            
            if not len(sorted_data):
                key = keys[0]
                # no active segments after VAD
                if writer is not None:
                    # Write empty results
                    ibest_writer["token"][key] = ""
                    ibest_writer["token_int"][key] = ""
                    ibest_writer["vad"][key] = ""
                    ibest_writer["text"][key] = ""
                    ibest_writer["text_with_punc"][key] = ""
                    if use_timestamp:
                        ibest_writer["time_stamp"][key] = ""

                logging.info("decoding, utt: {}, empty speech".format(key))
                continue

            batch_size_token_ms = batch_size_token*60
            if speech2text.device == "cpu":
                batch_size_token_ms = 0
            if len(sorted_data) > 0 and len(sorted_data[0]) > 0:
                batch_size_token_ms = max(batch_size_token_ms, sorted_data[0][0][1] - sorted_data[0][0][0])
            
            batch_size_token_ms_cum = 0
            beg_idx = 0
            beg_asr_total = time.time()
            for j, _ in enumerate(tqdm(range(0, n))):
                batch_size_token_ms_cum += (sorted_data[j][0][1] - sorted_data[j][0][0])
                if j < n - 1 and (batch_size_token_ms_cum + sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size_token_ms and (sorted_data[j + 1][0][1] - sorted_data[j + 1][0][0]) < batch_size_token_threshold_s:
                    continue
                batch_size_token_ms_cum = 0
                end_idx = j + 1
                speech_j, speech_lengths_j = slice_padding_fbank(speech, speech_lengths, sorted_data[beg_idx:end_idx])
                beg_idx = end_idx
                batch = {"speech": speech_j, "speech_lengths": speech_lengths_j}
                batch = to_device(batch, device=device)
                # print("batch: ", speech_j.shape[0])
                beg_asr = time.time()
                results = speech2text(**batch)
                end_asr = time.time()
                # print("time cost asr: ", end_asr - beg_asr)

                if len(results) < 1:
                    results = [["", [], [], [], [], [], []]]
                results_sorted.extend(results)
            end_asr_total = time.time()
            print("total time cost asr: ", end_asr_total-beg_asr_total)
            restored_data = [0] * n
            for j in range(n):
                index = sorted_data[j][1]
                restored_data[index] = results_sorted[j]
            result = ["", [], [], [], [], [], []]
            for j in range(n):
                result[0] += restored_data[j][0]
                result[1] += restored_data[j][1]
                result[2] += restored_data[j][2]
                if len(restored_data[j][4]) > 0:
                    for t in restored_data[j][4]:
                        t[0] += vadsegments[j][0]
                        t[1] += vadsegments[j][0]
                    result[4] += restored_data[j][4]
                # result = [result[k]+restored_data[j][k] for k in range(len(result[:-2]))]

            key = keys[0]
            # result = result_segments[0]
            text, token, token_int = result[0], result[1], result[2]
            time_stamp = result[4] if len(result[4]) > 0 else None

            if use_timestamp and time_stamp is not None and len(time_stamp):
                postprocessed_result = postprocess_utils.sentence_postprocess(token, time_stamp)
            else:
                postprocessed_result = postprocess_utils.sentence_postprocess(token)
            text_postprocessed = ""
            time_stamp_postprocessed = ""
            text_postprocessed_punc = postprocessed_result
            if len(postprocessed_result) == 3:
                text_postprocessed, time_stamp_postprocessed, word_lists = postprocessed_result[0], \
                                                                           postprocessed_result[1], \
                                                                           postprocessed_result[2]
            else:
                text_postprocessed, word_lists = postprocessed_result[0], postprocessed_result[1]

            text_postprocessed_punc = text_postprocessed
            punc_id_list = []
            if len(word_lists) > 0 and text2punc is not None:
                beg_punc = time.time()
                text_postprocessed_punc, punc_id_list = text2punc(word_lists, 20)
                end_punc = time.time()
                print("time cost punc: ", end_punc - beg_punc)

            item = {'key': key, 'value': text_postprocessed_punc}
            if text_postprocessed != "":
                item['text_postprocessed'] = text_postprocessed
            if time_stamp_postprocessed != "":
                item['time_stamp'] = time_stamp_postprocessed

            item['sentences'] = time_stamp_sentence(punc_id_list, time_stamp_postprocessed, text_postprocessed)

            asr_result_list.append(item)
            finish_count += 1
            # asr_utils.print_progress(finish_count / file_count)
            if writer is not None:
                # Write the result to each file
                ibest_writer["token"][key] = " ".join(token)
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["vad"][key] = "{}".format(vadsegments)
                ibest_writer["text"][key] = " ".join(word_lists)
                ibest_writer["text_with_punc"][key] = text_postprocessed_punc
                if time_stamp_postprocessed is not None:
                    ibest_writer["time_stamp"][key] = "{}".format(time_stamp_postprocessed)

            logging.info("decoding, utt: {}, predictions: {}".format(key, text_postprocessed_punc))
        torch.cuda.empty_cache()
        distribute_spk(asr_result_list[0]['sentences'], sv_output)
        return asr_result_list

    return _forward


def inference_paraformer_online(
        maxlenratio: float=0.0,
        minlenratio: float=0.0,
        batch_size: int=1,
        beam_size: int=1,
        ngpu: int=1,
        ctc_weight: float=0.0,
        lm_weight: float=0.0,
        penalty: float=0.0,
        log_level: Union[int, str]=logging.ERROR,
        # data_path_and_name_and_type,
        asr_train_config: Optional[str]=None,
        asr_model_file: Optional[str]=None,
        cmvn_file: Optional[str] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        token_type: Optional[str] = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        bpemodel: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        dtype: str = "float32",
        seed: int = 0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        num_workers: int = 1,
        output_dir: Optional[str] = None,
        param_dict: dict = None,
        **kwargs,
):

    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    export_mode = False

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        batch_size = 1

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
    )

    speech2text = Speech2TextParaformerOnline(**speech2text_kwargs)

    def _load_bytes(input):
        middle_data = np.frombuffer(input, dtype=np.int16)
        middle_data = np.asarray(middle_data)
        if middle_data.dtype.kind not in 'iu':
            raise TypeError("'middle_data' must be an array of integers")
        dtype = np.dtype('float32')
        if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(middle_data.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
        return array

    def _read_yaml(yaml_path: Union[str, Path]) -> Dict:
        if not Path(yaml_path).exists():
            raise FileExistsError(f'The {yaml_path} does not exist.')

        with open(str(yaml_path), 'rb') as f:
            data = yaml.load(f, Loader=yaml.Loader)
        return data

    def _prepare_cache(cache: dict = {}, chunk_size=[5, 10, 5], encoder_chunk_look_back=0,
                       decoder_chunk_look_back=0, batch_size=1):
        if len(cache) > 0:
            return cache
        config = _read_yaml(asr_train_config)
        enc_output_size = config["encoder_conf"]["output_size"]
        feats_dims = config["frontend_conf"]["n_mels"] * config["frontend_conf"]["lfr_m"]
        cache_en = {"start_idx": 0, "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)),
                    "cif_alphas": torch.zeros((batch_size, 1)), "chunk_size": chunk_size,
                    "encoder_chunk_look_back": encoder_chunk_look_back, "last_chunk": False, "opt": None,
                    "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)), "tail_chunk": False}
        cache["encoder"] = cache_en

        cache_de = {"decode_fsmn": None, "decoder_chunk_look_back": decoder_chunk_look_back, "opt": None, "chunk_size": chunk_size}
        cache["decoder"] = cache_de

        return cache

    def _cache_reset(cache: dict = {}, chunk_size=[5, 10, 5], encoder_chunk_look_back=0,
                     decoder_chunk_look_back=0, batch_size=1):
        if len(cache) > 0:
            config = _read_yaml(asr_train_config)
            enc_output_size = config["encoder_conf"]["output_size"]
            feats_dims = config["frontend_conf"]["n_mels"] * config["frontend_conf"]["lfr_m"]
            cache_en = {"start_idx": 0, "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)),
                        "cif_alphas": torch.zeros((batch_size, 1)), "chunk_size": chunk_size,
                        "encoder_chunk_look_back": encoder_chunk_look_back, "last_chunk": False, "opt": None,
                        "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)), "tail_chunk": False}
            cache["encoder"] = cache_en

            cache_de = {"decode_fsmn": None, "decoder_chunk_look_back": decoder_chunk_look_back, "opt": None, "chunk_size": chunk_size}
            cache["decoder"] = cache_de

        return cache

    #def _prepare_cache(cache: dict = {}, chunk_size=[5, 10, 5], batch_size=1):
    #    if len(cache) > 0:
    #        return cache
    #    config = _read_yaml(asr_train_config)
    #    enc_output_size = config["encoder_conf"]["output_size"]
    #    feats_dims = config["frontend_conf"]["n_mels"] * config["frontend_conf"]["lfr_m"]
    #    cache_en = {"start_idx": 0, "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)),
    #                "cif_alphas": torch.zeros((batch_size, 1)), "chunk_size": chunk_size, "last_chunk": False,
    #                "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)), "tail_chunk": False}
    #    cache["encoder"] = cache_en

    #    cache_de = {"decode_fsmn": None}
    #    cache["decoder"] = cache_de

    #    return cache

    #def _cache_reset(cache: dict = {}, chunk_size=[5, 10, 5], batch_size=1):
    #    if len(cache) > 0:
    #        config = _read_yaml(asr_train_config)
    #        enc_output_size = config["encoder_conf"]["output_size"]
    #        feats_dims = config["frontend_conf"]["n_mels"] * config["frontend_conf"]["lfr_m"]
    #        cache_en = {"start_idx": 0, "cif_hidden": torch.zeros((batch_size, 1, enc_output_size)),
    #                    "cif_alphas": torch.zeros((batch_size, 1)), "chunk_size": chunk_size, "last_chunk": False,
    #                    "feats": torch.zeros((batch_size, chunk_size[0] + chunk_size[2], feats_dims)),
    #                    "tail_chunk": False}
    #        cache["encoder"] = cache_en

    #        cache_de = {"decode_fsmn": None}
    #        cache["decoder"] = cache_de

    #    return cache

    def _forward(
            data_path_and_name_and_type,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            output_dir_v2: Optional[str] = None,
            fs: dict = None,
            param_dict: dict = None,
            **kwargs,
    ):

        # 3. Build data-iterator
        if data_path_and_name_and_type is not None and data_path_and_name_and_type[2] == "bytes":
            raw_inputs = _load_bytes(data_path_and_name_and_type[0])
            raw_inputs = torch.tensor(raw_inputs)
        if data_path_and_name_and_type is not None and data_path_and_name_and_type[2] == "sound":
            try:
                raw_inputs = torchaudio.load(data_path_and_name_and_type[0])[0][0]
            except:
                # raw_inputs = librosa.load(data_path_and_name_and_type[0], dtype='float32')[0]
                raw_inputs, sr = librosa.load(data_path_and_name_and_type[0], dtype='float32')
                if raw_inputs.ndim == 2:
                    raw_inputs = raw_inputs[:, 0]
                raw_inputs = torch.tensor(raw_inputs)
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, np.ndarray):
                raw_inputs = torch.tensor(raw_inputs)
        is_final = False
        cache = {}
        chunk_size = [5, 10, 5]
        encoder_chunk_look_back = 0
        decoder_chunk_look_back = 0
        if param_dict is not None and "cache" in param_dict:
            cache = param_dict["cache"]
        if param_dict is not None and "is_final" in param_dict:
            is_final = param_dict["is_final"]
        if param_dict is not None and "chunk_size" in param_dict:
            chunk_size = param_dict["chunk_size"]
        if param_dict is not None and "encoder_chunk_look_back" in param_dict:
            encoder_chunk_look_back = param_dict["encoder_chunk_look_back"]
            if encoder_chunk_look_back > 0:
                chunk_size[0] = 0
        if param_dict is not None and "decoder_chunk_look_back" in param_dict:
            decoder_chunk_look_back = param_dict["decoder_chunk_look_back"]

        # 7 .Start for-loop
        # FIXME(kamo): The output format should be discussed about
        raw_inputs = torch.unsqueeze(raw_inputs, axis=0)
        asr_result_list = []
        cache = _prepare_cache(cache, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, 
                               decoder_chunk_look_back=decoder_chunk_look_back, batch_size=1)
        item = {}
        if data_path_and_name_and_type is not None and data_path_and_name_and_type[2] == "sound":
            sample_offset = 0
            speech_length = raw_inputs.shape[1]
            stride_size = chunk_size[1] * 960
            cache = _prepare_cache(cache, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, 
                                   decoder_chunk_look_back=decoder_chunk_look_back, batch_size=1)
            final_result = ""
            for sample_offset in range(0, speech_length, min(stride_size, speech_length - sample_offset)):
                if sample_offset + stride_size >= speech_length - 1:
                    stride_size = speech_length - sample_offset
                    cache["encoder"]["is_final"] = True
                else:
                    cache["encoder"]["is_final"] = False
                input_lens = torch.tensor([stride_size])
                asr_result = speech2text(cache, raw_inputs[:, sample_offset: sample_offset + stride_size], input_lens)
                if len(asr_result) != 0:
                    final_result += " ".join(asr_result) + " "
            item = {'key': "utt", 'value': final_result.strip()}
        else:
            input_lens = torch.tensor([raw_inputs.shape[1]])
            cache["encoder"]["is_final"] = is_final
            asr_result = speech2text(cache, raw_inputs, input_lens)
            item = {'key': "utt", 'value': " ".join(asr_result)}

        asr_result_list.append(item)
        if is_final:
            cache = _cache_reset(cache, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, 
                                 decoder_chunk_look_back=decoder_chunk_look_back, batch_size=1)
        return asr_result_list

    return _forward


def inference_uniasr(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        beam_size: int,
        ngpu: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        log_level: Union[int, str],
        # data_path_and_name_and_type,
        asr_train_config: Optional[str],
        asr_model_file: Optional[str],
        ngram_file: Optional[str] = None,
        cmvn_file: Optional[str] = None,
        # raw_inputs: Union[np.ndarray, torch.Tensor] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        token_type: Optional[str] = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        bpemodel: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        streaming: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        num_workers: int = 1,
        token_num_relax: int = 1,
        decoding_ind: int = 0,
        decoding_mode: str = "model1",
        param_dict: dict = None,
        **kwargs,
):
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if param_dict is not None and "decoding_model" in param_dict:
        if param_dict["decoding_model"] == "fast":
            decoding_ind = 0
            decoding_mode = "model1"
        elif param_dict["decoding_model"] == "normal":
            decoding_ind = 0
            decoding_mode = "model2"
        elif param_dict["decoding_model"] == "offline":
            decoding_ind = 1
            decoding_mode = "model2"
        else:
            raise NotImplementedError("unsupported decoding model {}".format(param_dict["decoding_model"]))

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        ngram_file=ngram_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
        token_num_relax=token_num_relax,
        decoding_ind=decoding_ind,
        decoding_mode=decoding_mode,
    )
    speech2text = Speech2TextUniASR(**speech2text_kwargs)

    def _forward(data_path_and_name_and_type,
                 raw_inputs: Union[np.ndarray, torch.Tensor] = None,
                 output_dir_v2: Optional[str] = None,
                 fs: dict = None,
                 param_dict: dict = None,
                 **kwargs,
                 ):
        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = build_streaming_iterator(
            task_name="asr",
            preprocess_args=speech2text.asr_train_args,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            fs=fs,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
        )

        finish_count = 0
        file_count = 1
        # 7 .Start for-loop
        # FIXME(kamo): The output format should be discussed about
        asr_result_list = []
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            writer = DatadirWriter(output_path)
        else:
            writer = None

        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            # batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, hyp_object)
            try:
                results = speech2text(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["sil"], [2], hyp]] * nbest

            # Only supporting batch_size==1
            key = keys[0]
            logging.info(f"Utterance: {key}")
            for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                if writer is not None:
                    ibest_writer = writer[f"{n}best_recog"]

                    # Write the result to each file
                    ibest_writer["token"][key] = " ".join(token)
                    # ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["score"][key] = str(hyp.score)

                if text is not None:
                    text_postprocessed, word_lists = postprocess_utils.sentence_postprocess(token)
                    item = {'key': key, 'value': text_postprocessed}
                    asr_result_list.append(item)
                    finish_count += 1
                    asr_utils.print_progress(finish_count / file_count)
                    if writer is not None:
                        ibest_writer["text"][key] = " ".join(word_lists)
        return asr_result_list

    return _forward


def inference_mfcca(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        beam_size: int,
        ngpu: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        log_level: Union[int, str],
        # data_path_and_name_and_type,
        asr_train_config: Optional[str],
        asr_model_file: Optional[str],
        cmvn_file: Optional[str] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        token_type: Optional[str] = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        bpemodel: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        streaming: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        num_workers: int = 1,
        param_dict: dict = None,
        **kwargs,
):
    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
    )
    logging.info("speech2text_kwargs: {}".format(speech2text_kwargs))
    speech2text = Speech2TextMFCCA(**speech2text_kwargs)

    def _forward(data_path_and_name_and_type,
                 raw_inputs: Union[np.ndarray, torch.Tensor] = None,
                 output_dir_v2: Optional[str] = None,
                 fs: dict = None,
                 param_dict: dict = None,
                 **kwargs,
                 ):
        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = build_streaming_iterator(
            task_name="asr",
            preprocess_args=speech2text.asr_train_args,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            fs=fs,
            mc=True,
            key_file=key_file,
            num_workers=num_workers,
        )

        finish_count = 0
        file_count = 1
        # 7 .Start for-loop
        # FIXME(kamo): The output format should be discussed about
        asr_result_list = []
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            writer = DatadirWriter(output_path)
        else:
            writer = None

        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            # batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, hyp_object)
            try:
                results = speech2text(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["<space>"], [2], hyp]] * nbest

            # Only supporting batch_size==1
            key = keys[0]
            for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                if writer is not None:
                    ibest_writer = writer[f"{n}best_recog"]

                    # Write the result to each file
                    ibest_writer["token"][key] = " ".join(token)
                    # ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["score"][key] = str(hyp.score)

                if text is not None:
                    text_postprocessed = postprocess_utils.sentence_postprocess(token)
                    item = {'key': key, 'value': text_postprocessed}
                    asr_result_list.append(item)
                    finish_count += 1
                    asr_utils.print_progress(finish_count / file_count)
                    if writer is not None:
                        ibest_writer["text"][key] = text
        return asr_result_list

    return _forward


def inference_transducer(
        output_dir: str,
        batch_size: int,
        dtype: str,
        beam_size: int,
        ngpu: int,
        seed: int,
        lm_weight: float,
        nbest: int,
        num_workers: int,
        log_level: Union[int, str],
        # data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
        asr_train_config: Optional[str],
        asr_model_file: Optional[str],
        cmvn_file: Optional[str] = None,
        beam_search_config: Optional[dict] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        model_tag: Optional[str] = None,
        token_type: Optional[str] = None,
        bpemodel: Optional[str] = None,
        key_file: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        quantize_asr_model: Optional[bool] = False,
        quantize_modules: Optional[List[str]] = None,
        quantize_dtype: Optional[str] = "float16",
        streaming: Optional[bool] = False,
        fake_streaming: Optional[bool] = False,
        full_utt: Optional[bool] = False,
        chunk_size: Optional[int] = 16,
        left_context: Optional[int] = 16,
        right_context: Optional[int] = 0,
        display_partial_hypotheses: bool = False,
        **kwargs,
) -> None:
    """Transducer model inference.
    Args:
        output_dir: Output directory path.
        batch_size: Batch decoding size.
        dtype: Data type.
        beam_size: Beam size.
        ngpu: Number of GPUs.
        seed: Random number generator seed.
        lm_weight: Weight of language model.
        nbest: Number of final hypothesis.
        num_workers: Number of workers.
        log_level: Level of verbose for logs.
        data_path_and_name_and_type:
        asr_train_config: ASR model training config path.
        asr_model_file: ASR model path.
        beam_search_config: Beam search config path.
        lm_train_config: Language Model training config path.
        lm_file: Language Model path.
        model_tag: Model tag.
        token_type: Type of token units.
        bpemodel: BPE model path.
        key_file: File key.
        allow_variable_data_keys: Whether to allow variable data keys.
        quantize_asr_model: Whether to apply dynamic quantization to ASR model.
        quantize_modules: List of module names to apply dynamic quantization on.
        quantize_dtype: Dynamic quantization data type.
        streaming: Whether to perform chunk-by-chunk inference.
        chunk_size: Number of frames in chunk AFTER subsampling.
        left_context: Number of frames in left context AFTER subsampling.
        right_context: Number of frames in right context AFTER subsampling.
        display_partial_hypotheses: Whether to display partial hypotheses.
    """

    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        beam_search_config=beam_search_config,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        dtype=dtype,
        beam_size=beam_size,
        lm_weight=lm_weight,
        nbest=nbest,
        quantize_asr_model=quantize_asr_model,
        quantize_modules=quantize_modules,
        quantize_dtype=quantize_dtype,
        streaming=streaming,
        fake_streaming=fake_streaming,
        full_utt=full_utt,
        chunk_size=chunk_size,
        left_context=left_context,
        right_context=right_context,
    )
    speech2text = Speech2TextTransducer(**speech2text_kwargs)

    def _forward(data_path_and_name_and_type,
                 raw_inputs: Union[np.ndarray, torch.Tensor] = None,
                 output_dir_v2: Optional[str] = None,
                 fs: dict = None,
                 param_dict: dict = None,
                 **kwargs,
                 ):
        # 3. Build data-iterator
        loader = build_streaming_iterator(
            task_name="asr",
            preprocess_args=speech2text.asr_train_args,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
        )
        asr_result_list = []

        if output_dir is not None:
            writer = DatadirWriter(output_dir)
        else:
            writer = None

        # 4 .Start for-loop
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys

            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            assert len(batch.keys()) == 1

            try:
                if speech2text.streaming:
                    speech = batch["speech"]

                    _steps = len(speech) // speech2text._ctx
                    _end = 0
                    for i in range(_steps):
                        _end = (i + 1) * speech2text._ctx

                        speech2text.streaming_decode(
                            speech[i * speech2text._ctx: _end + speech2text._right_ctx], is_final=False
                        )

                    final_hyps = speech2text.streaming_decode(
                        speech[_end: len(speech)], is_final=True
                    )
                elif speech2text.fake_streaming:
                    final_hyps = speech2text.fake_streaming_decode(**batch)
                elif speech2text.full_utt:
                    final_hyps = speech2text.full_utt_decode(**batch)
                else:
                    final_hyps = speech2text(**batch)

                results = speech2text.hypotheses_to_results(final_hyps)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, yseq=[], dec_state=None)
                results = [[" ", ["<space>"], [2], hyp]] * nbest

            key = keys[0]
            for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                item = {'key': key, 'value': text}
                asr_result_list.append(item)
                if writer is not None:
                    ibest_writer = writer[f"{n}best_recog"]

                    ibest_writer["token"][key] = " ".join(token)
                    ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["score"][key] = str(hyp.score)

                    if text is not None:
                        ibest_writer["text"][key] = text

                logging.info("decoding, utt: {}, predictions: {}".format(key, text))
        return asr_result_list
    return _forward


def inference_sa_asr(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        beam_size: int,
        ngpu: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        log_level: Union[int, str],
        # data_path_and_name_and_type,
        asr_train_config: Optional[str],
        asr_model_file: Optional[str],
        cmvn_file: Optional[str] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        token_type: Optional[str] = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        bpemodel: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        streaming: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        num_workers: int = 1,
        mc: bool = False,
        param_dict: dict = None,
        **kwargs,
):
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
    )
    logging.info("speech2text_kwargs: {}".format(speech2text_kwargs))
    speech2text = Speech2TextSAASR(**speech2text_kwargs)

    def _forward(data_path_and_name_and_type,
                 raw_inputs: Union[np.ndarray, torch.Tensor] = None,
                 output_dir_v2: Optional[str] = None,
                 fs: dict = None,
                 param_dict: dict = None,
                 **kwargs,
                 ):
        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = build_streaming_iterator(
            task_name="asr",
            preprocess_args=speech2text.asr_train_args,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            fs=fs,
            mc=mc,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
        )

        finish_count = 0
        file_count = 1
        # 7 .Start for-loop
        # FIXME(kamo): The output format should be discussed about
        asr_result_list = []
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            writer = DatadirWriter(output_path)
        else:
            writer = None

        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            # batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            # N-best list of (text, token, token_int, hyp_object)
            try:
                results = speech2text(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["sil"], [2], hyp]] * nbest

            # Only supporting batch_size==1
            key = keys[0]
            for n, (text, text_id, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                if writer is not None:
                    ibest_writer = writer[f"{n}best_recog"]

                    # Write the result to each file
                    ibest_writer["token"][key] = " ".join(token)
                    ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["score"][key] = str(hyp.score)
                    ibest_writer["text_id"][key] = text_id

                if text is not None:
                    text_postprocessed, _ = postprocess_utils.sentence_postprocess(token)
                    item = {'key': key, 'value': text_postprocessed}
                    asr_result_list.append(item)
                    finish_count += 1
                    asr_utils.print_progress(finish_count / file_count)
                    if writer is not None:
                        ibest_writer["text"][key] = text

                logging.info("uttid: {}".format(key))
                logging.info("text predictions: {}".format(text))
                logging.info("text_id predictions: {}\n".format(text_id))
        return asr_result_list

    return _forward

def inference_whisper(
        maxlenratio: float,
        minlenratio: float,
        batch_size: int,
        beam_size: int,
        ngpu: int,
        ctc_weight: float,
        lm_weight: float,
        penalty: float,
        log_level: Union[int, str],
        # data_path_and_name_and_type,
        asr_train_config: Optional[str],
        asr_model_file: Optional[str],
        cmvn_file: Optional[str] = None,
        lm_train_config: Optional[str] = None,
        lm_file: Optional[str] = None,
        token_type: Optional[str] = None,
        key_file: Optional[str] = None,
        word_lm_train_config: Optional[str] = None,
        bpemodel: Optional[str] = None,
        allow_variable_data_keys: bool = False,
        streaming: bool = False,
        output_dir: Optional[str] = None,
        dtype: str = "float32",
        seed: int = 0,
        ngram_weight: float = 0.9,
        nbest: int = 1,
        num_workers: int = 1,
        mc: bool = False,
        param_dict: dict = None,
        **kwargs,
):

    ncpu = kwargs.get("ncpu", 1)
    torch.set_num_threads(ncpu)
    if param_dict:
        language = param_dict.get("language", None)
        task = param_dict.get("task", "transcribe")
    else:
        language = None
        task = "transcribe"
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text_kwargs = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        cmvn_file=cmvn_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        ngram_weight=ngram_weight,
        penalty=penalty,
        nbest=nbest,
        streaming=streaming,
        language=language,
        task=task,
    )
    logging.info("speech2text_kwargs: {}".format(speech2text_kwargs))
    speech2text = Speech2TextWhisper(**speech2text_kwargs)

    def _forward(data_path_and_name_and_type,
                 raw_inputs: Union[np.ndarray, torch.Tensor] = None,
                 output_dir_v2: Optional[str] = None,
                 fs: dict = None,
                 param_dict: dict = None,
                 **kwargs,
                 ):
        # 3. Build data-iterator
        if data_path_and_name_and_type is None and raw_inputs is not None:
            if isinstance(raw_inputs, torch.Tensor):
                raw_inputs = raw_inputs.numpy()
            data_path_and_name_and_type = [raw_inputs, "speech", "waveform"]
        loader = build_streaming_iterator(
            task_name="asr",
            preprocess_args=speech2text.asr_train_args,
            data_path_and_name_and_type=data_path_and_name_and_type,
            dtype=dtype,
            fs=fs,
            mc=mc,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
        )

        finish_count = 0
        file_count = 1
        # 7 .Start for-loop
        # FIXME(kamo): The output format should be discussed about
        asr_result_list = []
        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            writer = DatadirWriter(output_path)
        else:
            writer = None

        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"
            # batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            # N-best list of (text, token, token_int, hyp_object)
            try:
                results = speech2text(**batch)
            except TooShortUttError as e:
                logging.warning(f"Utterance {keys} {e}")
                hyp = Hypothesis(score=0.0, scores={}, states={}, yseq=[])
                results = [[" ", ["sil"], [2], hyp]] * nbest

            # Only supporting batch_size==1
            key = keys[0]

            for n, (text, language) in zip(range(1, nbest + 1), results):
                # Create a directory: outdir/{n}best_recog
                if writer is not None:
                    ibest_writer = writer[f"{n}best_recog"]

                    # Write the result to each file
                    ibest_writer["language"][key] = language

                if text is not None:
                    item = {'key': key, 'value': text}
                    asr_result_list.append(item)
                    finish_count += 1
                    if writer is not None:
                        ibest_writer["text"][key] = text

                logging.info("uttid: {}".format(key))
                logging.info("text predictions: {}\n".format(text))
        return asr_result_list

    return _forward

def inference_launch(**kwargs):
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        logging.info("Unknown decoding mode.")
        return None
    if mode == "asr":
        return inference_asr(**kwargs)
    elif mode == "uniasr":
        return inference_uniasr(**kwargs)
    elif mode == "paraformer":
        return inference_paraformer(**kwargs)
    elif mode == "paraformer_fake_streaming":
        return inference_paraformer(**kwargs)
    elif mode == "paraformer_streaming":
        return inference_paraformer_online(**kwargs)
    elif mode.startswith("paraformer_vad_speaker"):
        return inference_paraformer_vad_speaker(**kwargs)
    elif mode.startswith("paraformer_vad"):
        return inference_paraformer_vad_punc(**kwargs)
    elif mode == "mfcca":
        return inference_mfcca(**kwargs)
    elif mode == "rnnt":
        return inference_transducer(**kwargs)
    elif mode == "bat":
        return inference_transducer(**kwargs)
    elif mode == "sa_asr":
        return inference_sa_asr(**kwargs)
    elif mode == "whisper":
        return inference_whisper(**kwargs)
    else:
        logging.info("Unknown decoding mode: {}".format(mode))
        return None

def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    from funasr.bin.argument import get_parser
    parser = get_parser()
    parser.add_argument(
        "--mode",
        type=str,
        default="asr",
        help="The decoding mode",
    )
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)

    # set logging messages
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.info("Decoding args: {}".format(kwargs))

    # gpu setting
    if args.ngpu > 0:
        jobid = int(args.output_dir.split(".")[-1])
        gpuid = args.gpuid_list.split(",")[(jobid - 1) // args.njob]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

    inference_pipeline = inference_launch(**kwargs)
    return inference_pipeline(kwargs["data_path_and_name_and_type"], hotword=kwargs.get("hotword", None))


if __name__ == "__main__":
    main()
