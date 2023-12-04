import argparse
import logging
import os
import uuid

import aiofiles
import ffmpeg
import uvicorn
from fastapi import FastAPI, File, UploadFile, Body
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="0.0.0.0",
                    required=False,
                    help="host ip, localhost, 0.0.0.0")
parser.add_argument("--port",
                    type=int,
                    default=8000,
                    required=False,
                    help="server port")
parser.add_argument("--asr_model",
                    type=str,
                    default="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    help="offline asr model from modelscope")
parser.add_argument("--vad_model",
                    type=str,
                    default="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    help="vad model from modelscope")
parser.add_argument("--punc_model",
                    type=str,
                    default="damo/punc_ct-transformer_cn-en-common-vocab471067-large",
                    help="punc model from modelscope")
parser.add_argument("--ngpu",
                    type=int,
                    default=1,
                    help="0 for cpu, 1 for gpu")
parser.add_argument("--ncpu",
                    type=int,
                    default=4,
                    help="cpu cores")
parser.add_argument("--hotword_path",
                    type=str,
                    default=None,
                    help="hot word txt path, only the hot word model works")
parser.add_argument("--certfile",
                    type=str,
                    default=None,
                    required=False,
                    help="certfile for ssl")
parser.add_argument("--keyfile",
                    type=str,
                    default=None,
                    required=False,
                    help="keyfile for ssl")
parser.add_argument("--temp_dir",
                    type=str,
                    default="temp_dir/",
                    required=False,
                    help="temp dir")
args = parser.parse_args()
print("-----------  Configuration Arguments -----------")
for arg, value in vars(args).items():
    print("%s: %s" % (arg, value))
print("------------------------------------------------")


os.makedirs(args.temp_dir, exist_ok=True)

print("model loading")
param_dict = {}
if args.hotword_path is not None and os.path.exists(args.hotword_path):
    param_dict['hotword'] = args.hotword_path
# asr
inference_pipeline_asr = pipeline(task=Tasks.auto_speech_recognition,
                                  model=args.asr_model,
                                  vad_model=args.vad_model,
                                  ngpu=args.ngpu,
                                  ncpu=args.ncpu,
                                  param_dict=param_dict)
print(f'loaded asr models.')

if args.punc_model != "":
    inference_pipeline_punc = pipeline(task=Tasks.punctuation,
                                       model=args.punc_model,
                                       ngpu=args.ngpu,
                                       ncpu=args.ncpu)
    print(f'loaded pun models.')
else:
    inference_pipeline_punc = None

app = FastAPI(title="FunASR")


@app.post("/recognition")
async def api_recognition(audio: UploadFile = File(..., description="audio file"),
                          add_pun: int = Body(1, description="add punctuation", embed=True)):
    suffix = audio.filename.split('.')[-1]
    audio_path = f'{args.temp_dir}/{str(uuid.uuid1())}.{suffix}'
    async with aiofiles.open(audio_path, 'wb') as out_file:
        content = await audio.read()
        await out_file.write(content)
    audio_bytes, _ = (
        ffmpeg.input(audio_path, threads=0)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
    rec_result = inference_pipeline_asr(audio_in=audio_bytes, param_dict={})
    if add_pun:
        rec_result = inference_pipeline_punc(text_in=rec_result['text'], param_dict={'cache': list()})
    ret = {"results": rec_result['text'], "code": 0}
    print(ret)
    return ret


if __name__ == '__main__':
    uvicorn.run(app, host=args.host, port=args.port, ssl_keyfile=args.keyfile, ssl_certfile=args.certfile)
