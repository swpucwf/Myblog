
from funasr import infer

p = infer(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", model_hub="ms")





for i in range(100):
    res = p("./Rs_16k.wav", batch_size_token=1000)
    print(res)