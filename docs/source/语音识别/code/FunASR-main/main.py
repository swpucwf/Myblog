from funasr import infer

p = infer(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", model_hub="ms")

res = p("Rs_16k.wav", batch_size_token=5000)
print(res)