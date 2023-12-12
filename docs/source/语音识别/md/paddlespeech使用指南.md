#  

[TOC]

paddlespeech使用指南

## 1. 安装环境

### 1.1 基本环境安装

```
+ gcc >= 4.8.5
+ paddlepaddle <= 2.5.1
+ python >= 3.8
+ linux(推荐), mac, windows
**Linux** 环境下，*3.8* 以上版本的 *python* 上安装 PaddleSpeech
```

```
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
# 安装2.4.1版本. 注意：2.4.1只是一个示例，请按照对paddlepaddle的最小依赖进行选择。
pip install paddlepaddle==2.4.1 -i https://mirror.baidu.com/pypi/simple
# 安装 develop 版本
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
```

### 1.2 pip 安装

```shell
pip install pytest-runner
pip install paddlespeech
```

### 1.3 源码编译

```shell
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech
pip install pytest-runner
pip install .
```

更多关于安装问题，如 conda 环境，librosa 依赖的系统库，gcc 环境问题，kaldi 安装等，可以参考这篇[安装文档](docs/source/install_cn.md)，如安装上遇到问题可以在 [#2150](https://github.com/PaddlePaddle/PaddleSpeech/issues/2150) 上留言以及查找相关问题

## 2. 快速使用

### 2.1 语音合成

```shell
paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！" --output output.wav
```

```python
from paddlespeech.cli.tts.infer import TTSExecutor
tts = TTSExecutor()
tts(text="今天天气十分不错。", output="output.wav")
```

- 语音合成的 web demo 已经集成进了 [Huggingface Spaces](https://huggingface.co/spaces). 请参考: [TTS Demo](https://huggingface.co/spaces/KPatrick/PaddleSpeechTTS)

### 2.2声音分类

 527 个类别的声音分类模型

```shell
paddlespeech cls --input zh.wav
```

```python
from paddlespeech.cli.cls.infer import CLSExecutor
cls = CLSExecutor()
result = cls(audio_file="zh.wav")
print(result)
```

### 2.3 声纹提取

```shell
paddlespeech vector --task spk --input zh.wav
```

```python
from paddlespeech.cli.vector import VectorExecutor
vec = VectorExecutor()
result = vec(audio_file="zh.wav")
print(result) # 187维向量
[ -0.19083306   9.474295   -14.122263    -2.0916545    0.04848729
   4.9295826    1.4780062    0.3733844   10.695862     3.2697146
  -4.48199     -0.6617882   -9.170393   -11.1568775   -1.2358263 ...]
```

![image-20231212092748711](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images202312120927030.png)

###2.4 标点恢复

```shell
paddlespeech text --task punc --input 今天的天气真不错啊你下午有空吗我想约你一起去吃饭
```

```python
from paddlespeech.cli.text.infer import TextExecutor
text_punc = TextExecutor()
result = text_punc(text="今天的天气真不错啊你下午有空吗我想约你一起去吃饭")
```

![image-20231212093217593](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images202312120932050.png)

### 2.5 语音翻译

```shell
paddlespeech st --input en.wav
```

```python
from paddlespeech.cli.st.infer import STExecutor
st = STExecutor()
result = st(audio_file="en.wav")
```

## 3. 快速使用服务

 AI Studio 中快速体验：[SpeechServer 一键部署](https://aistudio.baidu.com/aistudio/projectdetail/4354592?sUid=2470186&shared=1&ts=1660878208266)

### 3.1 命令行

#### 3.1.1 启动服务

```shell
paddlespeech_server start --config_file ./demos/speech_server/conf/application.yaml
```

#### 3.1.2 访问语音识别服务

```shell
paddlespeech_client asr --server_ip 127.0.0.1 --port 8090 --input input_16k.wav
```

#### 3.1.3 访问语音合成服务

```shell
paddlespeech_client tts --server_ip 127.0.0.1 --port 8090 --input "您好，欢迎使用语音合成服务。" --output output.wav
```

#### 3.1.4 访问音频分类服务

```shell
paddlespeech_client cls --server_ip 127.0.0.1 --port 8090 --input input.wav
```

### 3.2 快速使用流式服务

开发者可以尝试 [流式 ASR](./demos/streaming_asr_server/README.md) 和 [流式 TTS](./demos/streaming_tts_server/README.md) 服务.

#### 3.2.1启动流式 ASR 服务

```
paddlespeech_server start --config_file ./demos/streaming_asr_server/conf/application.yaml
```

#### 3.2.2 访问流式 ASR 服务

```
paddlespeech_client asr_online --server_ip 127.0.0.1 --port 8090 --input input_16k.wav
```

#### 3.2.3启动流式 TTS 服务

```
paddlespeech_server start --config_file ./demos/streaming_tts_server/conf/tts_online_application.yaml
```

#### 3.2.4 访问流式 TTS 服务

```
paddlespeech_client tts_online --server_ip 127.0.0.1 --port 8092 --protocol http --input "您好，欢迎使用百度合成服务。" --output output.wav
```

## 4. 训练

