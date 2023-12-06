[TOC]

# 基于MASR框架实现语音识别-windows

## 1. 环境安装

### 1.1 基本环境安装

默认已经安装conda，测试环境下，

python>=3.8

默认使用Python3.8的虚拟环境

首先安装的是Pytorch 1.13.1的GPU版本

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

### 1.2 MASR安装

**建议源码安装**，源码安装能保证使用最新代码。

```shell
git clone https://github.com/yeyupiaoling/MASR.git
cd MASR
pip install .
```

## 2.数据准备

### 2.1 开源数据集下载

在`download_data`目录下是公开数据集的下载和制作训练数据列表和词汇表的，本项目提供了下载公开的中文普通话语音数据集，分别是Aishell，Free ST-Chinese-Mandarin-Corpus，THCHS-30 这三个数据集，总大小超过28G。下载这三个数据只需要执行一下代码即可，当然如果想快速训练，也可以只下载其中一个。**注意：** `noise.py`可下载可不下载，这是用于训练时数据增强的，如果不想使用噪声数据增强，可以不用下载。

```shell script
cd download_data/
python aishell.py
python free_st_chinese_mandarin_corpus.py
python thchs_30.py
python noise.py
```

**注意：** 这样下载慢，可以获取程序中的`DATA_URL`单独下载，用迅雷等下载工具，这样下载速度快很多。然后把下载的压缩文件放在`dataset/audio`目录下，就会自动跳过下载，直接解压文件文本生成数据列表。

### 2.2 自定义数据集制作

如果开发者有自己的数据集，可以使用自己的数据集进行训练，当然也可以跟上面下载的数据集一起训练。

#### 2.2.1 自定义数据集要求

- 音频的采样率==16000Hz

如果音频采样率不一致，参考2.2.2数据集准备后，在`create_data.py`中提供了统一音频数据的采样率转换为16000Hz，请设置is_change_frame_rate`==True

![image-20231206093443561](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images202312060934791.png)

#### 2.2.2 数据集准备

- 语音文件(.wav)需要放在`dataset/audio/`目录下

- 数据列表文件存在`dataset/annotation/`目录下，要注意的是该中文文本只能包含纯中文，不能包含标点符号、阿拉伯数字以及英文字母。

##### 中文数据列表文件

```
dataset/audio/wav/0175/H0175A0171.wav   我需要把空调温度调到二十度
dataset/audio/wav/0175/H0175A0377.wav   出彩中国人
dataset/audio/wav/0175/H0175A0470.wav   据克而瑞研究中心监测
dataset/audio/wav/0175/H0175A0180.wav   把温度加大到十八
```

##### 英文数据列表文件

```
dataset/audio/LibriSpeech/dev-clean/1272/135031/1272-135031-0004.flac   the king has fled in disgrace and your friends are asking for you
dataset/audio/LibriSpeech/dev-clean/1272/135031/1272-135031-0005.flac   i begged ruggedo long ago to send him away but he would not do so
dataset/audio/LibriSpeech/dev-clean/1272/135031/1272-135031-0006.flac   i also offered to help your brother to escape but he would not go
dataset/audio/LibriSpeech/dev-clean/1272/135031/1272-135031-0007.flac   he eats and sleeps very steadily replied the new king
dataset/audio/LibriSpeech/dev-clean/1272/135031/1272-135031-0008.flac   i hope he doesn't work too hard said shaggy
```

执行以下文件，作用如下：

```shell
python create_data.py
```

- 将自定义数据集生成三个JSON格式的数据列表，分别是`manifest.test、manifest.train、manifest.noise`。
- 建立词汇表，把所有出现的字符都存放子在`vocabulary.txt`文件中，一行一个字符。
- 计算均值和标准差用于归一化，默认使用全部的语音计算均值和标准差，并将结果保存在`mean_istd.json`中。

以上生成的文件都存放在`dataset/`目录下。数据划分说明，如果`dataset/annotation`存在`test.txt`，那全部测试数据都使用这个数据，否则使用全部数据的1/500的数据，直到指定的最大测试数据量。

#### 2.2.3减少音频文件数量

- 针对**超大数据**的情况，例如有数万小时乃至数十万小时的语音数据，因为音频大多数都是一些短语音频，所以音频文件数量会超级多
- 功能: **多段短语音合成一个较长的音频文件，大幅度减少音频数量。**

使用方法如下，在创建数据列表的时候。指定参数`is_merge_audio`为`True`，这样的话就会把长语音合成短语音。默认的参数是合成10分钟的长语音。以平均每条音频10秒来计算的话，音频文件数量就会少60倍。但是并不会影响模型的训练，因为在模型训练的时候也是按照一个短语音片段裁剪出来进行训练的。

```shell
python create_data.py --is_merge_audio=True
```

![image-20231206093834733](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images202312060938535.png)

#### 2.2.4 减小数据列表文件

- **适用于训练数据多，数据文件的文件特别大**，读取数据的时候需要把全部的数据列表都加入到内存中
- 功能：**把数据列表转化成二进制**，在读取列表的时候，只需要加载较小的数据列表索引就可以。这样可以减少4~8倍的内存占用，一定程度上也提高了数据的读取速度。

使用方法如下，修改配置文件中的`manifest_type`参数，指定 其值为`binary`，这样在执行`create_data.py`创建数据列表的时候，就会多生成一份对应的二进制的数据列表，`.data`后缀的是数据列表的二进制文件，`.header`后缀是二进制数据列表的索引文件。然后在训练的时候也会只读取这个二进制的数据列表。

```yaml
# 数据集参数
dataset_conf:
  # 数据列表类型，支持txt、binary
  manifest_type: 'binary'
```

具体操作步骤如下：

![image-20231206094307794](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images202312060943975.png)

![image-20231206094610669](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/images202312060946369.png)

### 2.3 常见公开数据集

#### 2.3.1 公开数据集

|      数据集      |  语言  |  时长  |     大小      |                           下载地址                           |
| :--------------: | :----: | :----: | :-----------: | :----------------------------------------------------------: |
|     THCHS30      | 普通话 |  40h   |     6.01G     | [data_thchs30.tgz](http://openslr.magicdatatech.com/resources/18/data_thchs30.tgz) |
|     ST-CMDS      | 普通话 |  100h  |     7.67G     | [ST-CMDS-20170001_1-OS.tar.gz](http://openslr.magicdatatech.com/resources/38/ST-CMDS-20170001_1-OS.tar.gz) |
|    AIShell-1     | 普通话 |  178h  |    14.51G     | [data_aishell.tgz](http://openslr.magicdatatech.com/resources/33/data_aishell.tgz) |
|    Primewords    | 普通话 |  100h  |     8.44G     | [primewords_md_2018_set1.tar.gz](http://openslr.magicdatatech.com/resources/47/primewords_md_2018_set1.tar.gz) |
| aidatatang_200zh | 普通话 |  200h  |    17.47G     | [aidatatang_200zh.tgz](http://openslr.magicdatatech.com/resources/62/aidatatang_200zh.tgz) |
|    MagicData     | 普通话 |  755h  | 52G/1.0G/2.2G | [train_set.tar.gz](http://openslr.magicdatatech.com/resources/68/train_set.tar.gz>) / [dev_set.tar.gz](http://openslr.magicdatatech.com/resources/68/dev_set.tar.gz) / [test_set.tar.gz](http://openslr.magicdatatech.com/resources/68/test_set.tar.gz) |
|   WenetSpeech    | 普通话 | 10000h |     315G      |                 [下载教程](./wenetspeech.md)                 |

### 2.4 数据增强

通过在原始音频中添加小的随机扰动（标签不变转换）获得新音频来增强的语音数据。在训练模型的每个epoch中随机合成音频。

目前提供五个可选的增强组件供选择，配置并插入处理过程。

- 噪声干扰（需要背景噪音的音频文件）
- 随机采样率增强
- 速度扰动
- 移动扰动
- 音量扰动
- SpecAugment增强方式
- SpecSubAugment增强方式

为了让训练模块知道需要哪些增强组件以及它们的处理顺序，需要事先准备一个JSON格式的*扩展配置文件*。例如：

```yaml
[
  {
    "type": "noise",
    "aug_type": "audio",
    "params": {
      "min_snr_dB": 10,
      "max_snr_dB": 50,
      "repetition": 2,
      "noise_manifest_path": "dataset/manifest.noise"
    },
    "prob": 0.5
  },
  {
  	### 重采样
    "type": "resample",
    "aug_type": "audio",
    "params": {
      "new_sample_rate": [8000, 32000, 44100, 48000]
    },
    "prob": 0.0
  },
  {
    "type": "speed",
    "aug_type": "audio",
    "params": {
      "min_speed_rate": 0.9,
      "max_speed_rate": 1.1,
      "num_rates": 3
    },
    "prob": 1.0
  },
  {
    "type": "shift",
    "aug_type": "audio",
    "params": {
      "min_shift_ms": -5,
      "max_shift_ms": 5
    },
    "prob": 1.0
  },
  {
    "type": "volume",
    "aug_type": "audio",
    "params": {
      "min_gain_dBFS": -15,
      "max_gain_dBFS": 15
    },
    "prob": 1.0
  },
  {
    "type": "specaug",
    "aug_type": "feature",
    "params": {
      "inplace": true,
      "max_time_warp": 5,
      "max_t_ratio": 0.05,
      "n_freq_masks": 2,
      "max_f_ratio": 0.15,
      "n_time_masks": 2,
      "replace_with_zero": false
    },
    "prob": 1.0
  },
  {
    "type": "specsub",
    "aug_type": "feature",
    "params": {
      "max_t": 30,
      "num_t_sub": 3
    },
    "prob": 1.0
  }
]
```

- 设置位置：`train.py`的`--augment_conf_file`参数被设置为上述示例配置文件的路径时，每个epoch中的每个音频片段都将被处理。
- 首先，均匀随机采样速率会有50％的概率在 0.95 和 1.05之间对音频片段进行速度扰动。然后，音频片段有 50％ 的概率在时间上被挪移，挪移偏差值是 -5 毫秒和 5 毫秒之间的随机采样。最后，这个新合成的音频片段将被传送给特征提取器，以用于接下来的训练。
- 使用数据增强技术时要小心，由于扩大了训练和测试集的差异，不恰当的增强会对训练模型不利，导致训练和预测的差距增大。

### 2.5 合成语音数据

1. 为了拟补数据集的不足，我们合成一批语音用于训练，语音合成一批音频文件。首先安装PaddleSpeech，执行下面命令即可安装完成。

```shell
python -m pip install paddlespeech
```

2. 然后下载一个语料，如果开发者有其他更好的语料也可以替换。然后解压`dgk_lost_conv/results`目录下的压缩文件，windows用户可以手动解压。

```shell
cd tools/generate_audio
git clone https://github.com/aceimnorstuvwxz/dgk_lost_conv.git
cd dgk_lost_conv/results
unzip dgk_shooter_z.conv.zip
unzip xiaohuangji50w_fenciA.conv.zip
unzip xiaohuangji50w_nofenci.conv.zip
```

3. 接着执行下面命令生成中文语料数据集，生成的中文语料存放在`tools/generate_audio/corpus.txt`。

```shell
cd tools/generate_audio/
python generate_corpus.py
```

4. 最后执行以下命令即可自动合成语音，合成时会随机获取说话人进行合成语音，合成的语音会放在`dataset/audio/generate`， 标注文件会放在`dataset/annotation/generate.txt`。

```shell
cd tools/generate_audio/
python generate_audio.py
```

## 3. 训练测试

### 3.1 训练

 - 执行`create_data.py`程序，执行完成之后检查是否在`dataset`目录下生成了`manifest.test`、`manifest.train`、`mean_istd.json`、`vocabulary.txt`这四个文件，并确定里面已经包含数据。
 - 执行训练脚本，开始训练语音识别模型，详细参数请查看`configs`下的配置文件。每训练一轮和每10000个batch都会保存一次模型，模型保存在`models/<use_model>_<feature_method>/epoch_*/`目录下。
 - 默认会使用数据增强训练，如何不想使用数据增强，只需要将参数`augment_conf_path`设置为`None`即可。
 - 如果没有关闭测试，在每一轮训练结果之后，都会执行一次测试计算模型在测试集的准确率
 - 注意为了加快训练速度，训练只能用贪心解码。
 - 如果模型文件夹下包含`last_model`文件夹，在训练的时候会自动加载里面的模型，这是为了方便中断训练的之后继续训练，无需手动指定，如果手动指定了`resume_model`参数，则以`resume_model`指定的路径优先加载。如果不是原来的数据集或者模型结构，需要删除`last_model`这个文件夹。

```python
# 单机单卡训练
CUDA_VISIBLE_DEVICES=0 python train.py
```

```python
# 单机多卡训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
```

多机多卡的启动方式：

 - `--nproc_per_node=2`：表示在一个node上启动2个进程
 - `--nnodes=2`：表示一共有2个node进行分布式训练
 - `--node_rank=0`：当前node的id为0
 - `--master_addr="192.168.4.7"`：主节点的地址
 - `--master_port=8081`：主节点的port
 - `train.py`：训练代码

```python
# 第一台服务器
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.4.7" --master_port=8081 train.py

# 第二台服务器
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr="192.168.4.7" --master_port=8081 train.py
```

在训练过程中，程序会使用VisualDL记录训练结果，可以通过在根目录执行以下命令启动VisualDL。

```python
visualdl --logdir=log --host=0.0.0.0
```

### 3.2 测试

```shell
python eval.py --resume_model=models/conformer_online_fbank/best_model/
```

### 3.3 推理

#### 3.3.1 本地模型预测

```shell
python infer_path.py --wav_path=./dataset/test.wav
```

#### 3.3.2 长语音预测

通过参数`--is_long_audio`可以指定使用长语音识别方式，这种方式通过VAD分割音频，再对短音频进行识别，拼接结果，最终得到长语音识别结果。

```shell script
python infer_path.py --wav_path=./dataset/test_long.wav --is_long_audio=True
```

#### 3.3.3 模拟实时识别

在`--real_time_demo`指定为True。

```shell
python infer_path.py --wav_path=./dataset/test.wav --real_time_demo=True
```

#### 3.3.4 Web部署

在服务器执行下面命令通过创建一个Web服务，通过提供HTTP接口来实现语音识别。启动服务之后，如果在本地运行的话，在浏览器上访问`http://localhost:5000`，否则修改为对应的 IP地址。打开页面之后可以选择上传长音或者短语音音频文件，也可以在页面上直接录音，录音完成之后点击上传，播放功能只支持录音的音频。支持中文数字转阿拉伯数字，将参数`--is_itn`设置为True即可，默认为False。

```shell
python infer_server.py
```

#### 3.3.5 GUI界面部署

通过打开页面，在页面上选择长语音或者短语音进行识别，也支持录音识别实时识别，带播放音频功能。该程序可以在本地识别，也可以通过指定服务器调用服务器的API进行识别。

```shell script
python infer_gui.py
```

## 4. 模型导出

训练保存的或者下载作者提供的模型都是模型参数，我们要将它导出为预测模型，这样可以直接使用模型，不再需要模型结构代码，同时使用Inference接口可以加速预测，详细参数请查看该程序。

```shell
python export_model.py --resume_model=models/conformer_online_fbank/best_model/
```

## 5.标点符号

在语音识别中，模型输出的结果只是单纯的文本结果，并没有根据语法添加标点符号，本教程就是针对这种情况，在语音识别文本中根据语法情况加入标点符号，使得语音识别系统能够输出在标点符号的最终结果。

### 5.1 使用

使用主要分为三4步：

1. 首先是[下载五个标点的模型](https://download.csdn.net/download/qq_33200967/75664996)或者[下载三个标点符号的模型](https://download.csdn.net/download/qq_33200967/86539773)，并解压到`models/`目录下，注意这个模型只支持中文，如果想自己训练模型的话，可以在[PunctuationModel](https://github.com/yeyupiaoling/PunctuationModel)训练模型，然后导出模型复制到`models/`目录。


2. 需要使用PaddleNLP工具，所以需要提前安装PaddleNLP，安装命令如下：

```shell
python -m pip install paddlenlp -i https://mirrors.aliyun.com/pypi/simple/ -U
```

3. 在使用时，将`use_pun`参数设置为True，输出的结果就自动加上了标点符号，如下。

```
消耗时间：101, 识别结果: 近几年，不但我用输给女儿压岁，也劝说亲朋，不要给女儿压岁钱，而改送压岁书。, 得分: 94
```

### 5.2 单独使用标点符号模型

如果只是使用标点符号模型的话，可以参考一下代码。

```python
from masr.infer_utils.pun_predictor import PunctuationPredictor

pun_predictor = PunctuationPredictor(model_dir='models/pun_models')
result = pun_predictor('近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书')
print(result)
```

输出结果：

```
[2022-01-13 15:27:11,194] [    INFO] - Found C:\Users\test\.paddlenlp\models\ernie-1.0\vocab.txt
近几年，不但我用书给女儿儿压岁，也劝说亲朋，不要给女儿压岁钱，而改送压岁书。
```

## 6. 集束搜索解码

本项目目前支持两种解码方法，分别是集束搜索(ctc_beam_search)和贪婪策略(ctc_greedy)，如果要使用集束搜索方法，首先要安装`paddlespeech_ctcdecoders`库，执行以下命令即可安装完成。

```shell
python -m pip install paddlespeech_ctcdecoders -U -i https://ppasr.yeyupiaoling.cn/pypi/simple/
```

### 6.1 语言模型

集束搜索解码需要使用到语言模型，在执行程序的时候，回自动下载语言模型，不过下载的是小语言模型，如何有足够大性能的机器，可以手动下载70G的超大语言模型，点击下载[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) ，并指定语言模型的路径。

注意，上面提到的语言模型都是中文语言模型，如果需要使用英文语言模型，需要手动下载，并指定语言模型路径。

```shell
https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm
```

### 6.2 寻找最优的alpha和beta

这一步可以跳过，使用默认的alpha和beta也是不错的，如果想精益求精，可以执行下面的命令，可能速度会比较慢。执行完成之后会得到效果最好的alpha和beta参数值。

```shell
python tools/tune.py --resume_model=models/deepspeech2/epoch_50
```

### 6.3 使用集束搜索解码

在需要使用到解码器的程序，如评估，预测，在`configs/config_zh.yml`配置文件中修改参数`decoder`为`ctc_beam_search`即可，如果alpha和beta参数值有改动，修改对应的值即可。

### 6.4 语言模型表格

|                           语言模型                           |                           训练数据                           | 数据量  | 文件大小 |                     说明                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :-----: | :------: | :------------------------------------------: |
| [自定义中文语言模型](https://pan.baidu.com/s/1vdQsqnoKHO9jdFU_1If49g?pwd=ea09) | [自定义中文语料](https://download.csdn.net/download/qq_33200967/87002687) | 约2千万 |  572 MB  |            训练参数`-o 5`，无剪枝            |
| [英文语言模型](https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm) | [CommonCrawl](http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.00.deduped.xz) | 18.5亿  |  8.3 GB  | 训练参数`-o 5`，剪枝参数`'--prune 0 1 1 1 1` |
| [中文语言模型（剪枝）](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm) |                        百度内部语料库                        |  1.3亿  |  2.8 GB  | 训练参数`-o 5`，剪枝参数`'--prune 0 1 1 1 1` |
| [中文语言模型](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) |                        百度内部语料库                        |  37亿   | 70.4 GB  |            训练参数`-o 5`，无剪枝            |

### 6.5 训练自己的语言模型

1. 首先安装kenlm，此步骤要从项目根目录开始，以下是Ubuntu的安装方式，其他系统请自行百度。

```shell
sudo apt install -y libbz2-dev liblzma-dev cmake build-essential libboost-all-dev
cd tools/
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
cd kenlm
mkdir -p build
cd build
cmake ..
make -j 4
```

2. 准备kenlm语料，此步骤要从项目根目录开始，使用的语料是训练数据集，所以要执行`create_data.py`完成之后才能执行下面操作。或者自己准备语料，修改生成语料的代码。

```shell
cd tools/
python create_kenlm_corpus.py
```

3. 有了kenlm语料之后，就可以训练kenlm模型了，此步骤要从项目根目录开始，执行下面命令训练和压缩模型。

```shell
cd tools/kenlm/build/ 
bin/lmplz -o 5 --verbose header --text ../../../dataset/corpus.txt --arpa ../../../lm/my_lm.arpa
# 把模型转为二进制，减小模型大小
bin/build_binary trie -a 22 -q 8 -b 8 ../../../lm/my_lm.arpa ../../../lm/my_lm.klm
```

4. 可以使用下面代码测试模型是否有效。

```python
import kenlm

model = kenlm.Model('kenlm1/build/model/test.klm')
result = model.score('近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书', bos=True, eos=True)
print(result)
```

## 7. 活动检测（VAD）

### 7.1 语音活动检测（VAD）

针对长语音识别，通过这个预测器可以对长语音进行分割。通过检测静音的位置，把长语音分割成多段短语音。然后把分割后的音频通过短语音识别的方式来实现来进行识别。

这个语音活动检测预测器是使用onnxruntime推理的，所以在使用的前提要按照这个库。

```shell
python -m pip install onnxruntime
```

在本项目的使用可以参考`infer_path.py`的长语音识别，相关文档在[本地预测](./infer.md)。

如果想要单独使用语音活动检测的话，可以参考一下代码，注意输入的数据`wav`是`np.float32`的，因为输入的音频采样率只能是8K或者16K。

```python
import numpy as np
import soundfile

from masr.infer_utils.vad_predictor import VADPredictor

vad_predictor = VADPredictor()

wav, sr = soundfile.read('dataset/test_long.wav', dtype=np.float32)
speech_timestamps = vad_predictor.get_speech_timestamps(wav, sr)
for t in speech_timestamps:
    crop_wav = wav[t['start']: t['end']]
    print(crop_wav.shape)
```

### 7.2 流式实时语音活动检测（VAD）

最新版本可以支持流式检测语音活动，在录音的时候可以试试检测是否停止说话，从而完成一些业务，如停止录音开始识别等。

```python
import numpy as np
import soundfile

from ppasr.infer_utils.vad_predictor import VADPredictor

vad = VADPredictor()

wav, sr = soundfile.read('dataset/test.wav', dtype=np.float32)

for i in range(0, len(wav), vad.window_size_samples):
    chunk_wav = wav[i: i + vad.window_size_samples]
    speech_dict = vad.stream_vad(chunk_wav, sampling_rate=sr)
    if speech_dict:
        print(speech_dict, end=' ')
```

实时输出检测结果：

```
{'start': 11296} {'end': 21984} {'start': 25632} {'end': 54752} {'start': 57376} {'end': 97760} {'start': 103456} {'end': 124896} 
```

## 8. 参考地址

