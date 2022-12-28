1. 创建虚拟环境并激活
```shell
conda create -n dt2 python=3.8
conda activate dt2
```
2. 安装 detecteon2 的依赖包
```shell
# pytorch cuda 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
# opencv
pip install opencv-python
# ninja
conda install -c conda-forge ninja
# pytorch 和 torchvision的 版本需要和 CUDA 相匹配，如果需要其他版本的 Pytorch，请到下面的网页找对应的安装包
```
