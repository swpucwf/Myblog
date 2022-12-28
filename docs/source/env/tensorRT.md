# windows上安装Tensorrt（python）

1. 下载TensorRT[TensorRT下载链接](https://developer.nvidia.com/nvidia-tensorrt-download)

- 版本 需要选取在8.2以后的，支持python

2. 在路径下安装对应python版本的tensorrt,例如我是3.8

```shell
pip install E:\TensorRT-8.2.1.8\python\tensorrt-8.2.1.8-cp37-none-win_amd64.whl
```

![image-20221216223149952](C:/Users/CWF/AppData/Roaming/Typora/typora-user-images/image-20221216223149952.png)

3. 添加系统变量

```txt
1. TensorRT-8.4.1.5中include复制到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1中，
2. TensorRT-8.4.1.5中lib中的dll文件复制到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin中
3. TensorRT-8.4.1.5中lib中lib的文件复制到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\lib中
```

![image-20221216224007151](C:/Users/CWF/AppData/Roaming/Typora/typora-user-images/image-20221216224007151.png)

4. 下载一个zlib包，解压缩后找到zlibwapi.dll文件，剪切到C:\Windows\System32位置下面（这是cudnn依赖的动态链接库）[链接](http://www.winimage.com/zLibDll/zlib123dllx64.zip)

5. 安装pycuda

[链接](https://www.lfd.uci.edu/~gohlke/pythonlibs/?cm_mc_uid=08085305845514542921829&cm_mc_sid_50200000=1456395916&cm_mc_uid=08085305845514542921829&cm_mc_sid_50200000=1456395916#pycuda)

激活虚拟环境
pip install  pycuda