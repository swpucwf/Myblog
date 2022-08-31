# OpenCV DNN模块中使用NVIDIA GPU加速--(基于Windows）

### **一、 准备环境**

1. **下载安装Visual Studio**

从https://visualstudio.microsoft.com/downloads/下载并安装 Visual Studio 。运行安装程序，选择使用 C++ 进行桌面开发，然后单击安装。

2.  安装Python环境，可以单独安装，也可以用Anaconda；
3. 安装cmake

CMake下载地址：https://cmake.org/download/.

4. 安装Git(本文使用2.30.1),下载Windows版本：

Git下载地址： https://git-scm.com/downloads/. 

5. CUDA下载安装

从https://developer.nvidia.com/cuda-downloads下载并安装最新版本的 CUDA 。您还可以从https://developer.nvidia.com/cuda-toolkit-archive获取存档的 CUDA 版本。

按照https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html上的 Windows CUDA 安装指南进行操作。

CUDA下载安装：

在https://developer.nvidia.com/cudnn-download-survey上回答 cuDNN 调查并下载您喜欢的 cuDNN 版本。

按照https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows上的 Windows cuDNN 安装指南进行操作。在这篇文章中，我们使用了 cuDNN 11.2，但您也可以使用其他 cuDNN 版本。

### 获取opencv源码

```shell
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout tags/4.5.0
cd ../
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout tags/4.5.0
```

### **使用 CUDA 支持构建 OpenCV**

第一步是使用 CMake 配置 OpenCV 构建。我们将几个选项传递给 CMake CLI。这些是：

- -G：它指定用于构建的 Visual Studio 编译器
- -T：指定主机工具架构
- CMAKE_BUILD_TYPE：它指定RELEASE或DEBUG安装模式
- CMAKE_INSTALL_PREFIX：指定安装目录
- OPENCV_EXTRA_MODULES_PATH：设置为 opencv_contrib 模块的位置
- PYTHON_EXECUTABLE：设置为 python3 可执行文件，用于构建。
- PYTHON3_LIBRARY：它指向 python3 库。
- WITH_CUDA：使用 CUDA 构建 OpenCV
- WITH_CUDNN：使用 cuDNN 构建 OpenCV
- OPENCV_DNN_CUDA：启用此项以构建具有 CUDA 支持的 DNN 模块
- WITH_CUBLAS：启用优化。

此外，还有两个优化标志ENABLE_FAST_MATH和CUDA_FAST_MATH，用于优化和加速数学运算。但是，当您启用这些标志时，不能保证浮点计算的结果符合 IEEE。如果您想要快速计算并且精度不是问题，您可以继续使用这些选项。此链接详细解释了准确性问题。

如果 CMake 可以找到安装在您的系统上的 CUDA 和 cuDNN，您应该会看到此输出。