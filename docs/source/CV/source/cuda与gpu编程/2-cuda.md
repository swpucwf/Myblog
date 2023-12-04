# CUDA编程入门

- .cu代表cuda程序

- <<<>>> ： nvcc 特有语法 
- cudnn 需要与tensorRT版本对应

nvcc 选择显卡版本 ，编译需要选择的参数

```shell
# GeForce RTX 3070, 3080, 3090 Tesla GA10x cards, RTX Ampere – RTX 3080, GA102 – RTX 3090, RTX A2000, A3000, RTX A4000, A5000, A6000, NVIDIA A40, GA106 – RTX 3060, GA104 – RTX 3070, GA107 – RTX 3050, RTX A10, RTX A16, RTX A40, A2 Tensor Core GPU
ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]

# Kepler GeForce GTX 770, GTX 760, GT 740
# ARCH= -gencode arch=compute_30,code=sm_30

# Tesla A100 (GA100), DGX-A100, RTX 3080
# ARCH= -gencode arch=compute_80,code=[sm_80,compute_80]

# Tesla V100
# ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]

# GeForce RTX 2080 Ti, RTX 2080, RTX 2070, Quadro RTX 8000, Quadro RTX 6000, Quadro RTX 5000, Tesla T4, XNOR Tensor Cores
# ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]

# Jetson XAVIER
# ARCH= -gencode arch=compute_72,code=[sm_72,compute_72]

# GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

# GP100/Tesla P100 - DGX-1
# ARCH= -gencode arch=compute_60,code=sm_60

# For Jetson TX1, Tegra X1, DRIVE CX, DRIVE PX - uncomment:
# ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

# For Jetson Tx2 or Drive-PX2 uncomment:
# ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]

# For Tesla GA10x cards, RTX 3090, RTX 3080, RTX 3070, RTX A6000, RTX A40 uncomment:
# ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]

```



grid  和 block

- 属于逻辑意义上的东西
- 启动一个kernel时候需要指定grid和block
- 

## 2.1 理解.cu和.cpp的相互引用及Makefile

```txt
CONFIG := ../../config/Makefile.config：定义一个变量CONFIG，指向配置文件的路径。
CONFIG_LOCAL := ./config/Makefile.config：定义一个变量CONFIG_LOCAL，指向本地的配置文件路径。
include $(CONFIG)：包含上面定义的CONFIG变量对应的Makefile。
include $(CONFIG_LOCAL)：包含上面定义的CONFIG_LOCAL变量对应的Makefile。
BUILD_PATH := build：定义一个变量BUILD_PATH，指向构建路径。
SRC_PATH := src：定义一个变量SRC_PATH，指向源代码路径。
CUDA_DIR := /usr/local/cuda-$(CUDA_VER)：定义一个变量CUDA_DIR，指向CUDA的安装路径，其中$(CUDA_VER)是一个环境变量，表示CUDA的版本。
KERNELS_SRC := $(wildcard $(SRC_PATH)/*.cu)：使用通配符规则匹配源代码路径下的所有.cu文件，并将它们赋值给变量KERNELS_SRC。
APP_OBJS += $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))：使用模式替换将所有在源代码路径下的.cu文件转换为.cu.o，并将它们放到构建路径下。这些.cu.o文件是CUDA源代码编译后生成的中间文件。
APP_DEPS += $(KERNELS_SRC)：将所有CUDA源代码文件添加到依赖列表中，这样当其中任何一个文件改变时，Makefile都会重新编译相关的对象文件。
CUCC := $(CUDA_DIR)/bin/nvcc：定义一个变量CUCC，指向NVIDIA的CUDA编译器。
CUDAFLAGS := -Xcompiler -fPIC：定义一个变量CUDAFLAGS，用于传递给CUDA编译器的标志。这里使用了-Xcompiler -fPIC，表示生成位置无关的代码。
INCS := -I $(CUDA_DIR)/include \：定义一个变量INCS，用于传递给编译器的包含目录标志。这里包含了CUDA的包含目录。
INCS += -I $(SRC_PATH)：添加源代码路径到包含目录列表。
LIBS := -L "$(CUDA_DIR)/lib64"：定义一个变量LIBS，用于传递给链接器的库目录标志。这里包含了CUDA的库目录。
LIBS += \：添加一个空行，使接下来的命令在新的行上执行。
ifeq ($(DEBUG),1)：判断变量DEBUG是否等于1。如果是，执行接下来的命令；如果不是，跳过接下来的命令。
CUDAFLAGS += -g -G -O0：如果变量DEBUG等于1，则向CUDAFLAGS添加编译选项-g -G -O0，表示启用调试信息、禁用优化并将编译器设置为生成较低级别的代码。
```

