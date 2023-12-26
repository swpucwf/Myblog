# 编程与入门

## 1.cmake 编译规则

### 1.1基本规则
```cmake
cmake_minimum_required(VERSION 3.10)
project(demo)
```
### 1.1.1 include指令
执行 include 指令时，它会把指定的文件内容读入当前的Makefile环境中。
```cmake
CONFIG        :=  ../../config/Makefile.config
CONFIG_LOCAL  :=  ./config/Makefile.config


# 执行 include 指令时，它会把指定的文件内容读入当前的Makefile环境中。
include $(CONFIG)
include $(CONFIG_LOCAL)
```
### 1.1.2 wildcard
wildcard 指令可以搜索指定目录下的所有文件，并将搜索到的文件名保存到变量中。
```cmake
KERNELS_SRC   :=  $(wildcard $(SRC_PATH)/*.cu)
```
### 1.1.3 patsubst
patsubst 指令可以搜索指定目录下的所有文件，并将搜索到的文件名保存到变量中。
```cmake
APP_OBJS      +=  $(patsubst $(SRC_PATH)%, $(BUILD_PATH)%, $(KERNELS_SRC:.cu=.cu.o))  
APP_DEPS      +=  $(KERNELS_SRC)
```
其中的%是通配符，表示任意字符串。KERNELS_SRC:.cu=.cu.o代表将KERNELS_SRC中的.cu替换为.cu.o。
### 1.1.4 编译器选项
- Xcompiler：这个选项允许您传递参数给C/C++编译器。
- fPIC：这是一个编译选项，表示生成位置无关的代码（Position Independent Code）。这意味着生成的代码可以在任何地址执行，而不仅仅是加载到固定的内存地址。这在创建共享库时很有用，因为共享库需要在多个进程或系统中共享。
```cmake
CUDAFLAGS     :=  -Xcompiler -fPIC 
```
### 1.1.5 调试选项
- g：生成用于调试的信息。
- G：生成图形调试信息，通常与NVIDIA的Nsight工具一起使用。
- O0：禁用所有优化，这有助于在调试时获得更一致的代码行为
- Wall：显示所有警告。
- Wunused-function：警告未使用的函数。
- Wunused-variable：警告未使用的变量。
- Wfatal-errors：将某些错误视为致命错误。
###  $@
Makefile中的自动变量，代表目标文件名。
