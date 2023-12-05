# Ubuntu 编译 Opencv

## 1. OpenCV

### 1.1 build opencv 

```shell
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
sudo apt install -y g++ cmake make git libgtk2.0-dev pkg-config
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
# Create build directory
mkdir -p build && cd build
# Configure
cmake  ../opencv-4.x
# Build
cmake --build .
```

## 1.2 Build with opencv_contrib

```shell
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
sudo apt install -y g++ cmake make git libgtk2.0-dev pkg-config
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
# Create build directory and switch into it
mkdir -p build && cd build
# Configure
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
# Build
cmake --build .
```

```shell
sudo make install
```

## 2.demo示例

```
mkdir opencv480_test
vim CMakeLists.txt
```

写入以下内容

```c++
cmake_minimum_required(VERSION 2.8)
project( OpenCV480_Test )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( OpenCV480_Test helloworld.cpp )
target_link_libraries( OpenCV480_Test ${OpenCV_LIBS} )
```

编写文件

```shell
vim helloworld.cpp
```

写入以下内容

```c++

#include <opencv2/opencv.hpp> 
#include <stdio.h> 
using namespace cv; 
int main(int argc, char** argv) 
{ 
    Mat image = imread("lena.jpg"); 
    if (image.empty()) { 
        printf("No image data \n"); 
        return -1; 
    } 
    namedWindow("OpenCV480_Test", WINDOW_AUTOSIZE); 
    imshow("OpenCV480_Test", image); 
    waitKey(0); 
    return 0; 
}
```

```shell
cmake .

./ OpenCV480_Test
```



##参考

[OpenCV: Installation in Linux](https://docs.opencv.org/4.8.0/d7/d9f/tutorial_linux_install.html)

[Ubuntu系统下编译OpenCV4.8源码记录](https://mp.weixin.qq.com/s/ewllcifk-1xA-SnX1zl09Q)