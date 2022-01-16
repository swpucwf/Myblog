# 斑点检测

## 绪论

### 斑点

- 什么是斑点？斑点通常是指**与周围有着颜色和灰度差别的区域**。
- 在计算机视觉中，斑点检测是指在**数字图像中找出和周围区域特性不同的区域**，这些特性包括**光照或颜色**等。
- 一般图像中斑点区域的像素特性相似甚至相同，**某种程度而言，斑点块中所有点是相似的**。

### 检测方法

1. 基于求导的微分方法，这类的方法称为微分检测器；即使用滤波核与图像进行卷积，找出与滤波核具有相同特征的区域，例如LoG 算法，DoH算法，DOG算法
2. 基于局部极值的分水岭算法。例如opencv 中的 SimpleBlobDetector

## 边缘检测

边缘一般分为两种：屋脊型边缘和**阶跃型边缘**。在实际应用中，一般只考虑阶跃边缘，因为只要采样足够或者说窗口足够小，屋脊型边缘也可以看做是阶跃边缘。

![image-20220108155358979](%E6%96%91%E7%82%B9%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95.assets/image-20220108155358979.png)



- 边缘的检测可以通过一阶导数、二阶导数计算。对于阶跃型边缘可以使用一阶导数的极值来判断
- 因此梯度算子的定义： $G(x, y)=\sqrt{\Delta_{x} f(x, y)^{2}+\Delta_{y} f(x, y)^{2}}$
- 为了简化计算，一般将梯度算子简化为：$G(x, y)=\left|\Delta_{x} f(x, y)\right|+\left|\Delta_{y} f(x, y)\right|$

### 一阶算子

#### Robert算子

- 被用到了图像增强中的锐化，原因是作为一阶微分算子，Robert简单，计算量小，对细节反应敏感。
- 与标准一阶差分不同，Robert采用对角线差分。
- Roberts边缘检测算子是一种利用**局部差分**算子寻找边缘的算子,Robert算子图像处理后结果边缘不是很平滑。
- 计算公式：$R(x, y)=\max \{|f(x, y)-f(x+1, y+1)|,|f(i+1, j)-f(i-1, j+1)|\}$

```python
from skimage import data,filters
import matplotlib.pyplot as plt
from skimage import io
edges = filters.roberts(img)
```

#### Sobel算子

- 邻域的像素对当前像素产生的影响不是等价的，所以距离不同的像素具有不同的权值，对算子结果产生的影响也不同。
- 一般来说，距离越远，产生的影响越小。
- 该算子其实是经过了高斯平滑后计算差分。

![preview](%E6%96%91%E7%82%B9%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95.assets/v2-21b4515e0fb5f09ba5c0b66f7ca684e2_r-16417169753504.jpg)

```python
from skimage import data,filters
import matplotlib.pyplot as plt
from skimage import io
edges = filters.sobel(img)
```

#### Prewitt 算子

- Prewitt算子是一种一阶微分算子的边缘检测

- 利用像素点上下、左右邻点的灰度差，在边缘处达到极值检测边缘，去掉部分伪边缘，对噪声具有平滑作用 。

  ![preview](%E6%96%91%E7%82%B9%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95.assets/v2-9401364220e4e3b704cc5e5d0108280c_r-16417169690633.jpg)

### 二阶算子

#### 拉普拉斯算子

- 二维的laplace算子：$\nabla=\Delta^{2}=\left[\frac{\partial}{\partial x} \frac{\partial}{\partial y}\right]\left[\frac{\partial}{\partial x} \frac{\partial}{\partial y}\right]^{T}=\frac{\partial^{2}}{\partial x}+\frac{\partial^{2}}{\partial y}$

- 二阶laplace算子的卷积模板：

  ![image-20220109165938054](%E6%96%91%E7%82%B9%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95.assets/image-20220109165938054.png)

- 考虑到四个方向，则卷积模板为：

  ![image-20220109170001549](%E6%96%91%E7%82%B9%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95.assets/image-20220109170001549.png)

```python
from skimage import data,filters
import matplotlib.pyplot as plt
from skimage import io
edges = filters.laplace(img)
```

#### **高斯拉普拉斯算子**(LOG)

![image-20220109170127191](%E6%96%91%E7%82%B9%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95.assets/image-20220109170127191.png)

#### 高斯差分算子(DOG)

### 边缘（高级特征）检测

#### canny

```python
from skimage import feature
import matplotlib.pyplot as plt
from skimage import io

# img = data.camera()
img = io.imread(r'000001.jpg', as_gray=True)
edges = feature.canny(img)
plt.imshow(edges, plt.cm.gray)
plt.show()
```

#### Gabor

```python
from skimage import io
from skimage import filters
import matplotlib.pyplot as plt

img = io.imread(r'01.jpg', as_gray=True)
filt_real, filt_imag = filters.gabor(img, frequency=0.6)
plt.figure('gabor', figsize=(8, 8))
plt.subplot(121)
plt.title('filt_real')
plt.imshow(filt_real, plt.cm.gray)
plt.subplot(122)
plt.title('filt-imag')
plt.imshow(filt_imag, plt.cm.gray)
plt.show()
```



## 斑点检测

### 计算流程(LOG)

1. 我们计算出在半径范围内的不同的尺度因子σ（即高斯核方差）的高斯核的拉普拉斯算子
2. 在图片上进行不同尺度因子LoG
3. 然后在尺度空间和图像空间上检测都是极值的点，确定为斑点的中心。

### DOG

1. 首先使用不同尺度的高斯算子对图像进行平滑
2. 其次计算相邻尺度下平滑图像的差分图像(DoG空间)
3. 最后在DoG空间寻找极值点

### Hessian（DoH）

- Determinant of Hessian，这是最快的方法。

- 它通过在图像的Hessian行列式矩阵中查找最大值来检测斑点。

- 检测速度与blob的大小无关，因为内部实现使用盒式过滤器而不是卷积。

- 基本方法流程和思路与LoG一样，区别不同的是使用了Hessian矩阵：
  $$
  H=\left[\begin{array}{ll}
  \frac{\partial^{2}}{\partial x^{2}} & \frac{\partial^{2}}{\partial x y} \\
  \frac{\partial^{2}}{\partial x y} & \frac{\partial^{2}}{\partial y^{2}}
  \end{array}\right]
  $$



```python
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt

#
# image = data.hubble_deep_field()[0:500, 0:500]
# image_gray = rgb2gray(image)

image = cv2.imread( r'D:\project\SLIC\gangue\Guobei_g_0008739.jpg')
# image = cv2.imread(r'D:\project\SLIC\coal\Guobei_c_0000001.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.5)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray, max_sigma=50, threshold=.1)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image, interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
plt.show()
```



### SimpleBlobDetector斑点检测

1. 、对一张图片,设定一个低阈值,设定一个高阈值,在设定一个阈值步进,然后从低阈值到高阈值按照阈值步进取一系列的阈值,即对[minThreshold,maxThreshold)区间，以thresholdStep为间隔，用每一个阈值对图像进行二值化，得到一系列图像；
2. 对每张二值图片，使用查找这些图像的边,并计算每一个轮廓的中心
3. 根据2得到每一个图片的轮廓的中心点，全部放在一起。定义一个最小距离,在这个距离区域内的特征中心点［由theminDistBetweenBlobs控制多少才算接近］被归为一个group,对应一个斑点特征,得到特征点集合。
4. 从3得到的那些点,估计最后的斑点特征和相应半径。
5. 对特征点进行相应的过滤,例如颜色过滤，面积过滤，圆度等。

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt


plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签

img0 = cv2.imread(r'D:\project\SLIC\coal\Guobei_c_0000001.jpg')
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gray, (9,9), 0) #高斯模糊，X,Y 方向的Ksiez分别为9和9

params = cv2.SimpleBlobDetector_Params()
#斑点检测的可选参数
#params.minThreshold= 10 #亮度最小阈值控制
#params.maxThreshold = 255 #亮度最大阈值控制
params.thresholdStep = 9 #亮度阈值的步长控制，越小检测出来的斑点越多

params.filterByColor = True #颜色控制
params.blobColor = 0 #只检测黑色斑点
#params.blobColor = 255 #只检测白色色斑点

params.filterByArea = True #像素面积大小控制
params.minArea = 20
#params.maxArea=2000

#params.filterByCircularity = True #圆度控制，圆度的定义是（4π×面积）/（周长的平方）
#params.minCircularity = 0.3

#params.filterByConvexity =True #凸度控制，凸性的定义是（斑点的面积/斑点凸包的面积
#params.minConvexity = 1.0

#params.filterByInertia = True# 惯性率控制
#params.minInertiaRatio = 0.2#圆形的惯性率等于1，惯性率越接近1，圆度越高
detector = cv2.SimpleBlobDetector_create(params)#创建斑点检测器
keypoints = detector.detect(gauss) #在哪个图上检测斑点
print("共检测出%d个斑点" % len(keypoints))

#在原图上画出检测到的斑点
im_with_keypoints=cv2.drawKeypoints(img0, keypoints, np.array([]), (0, 0, 255),
                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print("斑点中心坐标为：")
for (x, y) in keypoints[0].convert(keypoints):
    print(x, ",", y)
    cv2.circle(im_with_keypoints, (x, y), 1, (255, 255, 255), 1) #以白色标记处斑点中心（以斑点中心为中心画圆）

#绘出检测结果图
plt.subplot(1,1,1)
plt.imshow(cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title("OpenCV 斑点检测\n之小煤块", fontSize=16, color="b")
plt.show()
```

