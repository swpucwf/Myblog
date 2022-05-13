# YOLO v3

- 论文地址：https://pjreddie.com/media/files/papers/[YOLOv3](https://so.csdn.net/so/search?q=YOLOv3&spm=1001.2101.3001.7020).pdf
- 论文：YOLOv3: An Incremental Improvement

![image-20220202194620782](../images/yoloV3/image-20220202194620782.png)

![img](https://upload-images.jianshu.io/upload_images/18623053-237c7d6882c92cef.jpg)

### 先验框

 **(10×13)，(16×30)，(33×23)，(30×61)，(62×45)，(59× 119)， (116 × 90)， (156 × 198)，(373 × 326)** ，顺序为**w × h**

1. Yolov3中，只有卷积层，通过调节卷积步长控制输出特征图的尺寸。所以对于输入图片尺寸没 有特别限制。
2. Yolov3借鉴了**金字塔特征图**思想，**小尺寸特征图用于检测大尺寸物体，而大尺寸特征图检测小 尺寸物体**。特征图的输出维度为 $N \times N \times[3 \times(4+1+80)], N \times N$ 为输出特征 图格点数，一共3个Anchor框，每个框有 4 维预测框数值 $t_{x}, t_{y}, t_{w}, t_{h} ， 1$ 维预测框置信度， 80 维物体类别数。所以第一层特征图的输出维度为 $8 \times 8 \times 255$ 。
3. **多尺度输出**：Yolov3总共输出3个特征图，第一个特征图下采样**32**倍，第二个特征图下采样16倍，第三个下 采样**8**倍。输入图像经过Darknet-53 (无全连接层)，再经过Yoloblock生成的特征图被当作两 用，第一用为经过 $3^{*} 3$ 卷积层、 $1^{*} 1$ 卷积之后生成特征图一，第二用为经过 $1^{*} 1$ 卷积层加上采样层，与Darnet-53网络的中间层输出结果进行拼接，产生特征图二。同样的循环之后产生特征图。
4. **concat操作与加和操作的区别**：加和操作来源于ResNet思想，将输入的特征图，与输出特征图 对应维度进行相加，即 $y=f(x)+x$ ；而concat操作源于DenseNet网络的设计思路，将 特征图按照通道维度直接进行拼接，例如 $8^{*} 8^{*} 16$ 的特征图与 $8^{*} 8^{*} 16$ 的特征图拼接后生成 $8^{*} 8^{*} 32$ 的特征图。
5. **上采样层(upsample)**：作用是将小尺寸特征图通过揷值等方法，生成大尺寸图像。例如使用最 近邻揷值算法，将 $8^{*} 8$ 的图像变换为 $16^{*} 16$ 。上采样层不改变特征图的通道数。
6. 为了**防止信息丢失**，在卷积下采样过程中通道数加倍。
7. 特征输入尺寸一定的情况下，加深网络层数意义不大。输入尺寸为32的倍数。

Yolo的整个网络，吸取了Resnet、Densenet、FPN的精髓，可以说是融合了目标检测当前业界最 有效的全部技巧。

### 每个框的输出

针对coco：80(类别)+$t_{x}, t_{y}, t_{w}, t_{h} ，conf$(每个框的x,y,w,h,conf) ，一共85，三个框 ：85*3 = 255

### 损失函数

使用交叉熵进行类别计算6.Ground Truth的计算

### Ground Truth

既然网络预测的是偏移值，那么在计算损失时，也是按照偏移值计算损失。现在我们有预测的值， 还需要真值Ground Truth的偏移值，用于计算损失的GT按照以下公式得到:
$$
\begin{aligned}
t x &=G x-C x \\
t y &=G y-C y \\
t w &=\log (G w / P w) \\
t h &=\log (G h / P h)
\end{aligned}
$$

### 为什么在计算Ground Truth的tw，th时需要缩放到对数空间

tw和th是物体所在边框的长宽和anchor box长宽之间的比率。不直接回归bounding box的长 宽，而是为避免训练带来不稳定的梯度，将尺度缩放到对数空间。如果直接预测相对形变tw 和 th，那么要求tw, th $>0$ ，因为框的宽高不可能是负数，这样的话是在做一个有不等式条件约束的优 化问题，没法直接用SGD来做，所以先取一个对数变换，将其不等式约束去掉就可以了。

对于三个框，选取IOU值最大的那个框。

- 每个GT目标仅与一个anchor相关联，与GT匹配的anchor box计算坐标误差、置信度误差（此时target为1）以及分类误差，而其他anchor box只计算置信度误差（此时target为0）。
- 对于**重叠大于等于0.5的其他先验框(anchor)**，忽略，**不算损失**。
- 总的来说，正样本是与GT的IOU最大的框。负样本是与GT的IOU<0.5的框。忽略的样本是与GT的IOU>0.5 但不是最大的框。

### 代码实现

SPP

```python

class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
```

BottleneckCSP

```python
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
```

**Bottleneck**

```python
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

```

