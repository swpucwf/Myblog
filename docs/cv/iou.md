## 一、IOU(Intersection over Union)

### **1. 特性(优点)**

IoU就是我们所说的**交并比**，是目标检测中最常用的指标，他的作用不仅用来确定正样本和负样本，还可以用来评价输出框（predict box）和ground-truth的距离。
$$
I o U=\frac{|A \cap B|}{|A \cup B|}
$$

1. 可以说**它可以反映预测检测框与真实检测框的检测效果。**

2. 还有一个很好的特性就是**尺度不变性**，也就是对尺度不敏感（scale invariant）， 在regression任务中，判断predict box和gt的距离最直接的指标就是IoU。**(满足非负性；同一性；对称性；三角不等性)**

   ```python
   import numpy as np
   def iou(box,boxes,is_Min= True):
       box_area = (box[2]-box[0])*(box[3]-box[1])
       boxes_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
       x1 = np.maximum(box[0],boxes[:,0])
       y1 = np.maximum(box[1],boxes[:,1])
       x2 = np.maximum(box[2],boxes[:,2])
       y2 = np.maximum(box[3],boxes[:,3])
   
       w = np.maximum(0,x2-x1)
       h = np.maximum(0,y2-y1)
   
       inter = w*h
   
       if is_Min:
           return  np.true_divide(inter/np.maximum(box_area,boxes_area))
       else:
           return np.true_divide(inter,(box_area+boxes_area-inter))
   ```

### **2. 作为损失函数会出现的问题(缺点)**

1. 如果两个框没有相交，根据定义，IoU=0，不能反映两者的距离大小（重合度）。同时因为loss=0，没有梯度回传，无法进行学习训练。
2. IoU无法精确的反映两者的重合度大小。如下图所示，三种情况IoU都相等，但看得出来他们的重合度是不一样的，左边的图回归的效果最好，右边的最差。
3. ![image-20220406234353271](IOU%E7%AE%97%E6%B3%95%E8%AF%A6%E8%A7%A3.assets/image-20220406234353271.png)

## GIOU(Generalized Intersection over Union)

提出了GIoU的思想。由于IoU是**比值**的概念，对目标物体的scale是不敏感的。然而检测任务中的BBox的回归损失(MSE loss, l1-smooth loss等）优化和IoU优化不是完全等价的，而且 Ln 范数对物体的scale也比较敏感，IoU无法直接优化没有重叠的部分。
$$
G I o U=I o U-\frac{\left|A_{c}-U\right|}{\left|A_{c}\right|}
$$
上面公式的意思是: 先计算两个框的最小闭包区域面积 $A_{c}$ (通俗理解: 同时包含了预测框和真实 框的最小框的面积)，再计算出loU，再计算闭包区域中不属于两个框的区域占闭包区域的比重，最 后用IoU减去这个比重得到 GloU。

### **2. 特性**

- 与IoU相似，GIoU也是一种距离度量，作为损失函数的话，

- $$
  \text  L_{G I o U}=1-G I o U
  $$

- 满足损失函数的基本要求

- GIoU对scale不敏感

- GIoU是IoU的下界，在两个框无限重合的情况下，IoU=GIoU=1

- IoU取值[0,1]，但GIoU有对称区间，取值范围[-1,1]。在两者重合的时候取最大值1，在两者无交集且无限远的时候取最小值-1，因此GIoU是一个非常好的距离度量指标。

- 与IoU只关注重叠区域不同，**GIoU不仅关注重叠区域，还关注其他的非重合区域**，能更好的反映两者的重合度。

### 三、DIoU(Distance-IoU)

DIoU要比GIou更加符合目标框回归的机制，**将目标与anchor之间的距离，重叠率以及尺度都考虑进去**，使得目标框回归变得更加稳定，不会像IoU和GIoU一样出现训练过程中发散等问题。论文中

> 基于IoU和GIoU存在的问题，作者提出了两个问题：
> \1. 直接最小化anchor框与目标框之间的归一化距离是否可行，以达到更快的收敛速度？
> \2. 如何使回归在与目标框有重叠甚至包含时更准确、更快？

$$
D I o U=I o U-\frac{\rho^{2}\left(b, b^{g t}\right)}{c^{2}}
$$

其中， $b ， b^{g t}$ 分别代表了预测框和真实框的中心点，且 $\rho$ 代表的是计算两个中心点间的欧式 距离。 $c$ 代表的是能够同时包含预测框和真实框的最小闭包区域的对角线距离。

### **2.优点**

- 与GIoU loss类似，DIoU loss:$$\text  L_{D I o U}=1-D I o U$$在与目标框不重叠时，仍然可以为边界框提供移动方向。
- DIoU loss可以直接最小化两个目标框的距离，因此比GIoU loss收敛快得多。
- 对于包含两个框在水平方向和垂直方向上这种情况，DIoU损失可以使回归非常快，而GIoU损失几乎退化为IoU损失。
- DIoU还可以替换普通的IoU评价策略，应用于NMS中，使得NMS得到的结果更加合理和有效。
- 中心点重合，但宽高比不同时，DIOU loss不变。

## 四、CIoU(**Complete-IoU**)

论文考虑到bbox回归三要素中的长宽比还没被考虑到计算中，因此，进一步在DIoU的基础上提出了CIoU。其惩罚项如下面公式：

论文考虑到bbox回归三要素中的长宽比还没被考虑到计算中，因此，进一步在DloU的基础上提出 了CloU。其惩罚项如下面公式：
$\mathcal{R}_{C I o U}=\frac{\rho^{2}\left(\mathbf{b}, \mathbf{b}^{g t}\right)}{c^{2}}+\alpha v$ 其中 $\alpha$ 是权重函数，
而 $\nu$ 用来度量长宽比的相似性，定义为 $v=\frac{4}{\pi^{2}}\left(\arctan \frac{w^{g t}}{h^{g t}}-\arctan \frac{w}{h}\right)^{2}$
完整的 CloU 损失函数定义:
$$
\mathcal{L}_{C I o U}=1-I o U+\frac{\rho^{2}\left(\mathbf{b}, \mathbf{b}^{g t}\right)}{c^{2}}+\alpha v
$$
最后，CloU loss的梯度类似于DloU loss，但还要考虑 $\nu$ 的梯度。在长宽在 $[0,1]$ 的情况下， $w^{2}+h^{2}$ 的值通常很小，会导致梯度爆炸，因此在 $\frac{1}{w^{2}+h^{2}}$ 实现时将替换成1。

```python
def bbox_overlaps_ciou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious
```

### Focal-EIOU

CIOU Loss虽然考虑了边界框回归的重叠面积、中心点距离、纵横比。

但是通过其公式中的v反映的纵横比的差异，而不是宽高分别与其置信度的真实差异，所以有时会阻碍模型有效的优化相似性。

在CIOU的基础上将纵横比拆开，提出了EIOU Loss，并且加入Focal聚焦优质的锚框。

EIOU的惩罚项是在CIOU的惩罚项基础上将纵横比的影响因子拆开分别计算目标框和锚框的长和宽，该损失函数包含三个部分：重叠损失，中心距离损失，宽高损失，前两部分延续CIOU中的方法，但是宽高损失直接使目标盒与锚盒的宽度和高度之差最小，使得收敛速度更快。

惩罚项公式如下：
$$
\begin{aligned}
&L_{E I O U}=L_{I O U}+L_{d i s}+L_{a s p} \\
&\quad=1-I O U+\frac{\rho^{2}\left(\mathbf{b}, \mathbf{b}^{\mathbf{g t}}\right)}{c^{2}}+\frac{\rho^{2}\left(w, w^{g t}\right)}{C_{w}^{2}}+\frac{\rho^{2}\left(h, h^{g t}\right)}{\mathrm{G}_{h}^{2}}
\end{aligned}
$$
1）将纵横比的损失项拆分成预测的宽高分别与最小外接框宽高的差值，加速了收敛提高了回归精度。

2）引入了Focal Loss优化了边界框回归任务中的样本不平衡问题，即减少与目标框重叠较少的大量锚框对BBox 回归的优化贡献，使回归过程专注于高质量锚框。



3. Focal-EloU Loss
通过整合EloU Loss和FocalL1 loss，我们得到了最终的Focal-EloU loss，它表示为式。
$$
\mathcal{L}_{\text {Focal E-IoU }}=\mathrm{IoU}^{\gamma} \mathcal{L}_{\text {EIoU }}
$$
其中 $\gamma$ 是一个用于控制曲线㧓度的超参。此外作者还尝试了式(14)的形式的Focal Loss，但效果 并不如式。
$$
\mathcal{L}_{\text {Focal E-IoU }}^{*}=-(1-\mathrm{IoU})^{\gamma} \log (\mathrm{IoU}) \mathrm{EIoU}
$$


###  Alpha-Iou

IOU损失对于bbox scales 是不变的，可以训练出更好的检测器，在后期工作中，有CIou、DIou、GIou、EIou等loss出现，至于之间区别，这里不做赘述，可以参考江大白的公众号，**α_iou loss**则是将之前的loss 集成进来，该系列具有一个Power IoU项和一个附加的Power正则项，具有单个Power参数α，称这种新的损失系列为α-IoU Loss。

α-IoU Loss是基于现有Iou_Loss 的统一幂化，具体计算公式为:
$$
\begin{aligned}
&\mathcal{L}_{\mathrm{loU}}=1-I o U \Longrightarrow \mathcal{L}_{\alpha-\mathrm{loU}}=1-I o U^{\alpha} \text {, }\\
&\mathcal{L}_{\text {GloU }}=1-I o U+\frac{\left|C \backslash\left(B \cup B^{g t}\right)\right|}{|C|} \Longrightarrow \mathcal{L}_{\alpha-\text { GloU }}=1-I o U^{\alpha}+\left(\frac{\left|C \backslash\left(B \cup B^{g t}\right)\right|}{|C|}\right)^{\alpha} \text {, }\\
&\mathcal{L}_{\text {DloU }}=1-I o U+\frac{\rho^{2}\left(b, b^{g t}\right)}{c^{2}} \Longrightarrow \mathcal{L}_{\alpha-\mathrm{DloU}}=1-I o U^{\alpha}+\frac{\rho^{2 \alpha}\left(b, b^{g t}\right)}{c^{2 \alpha}},\\
&\mathcal{L}_{\mathrm{CloU}}=1-I o U+\frac{\rho^{2}\left(\boldsymbol{b}, b^{g t}\right)}{c^{2}}+\beta v \Longrightarrow \mathcal{L}_{\alpha-\mathrm{CloU}}=1-I o U^{\alpha}+\frac{\rho^{2 \alpha}\left(\boldsymbol{b}, b^{g t}\right)}{c^{2 \alpha}}+(\beta v)^{\alpha},
\end{aligned}
$$

```python

def bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, alpha=2):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # elif whratio: # transform from xywhratio to xyxy
    #     b1_x1, b1_x2 = box1[0] - box1[2] / (whratio * 2), box1[0] + box1[2] / (whratio * 2)
    #     b1_y1, b1_y2 = box1[1] - box1[3] / (whratio * 2), box1[1] + box1[3] / (whratio * 2)
    #     b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    #     b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # iou = inter / union
    iou = torch.pow(inter / union + eps, alpha)
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2 # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha_ciou)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou # torch.log(iou+eps) or iou

```

a_iou_loss

```python
def bbox_alpha_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, alpha=2, eps=1e-9):
    # Returns tsqrt_he IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    # change iou into pow(iou+eps)
    # iou = inter / union
    iou = torch.pow(inter/union + eps, alpha)
    beta = 2 * alpha
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** beta + ch ** beta + eps  # convex diagonal
            rho_x = torch.abs(b2_x1 + b2_x2 - b1_x1 - b1_x2)
            rho_y = torch.abs(b2_y1 + b2_y2 - b1_y1 - b1_y2)
            rho2 = (rho_x ** beta + rho_y ** beta) / (2 ** beta)  # center distance
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha_ciou = v / ((1 + eps) - inter / union + v)
                # return iou - (rho2 / c2 + v * alpha_ciou)  # CIoU
                return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            # c_area = cw * ch + eps  # convex area
            # return iou - (c_area - union) / c_area  # GIoU
            c_area = torch.max(cw * ch + eps, union) # convex area
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU
    else:
        return iou # torch.log(iou+eps) or iou
```

