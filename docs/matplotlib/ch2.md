# 第二回：艺术画笔见乾坤

## 概述

1.`matplotlib.backend_bases.FigureCanvas` 代表了绘图区，所有的图像都是在绘图区完成的

2.`matplotlib.backend_bases.Renderer` 代表了渲染器，可以近似理解为画笔，控制如何在 FigureCanvas 上画图。

3.`matplotlib.artist.Artist` 代表了具体的图表组件，即调用了Renderer的接口在Canvas上作图。

## Atist两种类型

1. `primitives` 

   基本要素，它包含一些我们要在绘图区作图用到的标准图形对象，如**曲线Line2D，文字text，矩形Rectangle，图像image**等。

2. `containers`

   容器，即用来装基本要素的地方，包括**图形figure、坐标系Axes和坐标轴Axis**。

他们之间的关系如下图所示：

![分类](../images/%E8%89%BA%E6%9C%AF%E7%94%BB%E7%AC%94%E8%A7%81%E4%B9%BE%E5%9D%A4.assets/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODYwNDk2MQ==,size_16,color_FFFFFF,t_70%23pic_center.jpeg)
$$
\begin{array}{lll}
\hline \text { Axes helper method } & \text { Artist } & \text { Container } \\
\hline \text { bar-bar charts } & \text { Rectangle } & \text { ax.patches } \\
\hline \text { errorbar - error bar plots } & \text { Line2D and Rectangle } & \text { ax.lines and ax.patches } \\
\hline \text { fill - shared area } & \text { Polygon } & \text { ax.patches } \\
\hline \text { hist - histograms } & \text { Rectangle } & \text { ax.lines } \\
\hline \text { imshow - image data } & \text { AxesImage } & \text { ax.images } \\
\hline \text { plot-xy plots } & \text { Line2D } & \\
\hline
\end{array}
$$

## 基本元素讲解

### 基本元素 - primitives 

#### 2DLines

在matplotlib中曲线的绘制，主要是通过类 `matplotlib.lines.Line2D` 来完成的。

matplotlib中`线-line`的含义：它表示的可以是连接所有顶点的实线样式，也可以是每个顶点的标记。此外，这条线也会受到绘画风格的影响，比如，我们可以创建虚线种类的线。

其中常用的的参数有：

- **xdata**:需要绘制的line中点的在x轴上的取值，若忽略，则默认为range(1,len(ydata)+1)
- **ydata**:需要绘制的line中点的在y轴上的取值
- **linewidth**:线条的宽度
- **linestyle**:线型
- **color**:线条的颜色
- **marker**:点的标记，详细可参考[markers API](https://matplotlib.org/api/markers_api.html#module-matplotlib.markers)
- **markersize**:标记的size

##### 设置属性

1. 直接在plot()函数中设置

   ```python
   # 1) 直接在plot()函数中设置
   x = range(0,5)
   y = [2,5,7,8,10]
   plt.plot(x,y, linewidth=10); # 设置线的粗细参数为10
   ```

   ![../_images/index_2_01.png](../images/%E8%89%BA%E6%9C%AF%E7%94%BB%E7%AC%94%E8%A7%81%E4%B9%BE%E5%9D%A4.assets/index_2_01.png)

2. 通过获得线对象，对线对象进行设置

   ```python
   # 2) 通过获得线对象，对线对象进行设置
   x = range(0,5)
   y = [2,5,7,8,10]
   line, = plt.plot(x, y, '-') # 这里等号坐标的line,是一个列表解包的操作，目的是获取plt.plot返回列表中的Line2D对象
   line.set_antialiased(False); # 关闭抗锯齿功能
   ```

   ![../_images/index_3_01.png](../images/%E8%89%BA%E6%9C%AF%E7%94%BB%E7%AC%94%E8%A7%81%E4%B9%BE%E5%9D%A4.assets/index_3_01.png)

3. 获得线属性，使用setp()函数设置

   ```python
   # 3) 获得线属性，使用setp()函数设置
   x = range(0,5)
   y = [2,5,7,8,10]
   lines = plt.plot(x, y)
   plt.setp(lines, color='r', linewidth=10);
   ```

   ![../_images/index_4_01.png](../images/%E8%89%BA%E6%9C%AF%E7%94%BB%E7%AC%94%E8%A7%81%E4%B9%BE%E5%9D%A4.assets/index_4_01.png)

   

#### 如何绘制lines

1. 绘制直线line
2. errorbar绘制误差折线图

介绍两种绘制直线line常用的方法:

- **plot方法绘制**

  ```python
  # 1. plot方法绘制
  x = range(0,5)
  y1 = [2,5,7,8,10]
  y2= [3,6,8,9,11]
  
  fig,ax= plt.subplots()
  ax.plot(x,y1)
  ax.plot(x,y2)
  print(ax.lines); # 通过直接使用辅助方法画线，打印ax.lines后可以看到在matplotlib在底层创建了两个Line2D对象
  ```

  ![../_images/index_6_1.png](../images/%E8%89%BA%E6%9C%AF%E7%94%BB%E7%AC%94%E8%A7%81%E4%B9%BE%E5%9D%A4.assets/index_6_1.png)

- **Line2D对象绘制**

```python
# 2. Line2D对象绘制

x = range(0,5)
y1 = [2,5,7,8,10]
y2= [3,6,8,9,11]
fig,ax= plt.subplots()
lines = [Line2D(x, y1), Line2D(x, y2,color='orange')]  # 显式创建Line2D对象
for line in lines:
    ax.add_line(line) # 使用add_line方法将创建的Line2D添加到子图中
ax.set_xlim(0,4)
ax.set_ylim(2, 11);
```

![../_images/index_7_01.png](../images/%E8%89%BA%E6%9C%AF%E7%94%BB%E7%AC%94%E8%A7%81%E4%B9%BE%E5%9D%A4.assets/index_7_01.png)