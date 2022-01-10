# Matplotlib初相识

## 什么是matplotlib

- Matplotlib是一个Python 2D绘图库，能够以多种硬拷贝格式和跨平台的交互式环境生成出版物质量的图形，用来绘制各种静态，动态，交互式的图表。
- 可用于Python脚本，Python和IPython Shell、Jupyter notebook，Web应用程序服务器和各种图形用户界面工具包等。
- 所熟知的pandas和seaborn的绘图接口其实也是基于matplotlib所作的高级封装。

## example

- Matplotlib的图像是画在**figure**（如windows，jupyter窗体）上的，每一个figure又包含了一个或多个axes（一个可以指定坐标系的子区域）。最简单的创建figure以及axes的方式是通过`pyplot.subplots`命令，创建axes以后，可以使用`Axes.plot`绘制最简易的折线图。

  ```python
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  import numpy as np
  
  fig, ax = plt.subplots()  # 创建一个包含一个axes的figure
  ax.plot([1, 2, 3, 4], [1, 4, 2, 3]);  # 绘制图像
  #line =plt.plot([1, 2, 3, 4], [1, 4, 2, 3]) 
  ```

  ![../_images/index_2_0.png](../images/Matplotlib%E5%88%9D%E7%9B%B8%E8%AF%86.assets/index_2_0.png)

  ## Figure的组成

  ![img](../images/Matplotlib%E5%88%9D%E7%9B%B8%E8%AF%86.assets/anatomy-16418127505793.png)

  - `Figure`：顶层级，用来容纳所有绘图元素
  - `Axes`：matplotlib宇宙的核心，容纳了大量元素用来构造一幅幅子图，一个figure可以由一个或多个子图组成
  - `Axis`：axes的下属层级，用于处理所有和坐标轴，网格有关的元素
  - `Tick`：axis的下属层级，用来处理所有和刻度有关的元素

  理解容器之间的关系，方便画图理解时能够知道相互之间的层级关系，从而更方便的绘图。

  ## 绘图接口

  ### OO模式

  - 显式创建figure和axes，在上面调用绘图方法

  - ```python
    x = np.linspace(0, 2, 100)
    
    fig, ax = plt.subplots()  
    # label 对应的是标签名字
    
    ax.plot(x, x, label='linear')  
    ax.plot(x, x**2, label='quadratic')  
    ax.plot(x, x**3, label='cubic')  
    ax.set_xlabel('x label') 
    ax.set_ylabel('y label') 
    ax.set_title("Simple Plot")  
    # 画板
    ax.legend() 
    # 显示
    plt.show()
    ```

![../_images/index_6_0.png](../images/Matplotlib%E5%88%9D%E7%9B%B8%E8%AF%86.assets/index_6_0.png)

### 依赖pyplot自动创建figure和axes，并绘图

```python
x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear') 
plt.plot(x, x**2, label='quadratic')  
plt.plot(x, x**3, label='cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple Plot")
plt.legend()
plt.show()
```

![../_images/index_6_0.png](../images/Matplotlib%E5%88%9D%E7%9B%B8%E8%AF%86.assets/index_6_0-16418132453865.png)

## 绘图模板

- 任何复杂的图表几乎都可以基于这个模板骨架填充内容而成。初学者刚开始学习时只需要牢记这一模板就足以应对大部分简单图表的绘制，在学习过程中可以将这个模板模块化，了解每个模块在做什么，在绘制复杂图表时如何修改，填充对应的模块。

```python
# step1 准备数据
x = np.linspace(0, 2, 100)
y = x**2

# step2 设置绘图样式，这一模块的扩展参考第五章进一步学习，这一步不是必须的，样式也可以在绘制图像是进行设置
mpl.rc('lines', linewidth=4, linestyle='-.')

# step3 定义布局， 这一模块的扩展参考第三章进一步学习
fig, ax = plt.subplots()  

# step4 绘制图像， 这一模块的扩展参考第二章进一步学习
ax.plot(x, y, label='linear')  

# step5 添加标签，文字和图例，这一模块的扩展参考第四章进一步学习
ax.set_xlabel('x label') 
ax.set_ylabel('y label') 
ax.set_title("Simple Plot")  
ax.legend() ;
```

## 思考题

- 请思考两种绘图模式的优缺点和各自适合的使用场景

  ```txt
  - 模式调用更加的精细化，根据设置的对象参数从而能够实现目标的精细刻画，但是相对来说效率较低，代码较多。以类的方法来调用修改。
  - pyplot模式，调用更方便快捷，自动调用。以类属性的定义来进行修改。
  ```

  

- 在第五节绘图模板中我们是以OO模式作为例子展示的，请思考并写一个pyplot绘图模式的简单模板

```python
# step1 准备数据
x = np.linspace(0, 2, 100)
y = x**2

# step2 设置绘图样式，这一模块的扩展参考第五章进一步学习，这一步不是必须的，样式也可以在绘制图像是进行设置
mpl.rc('lines', linewidth=4, linestyle='-.')

# step3 定义布局， 这一模块的扩展参考第三章进一步学习
fig, ax = plt.subplots()  

# step4 绘制图像， 这一模块的扩展参考第二章进一步学习
plt.plot(x, y, label='linear')  

# step5 添加标签，文字和图例，这一模块的扩展参考第四章进一步学习
plt.xlabel('x label') 
plt.ylabel('y label') 
plt.title("Simple Plot")  
plt.legend() ;
```

