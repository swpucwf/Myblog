[TOC]

# W-GAN

## 1. W-GAN介绍

###  1.1解决问题

- 彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
- 基本解决了collapse mode的问题，确保了生成样本的多样性
- 训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高（如题图所示）
- 以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到

### 1.2 改进点

- 判别器最后一层去掉sigmoid
- 生成器和判别器的loss不取log
- 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
- 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行

[项目实现链接](https://github.com/martinarjovsky/WassersteinGAN)

## 2. 原始GAN问题?

### 2.1 原始GAN损失定义

原始GAN损失函数，原始GAN中判别器要最小化如下损失函数，尽可能把真实样本分为正例，生成样本分为负例:
$$
-\mathbb{E}_{x \sim P_r}[\log D(x)]-\mathbb{E}_{x \sim P_g}[\log (1-D(x))] \text { (公式1 ) }
$$
其中 $P_r$ 是真实样本分布， $P_g$ 是由生成器产生的样本分布。对于生成器，Goodfellow一开始提出 来一个损失函数，后来又提出了一个改进的损失函数，分别是
$$
\begin{aligned}
& \mathbb{E}_{x \sim P_g}[\log (1-D(x))] \text { (公式2) } \\
& \mathbb{E}_{x \sim P_g}[-\log D(x)] \text { (公式3) }
\end{aligned}
$$
后者在WGAN两篇论文中称为 "the - $\log D$ alternative" 或 "the - $\log D$ trick" 。

### 2.2 存在的问题

- **判别器越好，生成器梯度消失越严重。**

对于一个具体的样本x，它可能来自真实分布也可能来自生成分布，它对公式1损失函数的贡献是
$$
-P_r(x) \log D(x)-P_g(x) \log [1-D(x)]
$$
令其关于 $D(x)$ 的导数为 0 ，得
$$
-\frac{P_r(x)}{D(x)}+\frac{P_g(x)}{1-D(x)}=0
$$
化简得最优判别器为:
$$
D^*(x)=\frac{P_r(x)}{P_r(x)+P_g(x)} \text { (公式4) }
$$
一个样本 $x$ 来自真实分布和生成分布的可能性的相对比例。 如果 $P_r(x)=0$ 且 $P_g(x) \neq 0$ ，最优判别器就应该非常自信地给出概率0； 如果 $P_r(x)=P_g(x)$ ，说明该样本是真是假的可能性刚好一半一半，此时最优判别器也应该给出概率 $0.5$ 。
**但是**GANi川练有一个trick，就是别把判别器训练得太好，否则在实验中生成器会完全学不动 (loss降不下去)，为了探究背后的原因，我们就可以看看在极端情况一一判别器最优时，生成器 的损失函数变成什么。给公式2加上一个不依赖于生成器的项，使之变成：
$$
\mathbb{E}_{x \sim P_r}[\log D(x)]+\mathbb{E}_{x \sim P_g}[\log (1-D(x))]
$$
注意，最小化这个损失函数等价于最小化公式2，而且它刚好是判别器损失函数的反。代入最优判 别器即公式4，再进行简单的变换可以得到
$$
\mathbb{E}_{x \sim P_r} \log \frac{P_r(x)}{\frac{1}{2}\left[P_r(x)+P_g(x)\right]}+\mathbb{E}_{x \sim P_g} \log \frac{P_g(x)}{\frac{1}{2}\left[P_r(x)+P_g(x)\right]}-2 \log 2 \text { (公式5) }
$$
引入Kullback–Leibler divergence（简称KL散度）和 Jensen-Shannon divergence（简称JS散度）这两个重要的相似度衡量指标，——KL散度和JS散度：
$$
\begin{aligned}
& K L\left(P_1 \| P_2\right)=\mathbb{E}_{x \sim P_1} \log \frac{P_1}{P_2} \text { (公式6) } \\
& J S\left(P_1 \| P_2\right)=\frac{1}{2} K L\left(P_1 \| \frac{P_1+P_2}{2}\right)+\frac{1}{2} K L\left(P_2 \| \frac{P_1+P_2}{2}\right) \text { (公式7) }
\end{aligned}
$$
于是公式5就可以继续写成：
$$
2 J S\left(P_r \| P_g\right)-2 \log 2(公式8)
$$
结论：根据原始GAN定义的判别器loss，我们可以得到最优判别器的形式；而在最优判别器的下，我们可以**把原始GAN定义的生成器loss等价变换为最小化真实分布 $P_r$ 与生成分布 $P_g$ 之间的JS散度。**我们越训练判别器，它就越接近最优，**最小化生成器的loss也就会越近似于最小化 $P_r$ 和 $P_g$ 之间的JS散度。**

问题就出在这个**JS散度**上。

本意优化：我们会希望**如果两个分布之间越接近它们的JS散度越小，我们通过优化JS散度就能将 $P_g$ “拉向" $P_r$ ，最终以假乱真。**

- 结果建立于这个希望在两个分布有所重叠的时候是成立的
- 问题：如果两个分布完全没有重叠的部分，或者它们重叠的部分可忽略。此时它们的JS散度为$log2$

假设以下四种情形：

对于任意一个x只有四种可能:
$$
\begin{aligned}
& P_1(x)=0 \text { 且 } P_2(x)=0 \\
& P_1(x) \neq 0 \text { 且 } P_2(x) \neq 0 \\
& P_1(x)=0 \text { 且 } P_2(x) \neq 0 \\
& P_1(x) \neq 0 \text { 且 } P_2(x)=0
\end{aligned}
$$
**结论一：只要两个分布之间没有一点重叠或者重埴部分可忽略，JS散度就固定是常数 $\log 2$ ，基于梯度下降法得到的最优判别器同时生成器无法获取梯度信息。同理即使对于接近最优的判别器来说，生成器也有很 大机会面临梯度消失的问题。**

**结论二：。$P_r$ 与 $P_g$ 不重㥀或重直部分可忽略的可能性非常大，换句话说：当 $P_r$ 与 $P_g$ 的支撑集 (support) 是高维空间中的低维流形 (manifold) 时， $P_r$ 与 $P_g$ 重叠 部分测度 (measure) 为 0 的概率为 1 。**

- 支撑集 (support) 其实就是**函数的非零部分子集**，比如ReLU函数的支撑集就是 $(0,+\infty)$ ， **一个概率分布的支撑集就是所有概率密度非零部分的集合。**
- **流形 (manifold) 是高维空间中曲线、曲面概念的拓广**，我们可以在低维上直观理解这个概念，比如我们说三维空间中的一个曲面是一个二维流形，因为它的本质维度 (intrinsic dimension) 只有2，一个点在这个二维流形上移动只有两个方向的自由度。同理，三维空间或者二维空间中的一条曲线都是一个一维流形。
- **测度 (measure) 是高维空间中长度、面积、体积概念的拓广，可以理解为 "超体积" 。**

回过头来看第一句话， 

- "**当 $P_r$ 与 $P_g$ 的支撑集是高维空间中的低维流形时**"，基本上是成立的。原因是**GAN中的生成器一般是从某个低维（比如100维) 的随机分布中采样出一个编码向量，再经过一个神经网络生成出一个高维样本 (比如64x64的图片就有4096维) 。**
- 当生成器的参数固定时，生成样本的概率分布虽然是定义在 4096 维的空间上，但它本身所有可能产生的变化已经被那个100维的随机分布限定了，其本质维度就是100 ，再考虑到神经网络带来的映射降维，最终可能比 100 还小，所以生成样本分布的支撑集就在4096维空间中构成一个最多100维的低维流形， "撑不满" 整个高维空间。**生成样本的支撑集无法支撑完全的高维空间。**
- **"撑不满" 就会导致真实分布与生成分布难以 "碰到面"**。这很容易在二维空间中理解：一方面，二维平面中随机取两条曲线，它们之间刚好存在重叠线段的概率为 0；另一方面，虽然它们很大可 能会存在交叉点，但是相比于两条曲线而言，交叉点比曲线低一个维度，长度 (测度) 为 0 ，可忽略。三维空间中也是类似的，随机取两个曲面，它们之间最多就是比较有可能存在交叉线，但是交叉线比曲面低一个维度，面积（测度）是0，可忽略。从低维空间拓展到高维空间，就有了如下逻辑：因为一开始生成器随机初始化，所以 $P_g$ 几乎不可能与 $P_r$ 有什么关联，所以它们的支撑集之间 的重叠部分要么不存在，要么就比 $P_r$ 和 $P_g$ 的最小维度还要低至少一个维度，故而测度为0 。就是上文所言“不重叠或者重叠部分可忽略”的意思。

我们就得到了WGAN前作中关于生成器梯度消失的第一个论证: 

**在 (近似) 最优判别器下，最小化生成器的loss等价于最小化 $P_r$ 与 $P_g$ 之间的JS散度，而由于 $P_r$ 与 $P_g$ 几平不可能有不可忽略的重叠，所以无论它们相距多远，JS散度都是常数$log2$，最终导致生成器的梯度(近似)为0 ，梯度消失。**

- 首先， $P_r$ 与 $P_g$ 之间几乎不可能有不可忽略的重叠，所以无论它们之间的 “缝隙” 多狭小，都肯定存在一个最优分割曲面把它们隔开，最多就是在那些可忽略的重叠处隔不开而已。
- 由于判别器作为一个神经网络可以无限拟合这个分隔曲面，所以存在一个最优判别器，对几乎所有真实样本给出概率1，对几乎所有生成样本给出概率0，而那些隔不开的部分就是难以被最优判别器分类的样本，但是它们的测度为0，可忽略。
- 最优判别器在真实分布和生成分布的支撑集上给出的概率都是常数（1和0)，导致生成器的loss 梯度为 0 ，梯度消失。

## 3. 第二种原始GAN形式的问题

**最小化第二种生成器loss函数，会等价于最小化一个不合理的距离衡量，导致两个问题，一是梯度不稳定，二是collapse mode即多样性不足。**
如前文所说， Ian Goodfellow提出的 "- log D trick" 是把生成器loss改成
$$
\mathbb{E}_{x \sim P_g}[-\log D(x)] \text { (公式3) }
$$
上文推导已经得到在最优判别器 $D^*$ 下
$$
\mathbb{E}_{x \sim P_r}\left[\log D^*(x)\right]+\mathbb{E}_{x \sim P_g}\left[\log \left(1-D^*(x)\right)\right]=2 J S\left(P_r \| P_g\right)-2 \log 2 \text { (公式9) }
$$
我们可以把KL散度 (注意下面是先 $g$ 后 $r$ ) 变换成含 $D^*$ 的形式:
$$
\begin{aligned}
K L\left(P_g \| P_r\right) & =\mathbb{E}_{x \sim P_g}\left[\log \frac{P_g(x)}{P_r(x)}\right] \\
& =\mathbb{E}_{x \sim P_g}\left[\log \frac{P_g(x) /\left(P_r(x)+P_g(x)\right)}{P_r(x) /\left(P_r(x)+P_g(x)\right)}\right] \\
& =\mathbb{E}_{x \sim P_g}\left[\log \frac{1-D^*(x)}{D^*(x)}\right] \\
& =\mathbb{E}_{x \sim P_g} \log \left[1-D^*(x)\right]-\mathbb{E}_{x \sim P_g} \log D^*(x)          (公式10)
\end{aligned}
$$
由公式3，9，10可得最小化目标的等价变形
$$
\begin{aligned}
\mathbb{E}_{x \sim P_g}\left[-\log D^*(x)\right] & =K L\left(P_g \| P_r\right)-\mathbb{E}_{x \sim P_g} \log \left[1-D^*(x)\right] \\
& =K L\left(P_g \| P_r\right)-2 J S\left(P_r \| P_g\right)+2 \log 2+\mathbb{E}_{x \sim P_r}\left[\log D^*(x)\right]
\end{aligned}
$$
注意上式最后两项不依赖于生成器 $G$ ，最终得到最小化公式3等价于最小化
$$
K L\left(P_g \| P_r\right)-2 J S\left(P_r \| P_g\right) \text { (公式11) }
$$
这个等价最小化目标存在两个严重的问题。

### 3.1 梯度不稳定

- **最小化生成分布与真实分布的KL散度，却又要最大化两者的JS散度**，一个要拉近，一个却要推远! 这在直观上非常荒谬，**在数值上则会导致梯度不稳定**，这是后面那个JS散度项的毛病。

### 3.2 多样性不足

- 即便是前面那个正常的KL散度项也有毛病。因为 **$K L$ 散度不是一个对称的衡量**， $K L\left(P_g \| P_r\right)$ 与 $K L\left(P_r \| P_g\right)$ 是有差别的。以前者为例

  1. 当 $P_g(x) \rightarrow 0$ 而 $P_r(x) \rightarrow 1$ 时， $P_g(x) \log \frac{P_g(x)}{P_r(x)} \rightarrow 0$ ，对 $K L\left(P_g \| P_r\right)$ 贡献趋近0

  2. 当 $P_g(x) \rightarrow 1$ 而 $P_r(x) \rightarrow 0$ 时， $P_g(x) \log \frac{P_g(x)}{P_r(x)} \rightarrow+\infty$ ，对 $K L\left(P_g \| P_r\right)$ 贡献趋近正无穷

 $K L\left(P_g \| P_r\right)$ 对于上面两种错误的惩罚是不一样的，

- 第一种错误对应的是 “生成器没能生成真实的样本"，惩罚微小；

- 第二种错误对应的是 "生成器生成了不真实的样本"，惩罚巨大。

第一种错误对应的是**缺乏多样性**，第二种错误对应的是**缺乏准确性**。这一放一打之下，生成器宁可多生成一些重复但是很 ”安全“的样本，也不愿意去生成多样性的样本，因为那样一不小心就会产生第二种错误，得不偿失。这种现象就是大家常说的collapse mode。

### 3.3 小结 

**在原始GAN的 (近似) 最优判别器下，第一种生成器loss面临梯度消失问题**

**第 二种生成器loss面临优化目标荒谬、梯度不稳定、对多样性与准确性惩罚不平衡导致mode collapse这几个问题。**

## 4. 第二部分：WGAN之前的一个过渡解决方案

原始GAN问题的根源可以归结为两点，

- 等价优化的距离衡量（KL散度、JS散度) 不合理
- 生成器随机初始化后的生成分布很难与真实分布有不可忽略的重叠。

### 4.1 加噪解决方案

WGAN前作其实已经针对第二点提出了一个解决方案，就是**对生成样本和真实样本加噪声**，直观上说，**使得原本的两个低维流形 “弥散" 到整个高维空间，强行让它们产生不可忽略的重叠。**而一 旦存在重叠，JS散度就能真正发挥作用，此时如果两个分布越靠近，它们 “弥散” 出来的重叠越多，JS散度也会越小而不会一直是一个常数，于是（在第一种原始GAN形式下）梯度消失的问题就解决了。在训练过程中，我们可以对所加的噪声进行退火 (annealing)，慢慢减小其方差，到后面两个低维流形 "本体" 都已经有重叠 时，就算把噪声完全拿掉，JS散度也能照样发挥作用，继续产生有意义的梯度把两个低维流形拉近，直到它们接近完全重合。以上是对原文的直观解释。
在这个解决方案下我们可以放心地把判别器训练到接近最优，不必担心梯度消失的问题。而当判别器最优时，对公式9取反可得判别器的最小oss为：
$$
\begin{aligned}
\min L_D\left(P_{r+\epsilon}, P_{g+\epsilon}\right) & =-\mathbb{E}_{x \sim P_{r+\epsilon}}\left[\log D^*(x)\right]-\mathbb{E}_{x \sim P_{g+\epsilon}}\left[\log \left(1-D^*(x)\right)\right] \\
& =2 \log 2-2 J S\left(P_{r+\epsilon} \| P_{g+\epsilon}\right)
\end{aligned}
$$
其中 $P_{r+\epsilon}$ 和 $P_{g+\epsilon}$ 分别是加噪后的真实分布与生成分布。反过来说，从最优判别器的loss可以反推出当前两个加噪分布的JS散度。两个加噪分布的JS散度可以在某种程度上代表两个原本分布的距离，也就是说可以通过最优判别器的loss反映训练进程！.

但是，因为加噪JS散度的具体数值受到噪声的方差影响，随着噪声的退火，前后的数值就没法比较了，所以它不能成为 $P_r$ 和 $P_g$ 距离的本质性衡量。加噪方案是**针对原始GAN问题的第二点根源提出的，解决了训练不稳定的问题，不需要小心平衡判别器训练的火候，可以放心地把判别器训练到接近最优，但是仍然没能够提供一个衡量训练进程的数值指标。**但是**WGAN本作就从第一点根源出发，用Wasserstein距 离代替JS散度，同时完成了稳定训练和进程指标的问题**

### 4.2 Wasserstein距离

####  4.2.1 Wasserstein距离定义

Wasserstein距离又叫**Earth-Mover (EM) 距离**，定义如下:
$$
W\left(P_r, P_g\right)=\inf _{\gamma \sim \Pi\left(P_r, P_g\right)} \mathbb{E}_{(x, y) \sim \gamma}[\|x-y\|] \text { (公式12) }
$$
定义:

 $\Pi\left(P_r, P_g\right)$ 是 $P_r$ 和 $P_g$ 组合起来的所有可能的联合分布的集合，反过来说， $\Pi\left(P_r, P_g\right)$ 中每一个分布的边缘分布都是 $P_r$ 和 $P_g$ 。对于每一个可能的联合分布 $\gamma$ 而言，可以从中采样 $(x, y) \sim \gamma$ 得到一个真实样本 $x$ 和一个生成样本 $y$ ，并算出这对样本的距离 $\|x-y\|$ ，所以可以计算该联合分布 $\gamma$ 下样本对距离的期望值 $\mathbb{E}_{(x, y) \sim \gamma}[\|x-y\|]$ 。在所有可能的联合分布中能够对这个期望值取到的下界inf$f_{\gamma \sim \Pi\left(P_r, P_g\right)} \mathbb{E}_{(x, y) \sim \gamma}[\|x-y\|]$ ，就定义为Wasserstein距离。

直观上可以把 $\mathbb{E}_{(x, y) \sim \gamma}[\|x-y\|]$ 理解为在 $\gamma$ 这个 "路径规划" 下把 $P_r$ 这堆 “沙土" 挪到 $P_g$ "位 置" 所需的 "消耗" ，而 $W\left(P_r, P_g\right)$ 就是 “最优路径规划" 下的 “最小消耗"，所以才叫EarthMover（推土机）距离。

#### 4.2.2 优越性

1. 相比KL散度、JS散度的优越性在于**，即便两个分布没有重叠，Wasserstein距离仍然能够反映它们的远近。**

2. KL散度和JS散度是突变的，要么最大要么最小，Wasserstein距离却是平滑的，如果我们要用梯 度下降法优化 $\theta$ 这个参数，前两者根本提供不了梯度，Wasserstein距离却可以。同理，在高维空间中如果两个分布不重叠或者重叠部分可忽略，则KL和JS散度既反映不了远近，也提供不了梯度，但 是Wasserstein却可以提供有意义的梯度。

#### 4.3.3 从Wasserstein距离到WGAN

Wasserstein距离定义 (公式12) 中的inf $\operatorname{in~}_{\gamma \sim}\left(P_r, P_g\right)$ 没法直接求解，不过没关系，作者用了一个已有的定理把它变换为如下形式
$$
W\left(P_r, P_g\right)=\frac{1}{K} \sup _{\|f\|_L \leq K} \mathbb{E}_{x \sim P_r}[f(x)]-\mathbb{E}_{x \sim P_g}[f(x)] \text { (公式13) }
$$
首先需要介绍一个概念——**Lipschitz连续**。

在一个连续函数 $f$ 上面额外施加了一个限制，要求存在一个常数 $K \geq 0$ 使得定义域内的任意两个元素 $x_1$ 和 $x_2$ 都满足
$$
\left|f\left(x_1\right)-f\left(x_2\right)\right| \leq K\left|x_1-x_2\right|
$$
此时称函数 $f$ 的Lipschitz常数为 $K$ 。简单理解，比如说 $f$ 的定义域是实数集合，那上面的要求就等价于 $f$ 的导函数绝对值不超过 $K$ 。再比如说 $\log (x)$ 就不是Lipschitz连续，因为它的导函数没有上界。Lipschitz连续条件限制了一个连续函数的最大局部变动幅度。公式13的意思就是在要求函数 $f$ 的Lipschitz常数 $\|f\|_L$ 不超过 $K$ 的条件下，对所有可能满足条件的 $f$ 取到 $\mathbb{E}_{x \sim P_r}[f(x)]-\mathbb{E}_{x \sim P_g}[f(x)]$ 的上界，然后再除以 $K$ 。特别地，我们可以用一组参数 $w$ 来 定义一系列可能的函数 $f_w$ ，此时求解公式13可以近似变成求解如下形式：
$$
K \cdot W\left(P_r, P_g\right) \approx \max _{w:\left|f_w\right|_L \leq K} \mathbb{E}_{x \sim P_r}\left[f_w(x)\right]-\mathbb{E}_{x \sim P_g}\left[f_w(x)\right] \text { (公式14) }
$$
一般思考：用上搞深度学习的人最熟悉的那一套，不就可以把 $f$ 用一个带参数 $w$ 的神经网络来表示嘛! 由于神经网络的拟合能力足够强大，我们有理由相信，这样定义出来的一系列 $f_w$ 虽然无法囊括所 有可能，但是也足以高度近似公式 13 要求的那个 $s u p_{\|f\|_L \leq K}$ 了。
最后，还不能忘了满足公式 14 中 $\left\|f_w\right\|_L \leq K$ 这个限制。我们其实不关心具体的K是多少，只要它不是正无穷就行，因为它只是会使得梯度变大 $K$ 倍，并不会影响梯度的方向。所以作者采取了一个非常简单的做法，就是**限制神经网络 $f_\theta$ 的所有参数 $w_i$ 的不超过某个范围 $[-c, c]$ ，比如 $w_i \in[-0.01,0.01]$ ，此时关于输入样本 $x$ 的导数 $\frac{\partial f_w}{\partial x}$ 也不会超过某个范围，所以一定存在某个 不知道的常数 $K$ 使得 $f_w$ 的局部变动幅度不会超过它，Lipschitz连续条件得以满足。**具体在算法实现中，只需要每次更新完 $w$ 后把它clip回这个范围就可以了。此为止，我们可以构造一个含参数 $w$ 、最后一层不是非线性激活层的判别器网络 $f_w$ ，在限制 $w$ 不超过某个范围的条件下，使得
$$
L=\mathbb{E}_{x \sim P_r}\left[f_w(x)\right]-\mathbb{E}_{x \sim P_g}\left[f_w(x)\right] \text { (公式15) }
$$
尽可能取到最大，此时 $L$ 就会近似真实分布与生成分布之间的Wasserstein距离（忽略常数倍数 $K$ ）。注意原始GAN的判别器做的是真假二分类任务，所以最后一层是sigmoid，但是现在WGAN 中的判别器 $f_w$ 做的是近似拟合Wasserstein距离，属于回归任务，所以要把最后一层的sigmoid 拿掉。
接下来生成器要近似地最小化Wasserstein距离，可以最小化 $L$ ，由于Wasserstein距离的优良性 质，我们不需要担心生成器梯度消失的问题。再考虑到 $L$ 的第一项与生成器无关，就得到了 WGAN的两个loss。
$$
-\mathbb{E}_{x \sim P_g}\left[f_w(x)\right] \text { (公式16，WGAN生成器loss函数) }
$$

$$
\mathbb{E}_{x \sim P_g}\left[f_w(x)\right]-\mathbb{E}_{x \sim P_r}\left[f_w(x)\right](公式17， WGAN判别器loss函数)
$$

公式15是公式17的反，可以指示训练进程，其数值越小，表示真实分布与生成分布的 Wasserstein距离越小，GAN训练得越好。

Algorithm 1 WGAN, our proposed algorithm. All experiments in the paper used the default values $\alpha=0.00005, c=0.01, m=64, n_{\text {critic }}=5$.
Require: : $\alpha$, the learning rate. $c$, the clipping parameter. $m$, the batch size. $n_{\text {critic }}$, the number of iterations of the critic per generator iteration.
Require: : $w_0$, initial critic parameters. $\theta_0$, initial generator's parameters.
$\begin{array}{cc}\text { 1: } & \text { while } \theta \text { has not converged do } \\ \text { 2: } & \text { for } t=0, \ldots, n_{\text {critic }} \text { do } \\ \text { 3: } & \text { Sample }\left\{x^{(i)}\right\}_{i=1}^m \sim \mathbb{P}_r \text { a batch from the real data. } \\ 4: & \text { Sample }\left\{z^{(i)}\right\}_{i=1}^m \sim p(z) \text { a batch of prior samples. } \\ \text { 5: } & g_w \leftarrow \nabla_w\left[\frac{1}{m} \sum_{i=1}^m f_w\left(x^{(i)}\right)-\frac{1}{m} \sum_{i=1}^m f_w\left(g_\theta\left(z^{(i)}\right)\right)\right] \\ \text { 6: } & w \leftarrow w+\alpha \cdot \operatorname{RMSProp}\left(w, g_w\right) \\ 7: & w \leftarrow \operatorname{clip}(w,-c, c) \\ \text { 8: } & \text { end for } \\ 9: & \text { Sample }\left\{z^{(i)}\right\}_{i=1}^m \sim p(z) \text { a batch of prior samples. } \\ \text { 10: } & g_\theta \leftarrow-\nabla_\theta \frac{1}{m} \sum_{i=1}^m f_w\left(g_\theta\left(z^{(i)}\right)\right) \\ \text { 11: } & \theta \leftarrow \theta-\alpha \cdot \text { RMSProp }\left(\theta, g_\theta\right) \\ \text { 12: } & \text { end while }\end{array}$

上文说过，WGAN与原始GAN第一种形式相比，只改了四点:
- 判别器最后一层去掉sigmoid
- 生成器和判别器的loss不取log
- 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数 $c$
- 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行

前三点都是从理论分析中得到的，已经介绍完毕；第四点却是作者从实验中发现的，属于trick，相对比较“玄”。

- "注意原始GAN的判别器做的是真假二分类任务，所以最后一层是sigmoid，但是现在WGAN中的判别器做的是近似拟合Wasserstein距离，属于回归任务，所以要把最后一层的sigmoid拿掉。"

- 使用Adam，判别器的loss有时候会崩掉，当它崩掉时，Adam给出的更新方向与梯度方向夹角的cos值就变成负数，更新方向与梯度方向南辕北辙，这意味着判别器的loss梯度是不稳定的，所以不适合用Adam这类基于动量的优化算法。作者改用RMSProp之后，问题就解决了，因为RMSProp适合梯度不稳定的情况。

#### 4.2.4 补充

- 判别器所近似的Wasserstein距离能 够用来指示单次训练中的训练进程，这个没错；接着作者又说它可以用于比较多次训练进程，指引调参，我倒是觉得需要小心些。比如说我下次训练时改了判别器的层数、节点数等超参，判别器的 拟合能力就必然有所波动，再比如说我下次训练时改了生成器两次迭代之间，判别器的迭代次数， 这两种常见的变动都会使得Wasserstein距离的拟合误差就与上次不一样。那么这个拟合误差的变动究竟有多大，或者说不同的人做实验时判别器的拟合能力或迭代次数相差实在太大，那它们之间 还能不能直接比较上述指标，我都是存疑的。
- 相比于判别器迭代次数的改变，对判别器架构超参的改变会直接影响到对应的Lipschitz常数 $K$ ，进而改变近似Wasserstein距离的倍数，前后两轮训练的指标就肯定不能比较了，这是需要在实际应用中注意的。对此我想到了一个工程化的解决方式，不是很优雅: 取同样一对生成分布和真实分布，让前后两个不同架构的判别器各自拟合到收敛，看收敛到的指标差多少倍，可以近似认为是后面的 $K_2$ 相对前面 $K_1$ 的变化倍数，于是就可以用这个变化倍数校正前后两轮训练的指标。

## 参考

[令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)