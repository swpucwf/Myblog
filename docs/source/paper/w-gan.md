### W-GAN

- 彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
- 基本解决了collapse mode的问题，确保了生成样本的多样性
- 训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高（如题图所示）
- 以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到

[项目实现链接](https://github.com/martinarjovsky/WassersteinGAN)

### 原始GAN问题?

- 原始GAN损失函数，原始GAN中判别器要最小化如下损失函数，尽可能把真实样本分为正例，生成样本分为负例:

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