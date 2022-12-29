### DCGAN

稳定的深度卷积gan体系结构指南

•用strided convolutions (discriminator)和fraction -strided替换任何池化层

旋转(发电机)。

在生成器和鉴别器中都使用batchnorm。

•移除深层架构的全连接隐藏层。

•在生成器中使用ReLU激活所有层，除了输出，它使用Tanh。

•在所有层的鉴别器中使用LeakyReLU激活