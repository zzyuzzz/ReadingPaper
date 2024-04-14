# 对比预测编码表示学习

## 引言
![CPC Overview](picture/Overview%20of%20Contrastive%20Predictive%20Coding.png)
文章主要提出如下几点：首先**将高维数据压缩到更加紧凑的潜在嵌入（latent embdding）空间**，在这个空间中条件预测更容易被建模。第二，在这个**潜在空间中使用自回归模型，以对未来的多个步骤做预测**。在最后，**依靠噪声对比估计[文献](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)的损失函数，以类似的方式用于学习自然语言模型中的词嵌入，从而允许整个模型端到端的训练**。我们将得到的模型，对比预测编码( CPC )应用于广泛不同的数据模态，图像，语音，自然语言和强化学习，并表明相同的机制在这些领域的每个领域学习有趣的高级信息，表现优异。

## 对比预测编码（CPC，Constrastive Predicting Coding）
### 动机和直觉
模型背后的主要直觉是学习编码(高维)信号不同部分之间潜在共享信息的表示。同时它丢弃了更局部的低级信息和噪声。在时间序列和高维建模中，使用下一步预测的方法利用了信号的局部平滑性。在未来进一步预测时，共享的信息量变得更低，模型需要推断更多的全局结构。这些跨越多个时间步的"慢特征" 往往是更有趣的(例如,语音中的音素和语调,图像中的物体,或书籍中的故事情节。)。

**直接建模$p(x|c)$计算代价一般非常大，对于提取$x$和$c$之间的共享信息而言，可能不是最优的。当预测未来信息时，我们以最大限度保留原始信号$x$和$c$的互信息的方式将目标x (未来)和上下文c (当前)编码成一个紧凑的分布式向量表示(凭借非线性学习映射)**

这个方式定义为：

$$
I(x;c)=\sum_{x,c}p(x,c)log\frac{p(x,c)}{p(x)p(c)}=\sum_{x,c}p(x,c)log\frac{p(x|c)}{p(x)}
$$
