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
最大化编码表示之间的互信息。
### 对比预测编码CPC
Figure 1中展示了对比预测编码（CPC）模型架构，首先，**$g_{enc}$表示一个非线性编码器，它将观测量$x_t$的输入序列映射成潜在表示$z_t=g_{enc}(x_t)$，具有较低的时间分辨率**。然后，自回归模型$g_ar$总结所有潜在空间中的$z_{\le t}$并产生一个上下文的潜在表示$c_t=g_{ar}(z_{\le t})$。

不直接建模$p_k(x_{t+k}|c_t)$来预测$x_{t+k}$，而是建模$x_{t+k}$和$c_t$之间留存互信息的密度比率。

$$
f_k(x_{t+k},c_t)\propto \frac{p(x_{t+k}|c_t)}{p(x_{t+k})}
$$
注意到密度比f可以非正规化为(不必整合到1)。文章中使用了一个简单的对数双线性模型来建模它：
$$
f_k(x_{t+k},c_t)=exp(z_{t+k}^TW_kc_t)
$$
$W_kc_t$用于每一步k都有一个不同的$W_k$进行预测。或者，可以使用非线性网络或递归神经网络。

### InfoNCE Loss
给定N个随机样本集合X = { x1，.. xN }，其中1个来自$p(x_{t+k}|c_t)$的正样本，N - 1个来自"提议"分布$p(x_{t+k})$的负样本。
$$
\mathcal{L}_N=-\mathbb{E}\left[log\frac{f_k(x_{t+k},c_t)}{\sum_{x_j\in \mathbf{X}}f_k(x_j,c_t)}\right]
$$
优化这个损失函数将使得$f_k$估计密度比率。
将这种损失的最优概率记为$p( d = i | X , c_t)$，其中[ d = i]是样本xi为"正"样本的指标。样本xi是由条件分布$p(x_{t+k}|c_t)$而不是建议分布$p(x_{t+k})$得出的概率如下：
$$
p(d=i|\mathbf{X},c_t)=\frac{p(x_i|c_t)\prod_{l\neq i}p(x_l)}{\sum_{j=1}^N p(x_j|c_t)\prod_{l\neq j}p(x_l)}=\frac{\frac{p(x_i|c_t)}{p(x_i)}}{\sum_{j=1}^N \frac{p(x_j|c_t)}{p(x_j)}}
$$

$$
I(x_{t+k},c_t)\ge log(N)-\mathcal{L}_N
$$
N越大，越贴近。

##### prove
$$
\begin{align}
\mathcal{L}_N^{opt} &= -\mathbb{E}_{X}log\left[\frac{\frac{p(x_i|c_t)}{p(x_i)}}{\frac{p(x_i|c_t)}{p(x_i)}+\sum_{x_j\in X_{neg}} \frac{p(x_j|c_t)}{p(x_j)}}\right]\\

&=\mathbb{E}_{X}log\left[1+\frac{p(x_i)}{p(x_i|c_t)}\sum_{x_j\in X_{neg}}\frac{p(x_j|c_t)}{p(x_j)}\right]\\

&\approx \mathbb{E}_{X}log\left[1+\frac{p(x_i)}{p(x_i|c_t)}(N-1)\mathbb{E}_{x_j}\frac{p(x_j|c_t)}{p(x_j)}\right]\\

&=\mathbb{E}_{X}log\left[1+\frac{p(x_i)}{p(x_i|c_t)}(N-1)\right]\\
&\ge \mathbb{E}_{X}log\left[\frac{p(x_i)}{p(x_i|c_t)}(N-1)\right]\\
&= -I(x_i,c_t)+log(N-1)
\end{align}
$$

对于(5)(6)原论文为：
$$
\begin{align}
&=\mathbb{E}_{X}log\left[1+\frac{p(x_i)}{p(x_i|c_t)}(N-1)\right]\\
&\ge \mathbb{E}_{X}log\left[\frac{p(x_i)}{p(x_i|c_t)}N\right]\\
&= -I(x_i,c_t)+log(N)
\end{align}
$$
我认为是在最优化条件下，$p(x_i)\le p(x_i|c_t)$。

InfoNCE也与MINE（最大互信息估计）相关，记$f(x,c)=e^{F(x, c)}$,则：
$$
\begin{align}
\mathbb{E}_X\left[log\frac{f(x,c)}{\sum_{x\in \mathbf{X}}f(x,c)}\right]\
&=\mathbb{E}_{(x,c)}\left[F(x, c)\right]-\mathbb{E}_{(x,c)}\left[log\sum_{x_j\in X}e^{F(x_j, c)}\right]\\
&=\mathbb{E}_{(x,c)}\left[F(x, c)\right]-\mathbb{E}_{(x,c)}\left[log\left(e^{F(x, c)}+\sum_{x_j\in X_{neg}}e^{F(x_j, c)}\right)\right]\\
&\le \mathbb{E}_{(x,c)}\left[F(x, c)\right]-\mathbb{E}_{c}\left[log\left(\sum_{x_j\in X_{neg}}e^{F(x_j, c)}\right)\right]\\
\end{align}
$$