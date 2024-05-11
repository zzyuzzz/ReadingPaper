# Mutual Information Neural Estimation

## 摘要
文章认为高维连续随机变量之间的互信息可以通过在神经网络上梯度下降优化得到。
文中提出了互信息估计器(Mutual Information Neural Estimator),它在维度和
样本大小上都是可伸缩的，可以通过反向传播训练的，并且具有很强的一致性（一致性？？？什么意思？？？）。文章提出了一些可以使用MINE来最小化或最大化互信息的
应用。作者应用MINE来改进对抗训练的生成模型。还使用MINE来实现信息瓶颈，将其应用到监督分类中：结果表明，在这些设置中，灵活性和性能有很大的改善。


#### 背景知识补充1
与相关系数相比，**互信息可以捕获非线性关系**。
$$
\begin{align}
H(X) &= -\sum_x p(x)log(p(x))\\
H(X|Z) &= -\sum_z p(z)\sum_x p(x|z)log(p(x|z))\\
I(X, Z) &= \sum_x\sum_z p(x, z)log\frac{p(x, z)}{p(x)p(z)}\\
        &= D_{KL}(p(x, z)||p(x)p(z)) \\
        &= H(X)-H(X|Z)\\
        &= H(Z)-H(Z|X)
\end{align}
$$
$H$是信息熵，$I$是互信息。

## KL散度的对偶(Dual)表示

### Donsker-Varadhan represention
+ 理论1(Q,P指分布，dQ,dP密度函数)：
$$
D_{KL}(P||Q)=\sup\limits_{T:\Omega\rightarrow R}\mathbb{E}_P{[T]}-log(\mathbb{E}_Q[e^T])
$$
- 证明理论1:令$dG=\frac{1}{Z}e^TdQ$，其中$Z=\mathbb{E}_Q[e^T]$.
$$
\begin{align}
\frac{dG}{dQ}&=\frac{1}{Z}e^T\\
\mathbb{E}_P[\mathbb{E}_Q]&=\mathbb{E}_Q\\
\mathbb{E}_P[log\frac{dG}{dQ}]&=\mathbb{E}_P[T]-log(Z)\\
\Delta :&=D_{KL}(P||Q)-(\mathbb{E}_P[T]-log(Z))\\
 &= \mathbb{E}_P\left[log\frac{dP}{dQ}-log\frac{dG}{dQ}\right]\\
        &= \mathbb{E}_P\left[log \frac{dP}{dG}\right]=D_{KL}(P||G)\ge 0\\
D_{KL}(P||Q)&\ge \mathbb{E}_P{[T]}-log(\mathbb{E}_Q[e^T])
\end{align}
$$
可以看到，当$G==P$时，取等号，即边界是贴近的，$T^*=log\frac{dP}{dQ}+C$。

