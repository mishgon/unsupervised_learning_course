# Description

VICReg (https://arxiv.org/abs/2105.04906) is one of the SotA SSL methods.

Theoretically, VICReg's stochastic gradient is biased. There is a simple way how to fix it, described below.

## How to make VICReg's stochastic gradient unbiased?

$X \sim p(x)$ — a random image from the dataset.\
$T \sim \mathcal{T}$ — a random augmentation.\
$f_\theta$ — a neural net.\
$Z = f_\theta(T(X))$ — a random representation.\

Consider the following optimization problem (it is very similar to VICReg):

$$
\underset{X \sim p(x), T_1, T_2 \sim \mathcal{T}}{\mathbb{E}}\|f_\theta(T_1(X)) - f_\theta(T_2(X))\|^2 - \lambda \|\text{cov}(Z) - I\|_F^2 \to \min_\theta
$$

In order to obtain an unbiased estimate for the first term you sample a batch of images $X_1, \ldots, X_n \overset{i.i.d.}{\sim} p(x)$, sample two random augmentations for each image and compute the MSE loss.

In order to obtain an unbiased estimate for the second term, we use two independent batches of images $X_1, \ldots, X_{n/2}$ and $X_{n/2 + 1}, \ldots, X_n$, obtain their representations $Z_1, \ldots, Z_{n/2}$ and $Z_{n/2 + 1}, \ldots, Z_n$ and use each batch to obtain independent estimates of covariance matrix $\text{cov}(Z)$:

$$
\text{cov}_1 = \frac{1}{n/2 - 1}\sum_{i = 1}^{n/2}Z_iZ_i^T,
$$
and
$$
\text{cov}_2 = \frac{1}{n/2 - 1}\sum_{i = n/2 + 1}^{n}Z_iZ_i^T
$$

Then $\langle \text{cov}_1 - I, \text{cov}_2 - I \rangle$ is an unbiased estimate of $\|\text{cov}(Z) - I\|_F^2$ because
$$ \mathbb{E}\langle \text{cov}_1 - I, \text{cov}_2 - I \rangle = \langle \mathbb{E}\text{cov}_1 - I, \mathbb{E}\text{cov}_2 - I \rangle = \\ = \langle \text{cov}(Z) - I, \text{cov}(Z) - I \rangle = \|\text{cov}(Z) - I\|_F^2. $$
# Roadmap

1. Implement standard VICReg as a baseline.
2. Implement unbiased VICReg.
3. Compare them on linear probing on CIFAR10.