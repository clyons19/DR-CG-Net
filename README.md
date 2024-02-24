# Deep Regularized Compound Gaussian Network (DR-CG-Net) for Solving Linear Inverse Problems
All rights to this code are reserved. Commercial and research licenses are available on request. Please contact aly.hoeher@colostate.edu with any requests for licenses.


An implementation of Projected Gradient Descent DR-CG-Net (PGD DR-CG-Net) and Iterative Shrinkage and Thresholding Algorithm DR-CG-Net (ISTA DR-CG-Net) from
  1. Lyons C., Raj R. G., & Cheney M. (2023). "Deep Regularized Compound Gaussian Network for Solving Linear Inverse Problems." [arXiv preprint arXiv:2311.17248.](https://arxiv.org/abs/2311.17248#:~:text=Deep%20Regularized%20Compound%20Gaussian%20Network%20for%20Solving%20Linear%20Inverse%20Problems,-Carter%20Lyons%2C%20Raghu&text=Incorporating%20prior%20information%20into%20inverse,facilitating%20robust%20inverse%20problem%20solutions.)

  2. Lyons C., Raj R. G., & Cheney M. (2024). "Deep Regularized Compound Gaussian Network for Solving Linear Inverse Problems," in IEEE Transactions on Computational Imaging. in-press.


### Implementation Overview:
The DR-CG-Net method is an algorithm-unrolled deep neural network that solves linear inverse problems to the forward measurement model
\\[
y = \Psi\Phi c + \nu \equiv Ac+\nu
\\]
where $\Psi\in\mathbb{R}^{m\times n}$ is a measurement matrix, $\Phi\in\mathbb{R}^{n\times n}$ is a change-of-basis dictionary, $x = \Phi c$ is an underlying signal of interest (i.e. an image) for $c\in\mathbb{R}^n$ the change-of-basis coefficients, $\nu\in\mathbb{R}^m$ is additive white Gaussian noise, and $y\in\mathbb{R}^m$ is the measurement/observation of $x$. To solve inverse problems, statistical prior information on $c$ (or $x$) is typically incorporated for which we use a compound Gaussian (CG) prior. Using the CG prior we decompose $c = z\odot u$ for two independent random vectors $z$ and $u$ where $u$ is Gaussian and $z$ is positive and non-Gaussian. Thus, the maximum a posteriori (MAP) estimate of $z$ and $u$ from $y = A(z\odot u)+\nu$ is given by
\\[
\arg\min_{[z, u]} \frac{1}{2} ||y - A(z\odot u)||_2^2 + \frac{1}{2} u^TP_u^{-1} u + R(z)
\\]
where $P_u$ is the covariance matrix of $u$ and $R$ is a regularization function equal to the negative log prior of $z$.
