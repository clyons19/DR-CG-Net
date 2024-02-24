# Deep Regularized Compound Gaussian Network (DR-CG-Net) for Solving Linear Inverse Problems
All rights to this code are reserved. Commercial and research licenses are available on request. Please contact aly.hoeher@colostate.edu with any requests for licenses.


An implementation of Projected Gradient Descent DR-CG-Net (PGD DR-CG-Net) and Iterative Shrinkage and Thresholding Algorithm DR-CG-Net (ISTA DR-CG-Net) from
  1. Lyons C., Raj R. G., & Cheney M. (2023). "Deep Regularized Compound Gaussian Network for Solving Linear Inverse Problems." [arXiv preprint arXiv:2311.17248.](https://arxiv.org/abs/2311.17248#:~:text=Deep%20Regularized%20Compound%20Gaussian%20Network%20for%20Solving%20Linear%20Inverse%20Problems,-Carter%20Lyons%2C%20Raghu&text=Incorporating%20prior%20information%20into%20inverse,facilitating%20robust%20inverse%20problem%20solutions.)

  2. Lyons C., Raj R. G., & Cheney M. (2024). "Deep Regularized Compound Gaussian Network for Solving Linear Inverse Problems," in IEEE Transactions on Computational Imaging. in-press.


### Implementation Overview:
The DR-CG-Net method is an algorithm-unrolled deep neural network (DNN) that solves linear inverse problems to the forward measurement model
$$y = \Psi\Phi c + \nu \equiv Ac+\nu$$
where $\Psi\in\mathbb{R}^{m\times n}$ is a measurement matrix, $\Phi\in\mathbb{R}^{n\times n}$ is a change-of-basis dictionary, $x = \Phi c$ is an underlying signal of interest (i.e. an image) for $c\in\mathbb{R}^n$ the change-of-basis coefficients, $\nu\in\mathbb{R}^m$ is additive white Gaussian noise, and $y\in\mathbb{R}^m$ is the measurement/observation of $x$. As DR-CG-Net is an algorithm unrolled DNN, we first construct a compound Gaussian-based iterative algorithm to solve the above linear inverse problem and then unroll this algorithm into the DR-CG-Net framework.

#### Iterative Algorithm 
Often an iterative algorithm is used to solve inverse problems where statistical prior information on $c$ (or $x$) is incorporated. We use a compound Gaussian (CG) prior that decomposes $c = z\odot u$ for two independent random vectors $z$ and $u$ where $u$ is Gaussian and $z$ is positive and non-Gaussian. The maximum a posteriori (MAP) estimate of $z$ and $u$ from $y = A(z\odot u)+\nu$ is given by
$$[\hat{z}, \hat{u}] = \arg\min_{[z, u]} F(z,u)$$
for
$$F(z,u) \coloneqq  \frac{1}{2} ||y - A(z\odot u)||_2^2 + \frac{1}{2} u^TP_u^{-1} u + R(z)$$
where $P_u$ is the covariance matrix of $u$ and $R$ is a regularization function equal to the negative log prior of $z$ (which can be specified on a problem-specific basis). For notation, we write
$$A_z = A\text{diag}(z).$$
We consider an alternating block-coordinate descent to approximate $\hat{z}$ and $\hat{u}$. Thus, on iteration $k$
$$u_k = \arg\min_u F(z_k, u) \coloneqq P_u A_z^T(I+A_zP_uA_z^T)^{-1}y$$
(note for larger signal sizes where the above inverse is intractable to solve, we instead approximate the above inverse with Nesterov accelerated gradient descent steps) and
$$z_k = J \text{ applications of a descent function  } g(z,u):\mathbb{R}^n\times\mathbb{R}^n\to\mathbb{R}^n.$$
Two descent functions we employ are
1. Projected Gradient Descent (PGD)
$$g(z,u) = P_Z(z - \eta \nabla_z F(z,u)) = P_Z(z - \eta(A_u^T(A_uz-y)+\nabla R(z)))$$
2. Iterative Shrinkage and Thresholding (ISTA)
$$g(z,u) = \text{prox}_{\eta R}(z -\eta A_u^T(A_uz-y)) \coloneqq \arg\min_t \frac{1}{2}||t - (z-\eta A_u^T(A_uz-y))||_2^2+\eta R(t)$$

#### Unrolled DR-CG-Net
