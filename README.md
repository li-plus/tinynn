# TinyNN

An educational PyTorch-like neural network framework based on NumPy.

## Getting Started

Clone this repo and run the mnist example.

```sh
git clone https://github.com/li-plus/tinynn.git && cd tinynn
python3 -m examples.mnist
```

## Back Propagation

A deep neural network model is a function with huge amount of parameters that transforms the inputs (images/texts/videos) to the outputs (probabilities/embeddings). These models are usually made up of many basic layers or operators, such as dense layers and activation functions. These basic elements form a computational graph, representing their execution order and data flow.

Back propagation is an efficient algorithm to train a feedforward neural network, where the graph does not have any cycle. During training, the model first computes the outputs from the inputs by traversing forward the graph, and evaluates the error between the predicted outputs and the ground truth labels using a loss function. Then it computes the gradient of error with respect to model outputs, and back propagates the gradient to all the model parameters. Finally, the optimizer push the parameters a little bit towards the gradient descending direction, in order to decrease the error.

Formally, denote the model input as $X$ and the ground truth label as $Y$. The model has $n$ layers, and each layer has its own weight $W_i$ and forward function $f_i$.

$$
X_i = \begin{cases}
X, & i = 1 \\
f_{i-1}(X_{i-1}, W_{i-1}), & 1 < i \le n
\end{cases}
$$

The loss $l$ is evaluated by an arbitrary criterion function $C$.

$$
l = C(X_n, Y)
$$

Using the chain rule, we may back propagate the gradient of $l$ to all the model weights in a reversed order of forward computation.

$$
\frac{\partial l}{\partial X_i} = \frac{\partial l}{\partial X_{i+1}}\frac{\partial X_{i+1}}{\partial X_i}, \quad
\frac{\partial l}{\partial W_i} = \frac{\partial l}{\partial X_{i+1}}\frac{\partial X_{i+1}}{\partial W_i}, \quad
i = n-1, n-2,\cdots,1
$$

Note that each layer only needs to focus on the local gradient introduced by itself. On layer $i$, the local gradient $\partial X_{i+1}/\partial X_{i}$ and $\partial X_{i+1}/\partial W_{i}$ depends only on the layer function $f_i$. Given the gradient with respect to the layer output (output gradient) $\partial l/\partial X_{i+1}$, the gradient with respect to its input or weight (input gradient) is easily computed using the local graident. Meanwhile, the input gradient of layer $i$ is also the output gradient of the previous layer $i-1$, so all graidents are known after traversing backward the graph only once.

In general, the local gradient for a vector function $\mathbf{y} = f(\mathbf{x})$ is called the Jacobian matrix.

$$
J = \frac{\partial (y_1,\cdots,y_m)}{\partial (x_1,\cdots,x_n)}
= \left(\begin{matrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots &  \vdots \\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{matrix}\right)
$$

For a scalar function $l=g(\mathbf{y})$ depending on $\mathbf{y}$, the Jacobian matrix maps the output gradient $\partial l/\partial \mathbf{y}$ to the input gradient $\partial l/\partial \mathbf{x}$.

$$
\frac{\partial l}{\partial{\mathbf{x}}} = J^T \frac{\partial l}{\partial \mathbf{y}}
$$

## Layer Gradient

**ReLU**

ReLU is an element-wise operator that clips the negative value to zero.

$$
y = \text{ReLU}(x) = \max(x, 0)
$$

The local gradient is as below. Note that on $x=0$, the gradient is undefined.

$$
\frac{\partial y}{\partial x} = \begin{cases}
1, & x > 0 \\
0, & x < 0
\end{cases}
$$

**Softmax**

The softmax function rescales N unbounded values into probabilities between \[0,1\] that sum up to 1.

$$
y_i = \text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^N \exp(x_j)}, \quad i=1,2,\cdots,N
$$

The element at i-th row and j-th column of the transposed Jacobian $J^T$ is:

$$
\frac{\partial y_j}{\partial x_i} = \begin{cases}
-y_i y_j, & i\ne j \\
y_i - y_i y_j, & i = j
\end{cases}
$$

In matrix form, the Jacobian could be written as:

$$
J^T = \text{diag}(\mathbf{y}) - \mathbf{y}\mathbf{y}^T
$$

The Jacobian-vector product (jvp) can be further simplified as below, where $\odot$ is the element-wise multiplication operator.

$$
J^T \mathbf{v} = \mathbf{y} \odot \mathbf{v} - \mathbf{y}(\mathbf{y}^T\mathbf{v})
$$

Let $\mathbf{v}$ be the gradient of $l$ with respect to $\mathbf{y}$, then the jvp becomes the gradient of $l$ with respect to $\mathbf{x}$.

**Matmul**

Denote the weight matrix as $W \in \mathbb{R}^{M\times K}$, input matrix as $X \in \mathbb{R}^{K\times N}$, layer output as $Y=WX \in \mathbb{R}^{M\times N}$, and loss as $l = C(Y)$.

We start at the derivative of a specific element. Since $l$ is a function of $y_{11},\cdots,y_{MN}$, the total derivative of $l$ with respect to $w_{ij}$ is:

$$
\begin{aligned}
\frac{\partial l}{\partial w_{ij}} &= \sum_{m=1}^{M}\sum_{n=1}^{N}\frac{\partial l}{\partial y_{mn}}\frac{\partial y_{mn}}{\partial w_{ij}} \\
&= \sum_{m=1}^{M}\sum_{n=1}^{N}\frac{\partial l}{\partial y_{mn}}\frac{\partial}{\partial w_{ij}}\left(\sum_{k=1}^K w_{mk} x_{kn}\right) \\
&= \sum_{n=1}^N \frac{\partial l}{\partial y_{in}} x_{jn} \\
&= \sum_{n=1}^N \frac{\partial l}{\partial y_{in}} x^T_{nj}
\end{aligned}
$$

In matrix form, the derivative of $l$ with respect to $W$ could be written as:

$$
\frac{\partial l}{\partial W} = \frac{\partial l}{\partial Y}X^T
$$

Likewise, the derivative of $l$ with respect to $X$ is:

$$
\frac{\partial l}{\partial X} = W^T\frac{\partial l}{\partial Y}
$$

**Convolution**

For an input matrix $X\in\mathbb{R}^{M\times N}$ and convolutional kernel $W\in\mathbb{R}^{P\times Q}$. The 2-dimensional convolution is described as $Y=W*X$, where the result $Y$ is $(M+1-P)\times(N+1-Q)$ in shape. 

Each element $y_{ij}$ is computed as:

$$
y_{ij} = \sum_{p=1}^P\sum_{q=1}^Q w_{pq} x_{i+p-1,j+q-1}, \quad 1\le i\le M+1-P, \ 1\le j\le N+1-Q
$$

For a scalar function $l=C(Y)$ related to $Y$, we need to figure out the derivative of $l$ with respect to $X$ and $W$. Still starting at a specific element.

$$
\begin{aligned}
\frac{\partial l}{\partial x_{ij}} &= \sum_{m=1}^{M+1-P}\sum_{n=1}^{N+1-Q}\frac{\partial l}{\partial y_{mn}}\frac{\partial y_{mn}}{\partial x_{ij}} \\
&= \sum_{m=1}^{M+1-P}\sum_{n=1}^{N+1-Q}\frac{\partial l}{\partial y_{mn}}\frac{\partial}{\partial x_{ij}}\left(\sum_{p=1}^P\sum_{q=1}^Q w_{pq} x_{m+p-1,n+q-1}\right) \\
&= \sum_{m=i+1-P}^{i}\sum_{n=j+1-Q}^{j}\frac{\partial l}{\partial y_{mn}}w_{i+1-m,j+1-n}
\end{aligned}
$$

Let $p=m-i+P$ and $q=n-j+Q$, so that the summation starts from 1. Further denote $W^R$ as the reversed matrix of $W$, where $w^R_{pq} = w_{P+1-p,Q+1-q}$. Then we have:

$$
\frac{\partial l}{\partial x_{ij}}
= \sum_{p=1}^{P}\sum_{q=1}^{Q}\frac{\partial l}{\partial y_{i+p-P,j+q-Q}}w_{P+1-p,Q+1-q}
= \sum_{p=1}^{P}\sum_{q=1}^{Q}\frac{\partial l}{\partial y_{i+p-P,j+q-Q}} w^R_{pq}
$$

The above equation shows that each element of $\partial l/\partial X$ can be computed by convolving the corresponding block of $\partial l/\partial Y$ and $W^R$. After making a full padding for $\partial l/\partial Y$, the derivate for $X$ can be described in matrix form. This operation is also called the deconvolution or transposed convolution.

$$
\frac{\partial l}{\partial X} = W^R * \frac{\partial l}{\partial Y}
$$

Similarly, the derivative of $l$ with respect to $W$ is as follows.

$$
\frac{\partial l}{\partial W} = \frac{\partial l}{\partial Y} * X
$$
