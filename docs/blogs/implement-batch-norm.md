---
# Page settings
layout: default
keywords:
comments: false

# Hero section
title: Implement Batch Normalization
description:

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content:
        url: '#'
    next:
        content:
        url: '#'
---

### Batch Normalization

One preprocessing technique widely used across every Machine Learning algorithm is to normalize the input features to have zero mean and unit variance. In practice, this technique tends to make algorithms that are optimized with gradient descent converge faster to the solution.

One way we can look at deep neural networks is as stacks of different models (layers), where the output of one model is the input of the next.
And so the question is: can't we normalize the output of each layer? That is what proposed with the Batch Normalization layer.

In order to be able to introduce the normalization in the neural network's training pipeline, it should be fully differentiable (or at least almost everywhere differentiable like the ReLU function). The good news is that it is, but let's see a little example.

Assume the output of an hidden layer $$X$$ is an $$(N,D)$$ matrix, where $$N$$ is the number of examples present in the batch and $$D$$ is the number of hidden units. We start by normalizing $$X$$:

$$ \hat{X} = \frac{X - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}\ ,$$

where $$\mu_B$$ is the mean of the batch, $$\sigma_B^2$$ is the variance and $$\epsilon$$ is a small number to guarantee that there won't be any division by 0.

As $$\mu_B$$ and $$\sigma_B^2$$ are differentiable, we can see that $$ \hat{X}$$ is also differentiable, and so we are good to go.

Then, the authors did something really clever. They realized that by normalizing the output of a layer they could limit its representational power and, therefore, they wanted to make sure that the Batch Norm layer could fall back to the identity function.

$$ y = \gamma \hat{X} + \beta $$

Note that when $$\gamma = \sqrt{\sigma_B^2 + \epsilon}$$ and $$\beta = \mu_B$$ the Batch Norm simply outputs the previous layer's activations. The network has the ability to ignore the Batch Norm layer if that is the optimal thing to do. As you might have guessed by now, $$\gamma$$ and $$\beta$$ are learnable parameters that are initialized with $$\gamma = 1$$ and $$\beta = 0$$.

The authors claim that Batch Norm has several nice properties such as:

- **Reduce internal covariate shift**: define internal covariate shift as "the change in the distribution of the network activations due to the change in network parameters during training". Imagine how hard it is to learn a model when the inputs are always changing. With Batch Norm, the next layer can expect the same input distribution at each iteration.
- **Regularization effect**: It is claimed that with Batch Norm we could disregard or reduce the strength of Dropout. The value given to one training example depends on the other examples in the batch, since the mean and variance are computed on the batch level and, therefore, it is not deterministic. This property can increase the generalization of the network.
- **Reduce probability of vanishing or exploding gradients**: As the Batch Norm layer is placed prior to the non-linearity, it prevents the training to get stuck on saturated areas of non-linearities, solving the issue of vanishing gradients. Also, it makes the network more robust to the scale of the parameters, solving the issue of exploding gradients.

In practice, this allows us to increase the learning rate and train the network with a lot less iterations and, as this was not enough, get better results.

### Forward Pass

The math looks very simple, let's try to implement the forward pass in Python:

```python
def batchnorm_forward(x, gamma, beta, eps):
  mu = np.mean(x)
  var = np.var(x)
  xhat = (x - mu) / np.sqrt(var + eps)
  y = gamma * xhat + beta
  return y
```

Easy, right? Now what about the backward pass? Not that easy...

Instead of analytically deriving the formula of the gradient, it is easier to break the formula into atomic operations where we can directly compute the gradient, and then use the chain rule to get to the gradient of $$X$$, $$\gamma$$ and $$\beta$$. This can be represented as a graph, known as the computational graph, where nodes are mathematical operations and the edges connect the output of one node to the input of another:

{% include image.html description="" image="blogs/batch-norm/batch-norm-computational-graph.png" caption="false"%}

We follow the dark arrows for the forward pass and then we backpropagate the error using the red ones. The dimension of the output of each node is displayed on top of each node. For instance, $$X$$ is an $$(N, D)$$ matrix and the output of the node that computes the mean of each dimension of $$X$$ is a vector with $$D$$ elements.

The forward pass then becomes:

```python
def batchnorm_forward(x, gamma, beta, eps):

  mu = np.mean(x, axis=0)
  xcorrected = x - mu
  xsquarred = xcorrected**2
  var = np.mean(xsquarred, axis=0)
  std = np.sqrt(var + eps)
  istd = 1 / std
  xhat = xcorrected * istd
  y = xhat * gamma + beta

  cache = (gamma, xhat, xcorrected, istd, std)
  return y, cache
```

### Backward Pass

How do we get the gradient of $$X$$, $$\gamma$$ and $$\beta$$ with respect to the loss $$l$$? We use the chain rule to transverse the computational graph on the opposite direction (red arrows).

Let's start with the computation of the $$\frac{\partial l}{\partial \beta}$$ to get an idea of how it is done:

$$ \frac{\partial l}{\partial \beta} = \frac{\partial l}{\partial y} * \frac{\partial y}{\partial \beta} $$

$$\frac{\partial l}{\partial y}$$ tells us how the loss of the entire network would grow/decrease if the output $$y$$ would increase a tiny amount. This value is computed by the layer that is on top of the Batch Norm and so it is given to us as an input of the backward pass. Now we need to compute $$\frac{\partial y}{\partial \beta}$$ :

$$ \frac{\partial y}{\partial \beta} = \frac{\partial (\hat{X} * \gamma + \beta)}{\partial \beta} = 1$$

If you increase $$\beta$$ by a tiny amount $$h$$, then $$y$$ is expected to become $$y + 1 * h$$ as well. That makes sense! But what about the loss?

$$ \frac{\partial l}{\partial \beta} = \frac{\partial l}{\partial y} * 1 = \frac{\partial l}{\partial y} $$

So the gradient of $$\beta$$ is simply the gradient that reaches the network. But wait, the dimension of $$\beta$$ is not the same as the dimension of $$y$$! Something is wrong, right?

In truth, it is not possible to sum two matrices with different sizes. This line `y = xhat * gamma + beta` should not work in the first place. What numpy is doing behind the scenes is called broadcasting. In this case, it will simply add $$\beta$$ to each line of the other matrix (it does the same thing to multiply $$\gamma$$ and $$\hat{X}$$). To arrive to the $$(D,)$$ dimensional array $$\frac{\partial l}{\partial \beta}$$  we just need to sum each line of $$\frac{\partial l}{\partial y}$$ :

```python
  dbeta = np.sum(dy, axis=0)
```

Here are the remaining partial derivatives:

$$ \frac{\partial l}{\partial \gamma} =
\frac{\partial l}{\partial y} * \frac{\partial y}{\partial \gamma} =
\frac{\partial l}{\partial y} * \hat{X} $$

$$ \frac{\partial l}{\partial \hat{X}} =
\frac{\partial l}{\partial y} * \frac{\partial y}{\partial \hat{X}} =
\frac{\partial l}{\partial y} * \gamma $$

$$ \frac{\partial l}{\partial istd} =
\frac{\partial l}{\partial \hat{X}} * \frac{\partial \hat{X}}{\partial istd} =
\frac{\partial l}{\partial \hat{X}} * (X - \mu_B) $$

$$ \frac{\partial l}{\partial \sigma} =
\frac{\partial l}{\partial istd} * \frac{\partial istd}{\partial \sigma} =
\frac{\partial l}{\partial istd} * \frac{-1}{\sigma^2}$$

$$ \frac{\partial l}{\partial \sigma^2} =
\frac{\partial l}{\partial \sigma} * \frac{\partial \sigma}{\partial \sigma^2} = \frac{\partial l}{\partial \sigma} * \frac{1}{2\sigma} $$

$$ \frac{\partial l}{\partial xsquarred} =
\frac{\partial l}{\partial \sigma^2} * \frac{\partial \sigma^2}{\partial xsquarred} =
\frac{\partial l}{\partial \sigma^2} * \frac{1}{N}$$

$$ \frac{\partial l}{\partial xcorrected} =
\frac{\partial l}{\partial \hat{X}} * \frac{\partial \hat{X}}{\partial xcorrected} + \frac{\partial l}{\partial xsquarred} * \frac{\partial xsquarred}{\partial xcorrected} = \\
= \frac{\partial l}{\partial \hat{X}} * \frac{1}{\sigma} +
\frac{\partial l}{\partial xsquarred} * 2 * xcorrected $$

$$ \frac{\partial l}{\partial \mu} =
\frac{\partial l}{\partial xcorrected} * \frac{\partial xcorrected}{\partial \mu} =
\frac{\partial l}{\partial xcorrected} * -1$$

$$ \frac{\partial l}{\partial X} =
\frac{\partial l}{\partial xcorrected} * \frac{\partial xcorrected}{\partial X} +
\frac{\partial l}{\partial \mu} * \frac{\partial \mu}{\partial X} = \\
= \frac{\partial l}{\partial xcorrected} * 1 +
\frac{\partial l}{\partial \mu} * \frac{1}{N} $$

Just as when we are doing the forward pass we use output of the one node the input of another node, in the backward pass we use the gradient of
one node to compute the gradient of the previous one. It might be clearer in python:

```python
def batchnorm_backward(dldy, cache):
  gamma, xhat, xcorrected, istd, std = cache
  N, D = dldy.shape
  # (D,)
  dbeta = np.sum(dldy, axis=0)
  dgamma = np.sum(xhat * dout, axis=0)
  # (N,D)
  dxhat = dldy * gamma
  # (D,)
  distd = np.sum(dxhat * xcorrected, axis=0)
  # (D,)
  dstd = distd * -1 / (std**2)
  # (D,)
  dvar = dstd * 1 / (2 * std)
  # (N,D)
  dxsquarred = dvar * np.ones((N, D)) / N
  # (N,D)
  dxcorrected = dxhat * istd + dxsquarred * 2 * xcorrected
  # (D,)
  dmu = np.sum(dxcorrected, axis=0) * -1
  # (N,D)
  dx = dxcorrected * 1 + dmu * np.ones((N, D)) / N
  return dx, dgamma, dbeta
```

That's it, we successfully implemented the forward and backward pass of the Batch Norm layer.

### Test time

While it is acceptable to compute the mean and variance on a mini-batch when we are training the network, the same does not hold on test time. When the batch is large enough, its mean and variance will be close to the population's and, as previously stated, the non-deterministic mean and variance has regularizing effects.

On test time, we want the model to be deterministic. The classification result of an image does not depend on the images that we are feeding the model to be classified at the same time. Also, we want to be able to run the model on a single example and it does not make sense to use its mean and variance (the variance would be 0). In conclusion, we need a fixed $$\mu$$ and $$\sigma^2$$ that can be used during inference.

To solve this issue, the authors proposed to compute these values using the population instead of the mini-batch statistics. Since we need to consider the training examples as a sample from all the possible examples, the authors use the mean and the unbiased variance over the mini-batches:

$$ \mu = \frac{1}{B} \sum_i^B \mu_i $$

$$ \sigma^2 = \frac{1}{B - 1} \sum_i^B \sigma^2_i $$

Where $$B$$ is the number of mini-batches, $$\mu_i$$ is the mean of the mini-batch $$i$$ and $$\sigma^2_i$$ is the variance.

The authors also proposed to take a moving average of the mean and variance to track the accuracy of the model as it trains:

$$ \mu = m * \mu + (1 - m) * \mu_i $$

$$ \sigma^2 = m * \sigma^2 + (1 - m) * \sigma^2_i $$

Where $$m$$ is the momentum in the range $$[0, 1]$$ and is typically a value close to 1 (i.e. $$0.9$$). In fact, we can use the results of the moving average as our global statistics instead of using the previous more complicated method. This was the implementation choice made by the Keras library and as such, that is the method that we are going to implement here.

So the forward pass becomes:

```python
def batchnorm_forward(x, gamma, beta, eps, momentum, moving_mean, moving_var, mode):

  y, cache = None, None
  
  if mode == 'train':
    mu = np.mean(x, axis=0)
    xcorrected = x - mu
    xsquarred = xcorrected**2
    var = np.mean(xsquarred, axis=0)
    std = np.sqrt(var + eps)
    istd = 1 / std
    xhat = xcorrected * istd
    y = xhat * gamma + beta

    # Compute the moving average of the mean and variance
    moving_mean = momentum * moving_mean + (1 - momentum) * mu
    moving_var = momentum * moving_var + (1 - momentum) * var
    cache = (gamma, xhat, xcorrected, istd, std)
  elif mode == 'test':
    # Use the moving mean and variance on test time
    std = np.sqrt(moving_var + eps)
    y = (x - moving_mean) / std
    y = gamma * out + beta
  else:
    raise ValueError('Invalid forward batchnorm mode.')

  return y, cache, moving_mean, moving_var
```

And that's it! We have a fully functioning Batch Normalization layer. However, this is the **naive implementation**. We relied on the computational graph to derive the gradient expressions, although we could use calculus to derive a more efficient expression.

While relying on the computational graph and the chain rule to compute the gradient of $$X$$, $$\gamma$$ and $$\beta$$ with respect to the loss is the easiest and more intuitive way of reaching to a solution, it is not the most efficient one. It would be much better if we just used one equation to get the value of the gradient of $$X$$.

Since each equation that we derived uses a value computed on the previous one, we can write one big equation for $$\frac{\partial l}{\partial X}$$ and then simplify it. For example, we can replace the $$\frac{\partial l}{\partial \mu}$$ by $$\frac{\partial l}{\partial xcorrected} * -1$$ and do that recursively until we get one equation that is only a function of $$\frac{\partial l}{\partial y}$$ .

### Efficient Batch Normalization

There are two strategies to derive the gradient expression needed to update the network's weights:

1. Break down the mathematical expressions into atomic operations in order to build a computational graph and make it easier to use the chain rule.
2. Derive the gradient analytically.

The first strategy offers us a simple abstraction to compute the gradient but it is usually not the most efficient implementation. By deriving the gradient expression, it is usually possible to simplify it and remove unnecessary terms. It turns out that the second strategy yields, indeed, a **more efficient gradient expression** for the Batch Norm layer.

Let's start with the simpler parameters: $$\gamma$$ and $$\beta$$. In these cases, the computational graph already gives us the best possible expression, but this will help us remember the basics. Again, we will make use of the chain rule to derive the partial derivative of the loss with respect to $$\gamma$$:

$$
\begin{equation}
  \frac{\partial l}{\partial \gamma} =
    \sum_i^N \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma}
\end{equation}
$$

Why do we need to use the chain rule? We have the value of the $$\frac{\partial l}{\partial y_i}\ $$, as it is provided to us by the next layer, and we can actually derive  the $$\frac{\partial y_i}{\partial \gamma}\ $$. That way, we do not need to know anything about the loss function that was used nor what are the next layers. The gradient expression becomes self-contained, provided that we are given $$\frac{\partial l}{\partial y_i}\ $$.

Why the summation? $$\gamma$$ is a $$D$$ dimensional vector that is multiplied by the $$N$$ vectors $$x_i$$ of dimension $$D$$ and so its contributions must be summed. As I am a very visual person, I better understood this by thinking in terms of a computational graph:

{% include image.html description="" image="blogs/batch-norm/batch-norm-gamma-derivative.png" caption="false"%}

When we are backpropagating through the computational graph, the gradient flowing from each $$y_i$$ node arrives at the same $$\gamma$$ node and, as such, all the gradients must be summed.

The gradient expressions for the partial derivatives of the loss with respect to $$\gamma$$ and $$\beta$$ become:

$$
\begin{align}
  \frac{\partial l}{\partial \gamma}
    &= \sum_i^N \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma} \\
    &= \sum_i^N \frac{\partial l}{\partial y_i} \cdot \hat{x_i} \\
    \\
  \frac{\partial l}{\partial \beta}
    &= \sum_i^N \frac{\partial l}{\partial y_i} \cdot \frac{\partial y_i}{\partial \beta} \\
    &= \sum_i^N \frac{\partial l}{\partial y_i}
\end{align}
$$

What we need to compute next is the partial derivative of the loss with respect to the inputs $$x_i$$, so the previous layers can compute their gradients and update their parameters. We need to gather all the expressions where $$x_i$$ is used that has influence in the $$y_i$$ result. Do not forget that:

$$
\begin{align}
  \mu_B &= \frac{1}{N} \sum_i^N x_i \\
  \sigma^2_B &= \frac{1}{N} \sum_i^N (x_i - \mu_B)^2
\end{align}
$$

We can conclude that $$x_i$$ is used to compute $$\hat{x}_i$$, $$\mu_B$$ and $$\sigma^2$$ and therefore:

$$
\begin{align}
  \frac{\partial l}{\partial x_i}
  &= \frac{\partial l}{\partial \hat{x_i}} \cdot \frac{\partial \hat{x_i}}{\partial x_i} +
     \frac{\partial l}{\partial \sigma^2_B} \cdot \frac{\partial \sigma^2_B}{\partial x_i} +
     \frac{\partial l}{\partial \mu_B} \cdot \frac{\partial \mu_B}{\partial x_i}
\end{align}
$$

Let's compute and simplify each of these terms individually and then we come back to this expression.

The first term is the easiest one to derive. Using the same chain rule process as before we arrive to the following expressions:

$$
\begin{align}
  \frac{\partial l}{\partial \hat{x}_i} &= \frac{\partial l}{\partial y_i} \cdot \gamma \\
  \frac{\partial \hat{x_i}}{\partial x_i} &= (\sigma^2_B + \epsilon)^{-\frac{1}{2}} \\
  \frac{\partial l}{\partial \hat{x_i}} \cdot \frac{\partial \hat{x_i}}{\partial x_i}
    &= \frac{\partial l}{\partial y_i} \cdot \gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}}
\end{align}
$$

So far so good. The next expressions are a bit longer, but once you get the process of the chain rule they are just as simple.

$$
\begin{align}
  \frac{\partial l}{\partial \sigma^2_B}
    &= \sum_i^N \frac{\partial l}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \\
    &= \sum_i^N \frac{\partial l}{\partial y_i} \cdot \gamma \cdot (x_i - \mu_B) \cdot (-\frac{1}{2}) \cdot (\sigma^2_B + \epsilon)^{-\frac{3}{2}} \\
    &= -\frac{\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{3}{2}}}{2} \sum_i^N \frac{\partial l}{\partial y_i} \cdot (x_i - \mu_B)
\end{align}
$$

As what happened with the gradients of $$\gamma$$ and $$\beta$$, to compute the gradient of $$\sigma^2$$ we need to sum over the contributions of all elements from the batch. The same happens to the gradient of $$\mu_B$$, as it is also a $$D$$ dimensional vector:

$$
\begin{align}
  \frac{\partial l}{\partial \mu_B}
  &= \Bigg[\sum_i^N \frac{\partial l}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \mu_B} \Bigg] +
     \Bigg[ \frac{\partial l}{\partial \sigma^2_B} \cdot \frac{\partial \sigma^2_B}{\partial \mu_B} \Bigg] \\
  &= \Bigg[\sum_i^N \frac{\partial l}{\partial y_i} \cdot \gamma \cdot \frac{-1}{\sqrt{\sigma^2_B + \epsilon}} \Bigg] +
     \Bigg[\frac{\partial l}{\sigma^2_B} \cdot \frac{1}{N} \sum_i^N -2(x_i - \mu_B) \Bigg] \\
  &= -\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}} \Big( \sum_i^N \frac{\partial l}{\partial y_i} \Big) -
     \frac{\partial l}{\partial \sigma^2_B} \cdot \frac{2}{N} \Big( \sum_i^N (x_i - \mu_B) \Big) \\
  &= -\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}} \Big( \sum_i^N \frac{\partial l}{\partial y_i} \Big)
\end{align}
$$

In case you are wondering why I deleted the second term of the expression, it turns out that $$\sum_i^N x_i - \mu_B$$ is equal to $$0$$. We are translating all the points so their mean is equal to $$0$$. By looking at the mean's formula, we can see that the only way for the points to have $$0$$ mean is when their sum is also equal to $$0$$:

$$
\begin{equation}
\sum_i^N ( x_i - \mu_B) = \sum_i^N(x_i) - N \cdot \mu_B = \sum_i^N(x_i) - N \cdot \frac{1}{N} \sum_i^N (x_i) = 0
\end{equation}
$$

Now we can easily compute the rest of the terms:

$$
\begin{align}
  \frac{\partial l}{\partial \sigma^2_B} \cdot \frac{\partial \sigma^2_B}{\partial x_i}
  &= \frac{\partial l}{\partial \sigma^2_B} \cdot \frac{2(x_i - \mu_B)}{N} \\
  &= -\frac{\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{3}{2}}}{2} \Big( \sum_j^N \frac{\partial l}{\partial y_j} \cdot (x_j - \mu_B) \Big) \cdot \frac{2(x_i - \mu_B)}{N} \\
  &= -\frac{\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{3}{2}}}{N} \Big( \sum_j^N \frac{\partial l}{\partial y_j} \cdot (x_j - \mu_B) \Big) \cdot (x_i - \mu_B) \\
  &= \frac{\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}}}{N} \Big( \sum_j^N \frac{\partial l}{\partial y_j} \cdot (x_j - \mu_B) \Big) \cdot (x_i - \mu_B) \cdot - (\sigma^2_B + \epsilon)^{-1}
\end{align}
$$

Let's focus on this term of the equation:

$$
\Big( \sum_j^N \frac{\partial l}{\partial y_j} \cdot (x_j - \mu_B) \Big) \cdot \frac{xi - \mu_B}{\sigma^2 + \epsilon}
$$

Notice that $$x_j - \mu_B = \hat{x}_j \sqrt{\sigma_B^2 + \epsilon}$$ and so:

$$
\begin{align}
& \Big( \sum_j^N \frac{\partial l}{\partial y_j} \cdot \hat{x}_j \sqrt{\sigma_B^2 + \epsilon} \Big) \cdot \frac{xi - \mu_B}{\sigma^2 + \epsilon} \\
&= \Big( \sum_j^N \frac{\partial l}{\partial y_j} \cdot \hat{x}_j \Big) \cdot \frac{xi - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \\
&= \frac{\partial l}{\partial \gamma} \cdot \frac{xi - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \\
&= \frac{\partial l}{\partial \gamma} \cdot \hat{x}_i
\end{align}
$$

And finally:

$$
\begin{align}
  \frac{\partial l}{\partial \mu_B} \cdot \frac{\partial \mu_B}{\partial x_i}
  &= \frac{\partial l}{\partial \mu_B} \cdot \frac{1}{N} \\
  &= \frac{-\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}}}{N} \Big( \sum_j^N \frac{\partial l}{\partial y_j} \Big) \\
  &= \frac{-\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}}}{N} \cdot \frac{\partial l}{\partial \beta}
\end{align}
$$

Merging everything together:

$$
\begin{equation}
\frac{\partial l}{\partial x_i} =
   \frac{\gamma \cdot (\sigma^2_B + \epsilon)^{-\frac{1}{2}}}{N}
      \Bigg[
        N \frac{\partial l}{\partial y_i} -
        \frac{\partial l}{\partial \gamma} \cdot \hat{x}_i -
        \frac{\partial l}{\partial \beta}
      \Bigg]
\end{equation}
$$

Finally we have a single mathematical expression for the partial derivative of the loss with respect to each input. This is a simpler expression than what we get by deriving the expression using the computational graph.

Translating this to python, we end up with a much more compact method:

```python
def batchnorm_backward_alt(dout, cache):
  gamma, xhat, istd = cache
  N, _ = dout.shape

  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(xhat * dout, axis=0)
  dx = (gamma*istd/N) * (N*dout - xhat*dgamma - dbeta)
  
  return dx, dgamma, dbeta
```

When we implements **Layer Normalization** and **Group Normalization**, we need to transform the matrix so that we can reuse **Batch Normalization** code, so we derive the expression like the following. The benifits is that it only use `dxhat` and `xhat` to calucate, make our code more simpler when deal with matrix transform.

```python
= (gamma*istd/N) * (N*dout - xhat*dgamma - dbeta)
= (istd/N) * (N*gamma*dout - xhat*gamma*dgamma - gamma*dbeta)
= (istd/N) * (N*dxhat - xhat * np.sum(dxhat * xhat, axis=0) - np.sum(dxhat, axis=0))
```

so we have

```python
def batchnorm_backward_alt(dout, cache):
  gamma, xhat, istd = cache
  N, _ = dout.shape

  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(xhat * dout, axis=0)
  dxhat = dout * gamma
  
  dx = (istd/N) * (N*dxhat - xhat * np.sum(dxhat * xhat, axis=0) - np.sum(dxhat, axis=0))
  return dx, dgamma, dbeta
```

This method is 2 to 4 times faster than the one presented on computational graph. It might not seem much, but when we are talking about deep neural networks that may take weeks to train, every little improvement in the end makes a huge difference.
