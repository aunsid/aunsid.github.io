# Optimizers

## Gradient Descent

The process of continuously finding the gradient, and then taking a step in the gradient direction (parameter update).

```python
while True:
    weights_grad = evaluate_gradient(loss_function, data, weights)
    weights += step_size * weights_grad
```

## SGD with Momentum

## Adam

Adaptive Moment Estimation. Maintains a running mean and variance of gradients per parameter, adapting the effective learning rate for each weight.

## AdamW

Adam with decoupled weight decay — applies weight decay directly to the parameters rather than folding it into the gradient update. Fixes a subtle bug in Adam's L2 regularization.

*Coming soon.*
