# Optimizers

## Gradient Descent
The process of continuously finding the gradient, and then taking a step in the gradient direction (parameter update).

```python
while True:
    weights_grad = evaluate_gradient(loss_function, data, weights)
    weights += step_size * weights_grad
```