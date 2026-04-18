---
layout: post
title: "Convolutional Neural Networks"
category: ml
topic: backpropagation
date: 2026-04-17
description: "Forward pass, backpropagation through conv layers, and a minimal PyTorch implementation."
---

Convolutional Neural Networks (CNNs) apply learned filters spatially across an input, making them highly effective for image data.

## Architecture

A standard CNN stacks convolution, activation, and pooling layers before a dense classifier:

```
Input → [Conv → ReLU → Pool]* → Flatten → Dense → Softmax
```

Each convolutional layer learns a set of filters that detect local patterns (edges, textures, shapes) at different scales.

## The Forward Pass

For a 2D convolution, each output activation is a dot product between filter weights $W$ and a local patch of the input, plus a bias:

$$z^{(l)} = W^{(l)} * a^{(l-1)} + b^{(l)}$$

where $*$ denotes cross-correlation. After applying ReLU:

$$a^{(l)} = \max(0,\; z^{(l)})$$

Max pooling then downsamples by taking the maximum over non-overlapping $k \times k$ windows, reducing spatial dimensions by a factor of $k$.

## Backpropagation Through a Conv Layer

The gradient of loss $\mathcal{L}$ with respect to the kernel is a cross-correlation between the upstream gradient and the previous layer's activations:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} \star a^{(l-1)}$$

The gradient flowing to the previous layer is a full convolution (180° rotated kernel) with the upstream gradient:

$$\frac{\partial \mathcal{L}}{\partial a^{(l-1)}} = W^{(l)} \circledast \frac{\partial \mathcal{L}}{\partial z^{(l)}}$$

This is why convolutions are sometimes called "linear" in the backprop sense — the gradient computation is itself a convolution.

## Key Properties

| Property | Effect |
|---|---|
| Parameter sharing | Same filter slides over all positions → far fewer parameters than a dense layer |
| Local connectivity | Each neuron sees only a small receptive field |
| Translation invariance | Pooling makes representations robust to small shifts |
| Depth = abstraction | Early layers detect edges; deep layers detect high-level concepts |

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 → 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 → 14x14
        self.pool = nn.MaxPool2d(2)                                # halves spatial dims
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # → 32 x 14 x 14
        x = self.pool(F.relu(self.conv2(x)))  # → 64 x 7 x 7
        x = x.view(x.size(0), -1)             # flatten
        return self.fc(x)

# Training loop (single batch)
model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

images = torch.randn(32, 1, 28, 28)  # batch of 32 MNIST-like images
labels = torch.randint(0, 10, (32,))

optimizer.zero_grad()
loss = criterion(model(images), labels)
loss.backward()
optimizer.step()
```

## Gradient Flow Check

A quick sanity check — all parameters should have non-None gradients after `loss.backward()`:

```python
for name, param in model.named_parameters():
    assert param.grad is not None, f"No gradient for {name}"
    print(f"{name}: grad norm = {param.grad.norm():.4f}")
```

## Further Reading

- LeCun et al. (1998) — [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- Krizhevsky et al. (2012) — AlexNet (ImageNet breakthrough)
- He et al. (2015) — [ResNet](https://arxiv.org/abs/1512.03385) (residual connections)
