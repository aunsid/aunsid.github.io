# Normalization

Consider a simple neuron that computes a weighted sum of its inputs:

$$z = y_1 + y_2 + y_3 + y_4, \quad \text{where } y_i = x_i \cdot w_i$$

During backpropagation, we compute gradients via the chain rule. Starting from the output:

$$\frac{dL}{dz} = 1$$

$$\frac{dL}{dy_i} = \frac{dL}{dz} \cdot \frac{dz}{dy_i}$$

Since the $w_i$ are the learnable parameters, we ultimately need:

$$\frac{dL}{dw_i} = \frac{dL}{dy_i} \cdot \frac{dy_i}{dw_i} = \frac{dL}{dy_i} \cdot x_i$$

Notice that the gradient with respect to each weight scales directly with its input $x_i$. If inputs are very large or very small — or vary wildly across features — the gradients will be correspondingly large, small, or imbalanced. This makes learning unstable, and the problem compounds as layers are stacked: skewed activations in one layer feed skewed inputs into the next.

Normalization addresses this by rescaling activations to a controlled range, keeping gradients well-behaved throughout the network. The core operation is:

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

where $\mu$ and $\sigma^2$ are the mean and variance computed over some set of values, and $\varepsilon$ is a small constant added for numerical stability (prevents division by zero when variance is near zero).

Most normalization layers then apply a learned affine transform to restore expressive power:

$$\text{Norm}(x) = \gamma \cdot \hat{x} + \beta$$

$\gamma$ (scale) and $\beta$ (shift) are learnable parameters initialized to 1 and 0, allowing the network to undo normalization if needed.

## Layer Norm

In transformer models, layer norm normalizes each token's activations independently across the feature dimension. Unlike batch norm, it does not depend on batch statistics — $\mu$ and $\sigma^2$ are computed over the $d_\text{model}$ features of a single token:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta$$

This makes it well-suited for variable-length sequences and small batch sizes. In modern transformers, layer norm is typically applied *before* the attention and feed-forward blocks (pre-norm), rather than after (post-norm), which improves training stability.

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))   # gamma
        self.bias = nn.Parameter(torch.zeros(d_model))    # beta

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.weight * x_norm + self.bias
```

## RMS Norm

RMS Norm drops the mean-centering step entirely, normalizing only by the root mean square of the activations:

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x)}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \varepsilon}$$

There is no $\beta$ term — the shift is removed along with the mean subtraction. This is <u>cheaper to compute and empirically matches the performance of full layer norm</u>. It is used in LLaMA and most modern LLMs.


```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = nn.Parameter(
            torch.ones(d_model, dtype=dtype, device=device)
        )
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32) # handle precision
        rms = ((1/self.d_model) * (x ** 2).sum(dim=-1, keepdim=True) + self.eps) ** 0.5
        out = x / rms * self.gain
        return out.to(in_dtype)
```

