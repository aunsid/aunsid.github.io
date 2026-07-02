# KV Cache

During autoregressive generation, keys and values for all previous tokens are recomputed at every step — wasteful since they don't change. The KV cache stores them after the first computation and reuses them.

```python
# to generate the next token

idx_cropped = idx[:, -context_size:]
logits = model(idx_cropped)
logits = logits[:, -1, :] # last position of vocabs to sample next token
```

We repeat the above steps until we reach the EOS token or the maximum sequence length we want to produce.

## How It Works
The key observation is that in causal attention, the token at position $t$ only attends to positions $\leq t$. The Q, K, V vectors from $0, \dots, t-1$ do not change and need not be recomputed.

That's the key idea.

For the new token, we only need to compute its own query $Q_t$, and reuse the keys and values from previous positions — $K_{0 \dots t-1}$ and $V_{0 \dots t-1}$.

Note: Q is not cached. Past Q's are never reused, as each token's query attends only once.

## Memory Tradeoffs
Without cache, at step $t$, the model computes attention over a $t \times t$ matrix ($Q K^\top$ where both are length $t$).

With cache, step $t$ computes attention over a $1 \times t$ matrix — one new query against $t$ cached keys. The new K and V vectors then get appended to the cache.

Compute cost:

- Without cache — $O(t \cdot d)$ for projecting Q/K/V and $O(t^2 \cdot d)$ for attention.
- With cache — $O(d)$ for projecting and $O(t \cdot d)$ for attention.

The trade-off is memory. You are now storing K and V for every layer, every head, every past position. The cache size is:

`2 (K and V) x num_layers x num_heads x seq_len x dtype_bytes x batch`

For a small model this is trivial, but for a large model ($\geq$ 70B) this is a dominant cost of inference — which is why we use Multi-Query Attention, Grouped-Query Attention, Paged Attention, and quantized KV caches.

