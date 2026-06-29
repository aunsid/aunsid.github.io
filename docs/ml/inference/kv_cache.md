# KV Cache
At the time of generating the output response, everytime a new token is generated we compute the keys
During autoregressive generation, keys and values for all previous tokens are recomputed at every step — wasteful since they don't change. The KV cache stores them after the first computation and reuses them.

```python
# to generate the next token

idx_cropped = idx[:, -contex_size:]
logits = model(idx_cropped)
logits = logits[:, -1, :] # last position of vocabs to sample next token
```

We do the above steps repeatedly until we reach the EOS token or reach the maximum sequence we want to produce. 

## How It Works
The key observation is that in causal attention, token at position $t$ only attends to positions $\leq t$. The Q, K, V vectors from $ 0, \dots , t-1 $ do not change and need not be recomputed!

Thats the key idea!

For the new token, we only need to compute its own query - $Q_t$, and keys and values from previous positions - $K_{0\dots\t-1} and $T_{0\dots\t-1}.

Note: Q is not cached. Past Q's are not never reused  as each tokens query attends once.

## Memory Tradeoffs
Without cache, at step t, does attention over a t x t matrix (Q @ K.T where both are length t)

With cache, step t does attention over 1 x t matrix - One new query against t cache keys. The new K and V vectors get appended to the cache.

Compute Cost:
Without Cache - $O(t. d)$ for projecting Q/K/V and $O(t^2. d)$ for attention
With Cache - $O(d)$ for projecting and $O(t. d)$ for attention

The trade-off is memory. You are now storing K and V for every lauer, every head, every past position. The cache size is -
2 (K and V) x num_layer x num_heads x seq_len x dtypes_bytes x batch

For a small model this is trivial, but for a large model =>70B this is a dominant cost of inference so we use Multi-Query Attention, Grouped-Query Attention, Paged Attention and quantized KV caches.


