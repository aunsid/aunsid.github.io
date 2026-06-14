# KV Cache
At the time of generating the output response, everytime a new token is generated we compute the keys
During autoregressive generation, keys and values for all previous tokens are recomputed at every step — wasteful since they don't change. The KV cache stores them after the first computation and reuses them.

## How It Works

## Memory Tradeoffs

## Relation to Context Length

*Coming soon.*
