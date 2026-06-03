# Top-k and Top-p Sampling

## Temperature

Divide logits by temperature T before softmax. T < 1 sharpens the distribution (more confident); T > 1 flattens it (more random).

## Top-k Sampling

At each step, keep only the k highest-probability tokens and sample from that truncated distribution. Prevents sampling from the long tail of unlikely tokens.

## Top-p (Nucleus) Sampling

Instead of a fixed k, keep the smallest set of tokens whose cumulative probability exceeds p. The set size adapts dynamically — smaller on confident steps, larger on uncertain ones.

## Combining Temperature, Top-k, and Top-p

*Coming soon.*
