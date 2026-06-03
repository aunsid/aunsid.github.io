# Pre-Norm vs Post-Norm

Where normalization is applied relative to the residual connection has a large effect on training stability.

## Post-Norm (original Transformer)

Normalization is applied after the residual addition. Harder to train deep networks without careful learning rate warmup.

## Pre-Norm

Normalization is applied before the sublayer (attention or FFN). Used in most modern LLMs — more stable training.

*Coming soon.*
