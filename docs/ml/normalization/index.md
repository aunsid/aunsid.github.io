# Normalization

Normalization stabilizes training by rescaling activations, preventing gradients from vanishing or exploding.

## Layer Norm

Normalizes across the feature dimension for each token independently.

## RMS Norm

A simpler variant that skips the mean-centering step. Used in LLaMA and most modern LLMs.

*Coming soon.*
