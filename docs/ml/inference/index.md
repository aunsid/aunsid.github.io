# Decoding Strategies

At inference time, the model outputs a probability distribution over the vocabulary at each step. How you select the next token from that distribution determines the quality and diversity of the output.

## Greedy Decoding

Always pick the highest-probability token. Fast but repetitive.

## Beam Search

Maintain the top-k candidate sequences at each step. Better quality than greedy but expensive and prone to generic outputs.

*Coming soon.*
