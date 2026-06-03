# Types of Attention

## Causal (Masked) Self-Attention

Used in decoder-only models (GPT). Each token can only attend to itself and previous tokens — future positions are masked.

## Cross-Attention

Used in encoder-decoder models. Queries come from the decoder; keys and values come from the encoder output.

## Multi-Query Attention (MQA)

## Grouped-Query Attention (GQA)

## Flash Attention

*Coming soon.*
