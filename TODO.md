# Writing TODO

## Chapter 1 — Language Models From Scratch

### Tokenization
- [x] Tokens (`docs/ml/tokenizer/tokens.md`)
- [x] Byte Pair Encoding (`docs/ml/tokenizer/notes.md`)

### Embeddings
- [x] Token Embeddings + Positional Encodings (sinusoidal, RoPE, ALiBi) (`docs/ml/embeddings/index.md`)

### Attention
- [ ] Self-Attention & Multi-Head Attention (`docs/ml/attention/index.md`)
- [ ] Types of Attention — causal, cross, GQA, MQA, Flash Attention (`docs/ml/attention/types.md`)

### Normalization
- [ ] Layer Norm & RMS Norm (`docs/ml/normalization/index.md`)
- [ ] Pre-Norm vs Post-Norm (`docs/ml/normalization/pre_post.md`)

### Feed-Forward Networks
- [ ] FFN variants — ReLU, GELU, SwiGLU, GeGLU (`docs/ml/feed_forward/index.md`)

### Transformer Architecture
- [ ] Full transformer block, encoder vs decoder (`docs/ml/transformer/index.md`)

### Training
- [ ] Backpropagation & Loss (cross-entropy, perplexity, gradient clipping) (`docs/ml/training/index.md`)
- [ ] Optimizers — SGD, Adam, AdamW (`docs/ml/training/optimizers.md`)
- [ ] Learning Rate Schedules — warmup, cosine decay (`docs/ml/training/learning_rates.md`)

### Inference
- [ ] Decoding Strategies — greedy, beam search (`docs/ml/inference/index.md`)
- [ ] Top-k & Top-p (nucleus) Sampling (`docs/ml/inference/sampling.md`)
- [ ] KV Cache (`docs/ml/inference/kv_cache.md`)

### Scaling
- [ ] Scaling Laws — Kaplan, Chinchilla (`docs/ml/scaling/index.md`)
- [ ] Parallelism — data, tensor, pipeline + mixed precision (`docs/ml/scaling/parallelism.md`)

---

## Chapter 2 — Vision

- [ ] Image Representation — pixels, channels, normalization (`docs/ml/vision/image_representation.md`)
- [x] Convolutions & CNNs (`docs/ml/vision/cnn.md`)
- [ ] CNN Architectures — VGG, ResNet, EfficientNet (`docs/ml/vision/architectures.md`)
- [ ] Normalization — Batch Norm vs Layer Norm (`docs/ml/vision/normalization.md`)
- [ ] Vision Transformers (ViT) — patch embeddings, architecture (`docs/ml/vision/vit.md`)

---

## Chapter 3 — Vision-Language Models

- [ ] Contrastive Learning — InfoNCE, SimCLR (`docs/ml/vlm/contrastive.md`)
- [ ] CLIP — architecture, training objective, zero-shot (`docs/ml/vlm/clip.md`)
- [ ] VLM Architectures — Flamingo, LLaVA, GPT-4V (`docs/ml/vlm/architectures.md`)
- [ ] Training VLMs — instruction tuning, evaluation (`docs/ml/vlm/training.md`)

---

## Site
- [ ] Fill in About page (`docs/about.md`)
