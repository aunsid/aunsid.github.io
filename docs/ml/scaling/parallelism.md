# Parallelism

Training large models requires distributing work across many GPUs. Different parallelism strategies partition different things.

## Data Parallelism

Each GPU holds a full model copy and processes a different batch. Gradients are averaged across GPUs.

## Tensor Parallelism

Split individual weight matrices across GPUs. Reduces per-GPU memory but requires frequent communication.

## Pipeline Parallelism

Split the model's layers across GPUs. Each GPU handles a stage of the forward and backward pass.

## Mixed Precision (fp16, bf16)

*Coming soon.*
