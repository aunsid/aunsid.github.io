# Learning Rate Schedules

A fixed learning rate rarely works well for training large models. Schedules adjust the rate over the course of training.

## Linear Warmup

Start with a very small learning rate and ramp up linearly over the first N steps. Prevents large, destabilizing gradient updates early in training when weights are random.

## Cosine Decay

After warmup, decay the learning rate following a cosine curve down to near zero by the end of training.

## Linear Decay

## Warmup + Cosine (combined)

Used by most modern LLMs (GPT, LLaMA, etc.).

*Coming soon.*
