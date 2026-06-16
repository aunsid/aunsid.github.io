# Post-Training

If we take a raw pretrained model and ask it a simple question, instead of actually answering, the model may start to output options and turn the question into a quiz. A raw model is essentially an untrained text predictor — all it does is try to predict the next most likely token.

The process of turning a broad base model into a focused chat model is called "instruction tuning" or "alignment."

## SFT
The first step is supervised fine-tuning (SFT). We show the model what a good answer looks like.

Process:

1. **Collect data.** Human labelers are hired to manually write thousands of (prompt, ideal response) pairs. This step is slow and expensive.

2. **Train.** Fine-tune the base model with the same training objective — predict the next token. But this time, instead of raw data from the internet, we use the curated data from step 1.

Base model:
- Goal: predict the next token
- Data: unstructured internet data
- Behavior: completes patterns, with no sense of user intent

SFT model:
- Goal: imitate expert responses
- Data: curated (prompt, answer) pairs
- Behavior: follows instructions

The limitation of SFT is that the model treats the human-written response as the only correct answer and everything else as wrong.

## Preference Learning

Writing a single "perfect" response is hard, but humans are much better at *comparing* two responses and saying which is better. So instead of collecting more written answers, we collect preferences.

Steps:

1. Take a prompt.
2. Use the SFT model to generate multiple responses (A, B, C, D).
3. Ask humans to rank those responses (e.g. D > B > C > A).
4. Break the ranking down into pairwise comparisons.
5. Use these pairs to train the model.

From here, there are two main paths to actually using the preference data:

**DPO (Direct Preference Optimization)**
- Directly adjusts model probabilities to prefer the chosen response over the rejected one. Single-stage process.
- Analogy: giving edits — "say Y, not X."

**RLHF (Reinforcement Learning from Human Feedback)**
- Train a separate reward model on the preferences first, then use RL (typically PPO) to fine-tune the policy to maximize that reward.
- Analogy: a judge scores you, and you practice to maximize the score.


References:

1. RLHF in 90 minutes — https://www.youtube.com/watch?v=j3BdFm_Veq4&t=759s


