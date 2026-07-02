# GRPO


###  Language Models as Policies
In traditional RL, we think about a policy $\pi$ which takes in a state $s_t$, and outputs an action $a_t$. This action results in a reward $r_t = r(s_t, a_t)$, and a subsequent state $s_{t+1} \sim \text{next\_state}(s_t, a_t)$.

A language model $\pi_\theta$ with parameters $\theta$ defines a probability distribution over the next token $y_t$ given the current text prefix $y_{<t} = (y_1, \dots, y_{t-1})$.
In the context of RL, we can think of the next token $y_t$ as our action $a_t$ and the current text prefix $y_{<t}$ as the state $s_t$. Hence, the LM is a categorical stochastic policy.
$$ a_t \sim \pi_{\theta}(\cdot \mid s_t) = [\text{softmax}(f_{\theta}(s_t))]_{a_{t}} $$

To optimize a policy with policy gradients, we need two operations:
1. Sampling from the policy: drawing an action $a_{t}$ from the distribution.
2. Scoring the log-likelihood of an action: evaluating $\log \pi_{\theta}(a_t \mid s_t)$.

$s_t$ is the partial completion and each $a_t$ is the next token of the solution. The episode ends with the <|end_of_text|> token.

###  Trajectories
In reinforcement learning, we start at an initial state drawn from a starting distribution. We then sample an action $a_t$ from the policy $\pi_t$, transition to the next state according to some next-state distribution $s_{t+1} \sim \text{next\_state}(s_t, a_t)$, and repeat. This series of states and actions then forms a trajectory.

$$ \tau = (s_0, a_0, s_1, a_1, \dots , s_T, a_T)$$

where $T$ is the length of the trajectory, i.e., $a_T$ is an end-of-text token or we have reached the maximum budget of tokens. Trajectories are also called episodes or rollouts.


### Rewards and Return
In RL, a scalar reward $r_t = r(s_t, a_t)$ judges the immediate quality of action $a_t$ taken at state $s_t$. But for our use case, we observe zero reward for intermediate reasoning steps until we output the answer. We then receive a verified reward on the terminal action.

$$ r_T = r(s_T, a_T) := \begin{cases} 1 & \text{if trajectory } \tau \text{ matches the ground-truth according to our reward function} \\ 0 & \text{otherwise} \end{cases}.$$

In our case, $r_t$ is 1 if our final answer is correct, and 0 otherwise. The return $R(\tau)$ aggregates the rewards along the trajectory, and in our case it is just $r_T$.


The objective is to maximize the expected return
$$ J_\theta = \mathbb{E}_{\tau \sim \pi_{\theta}}[r(\tau)] $$

where $\tau \sim \pi_{\theta}$ is the distribution over trajectories where we first sample $s_0 \sim \rho$ and we then sample actions from the policy and the next states from the next-state distribution.
In our case, this means we first sample our math problem from our dataset, and then sample tokens until we hit the end of our response — a process we can denote by $y \sim \pi_{\theta}(y \mid x)$. This objective leads to the optimization of

$$ \theta^* = \arg\max_\theta J_\theta $$

### Policy Gradients
Instead of using action $a_t$ and states $s_t$, we will use $(x, y)$ denoting prompts and responses.

We want to optimize the model's expected reward (i.e., accuracy), given by

$$J_\theta = \mathbb{E}_{x \sim \rho} \mathbb{E}_{y \sim \pi_\theta(y \mid x)} [r(y \mid x)]\tag{1}$$


where $r(y \mid x)$ denotes whether the response $y$ is correct for prompt $x$. Our approach will be to do gradient ascent on the objective:

$$\theta_{k+1} = \theta_k + \alpha \nabla_\theta J_{\theta_k},\tag{2}$$

Equation 1 reads as follows: we draw a prompt from our training dataset of prompts, and then — given the prompt — the inner expectation draws a response $y$ from the policy's distribution.
$$\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}[r(y \mid x)] = \sum_{y} \pi_\theta(y \mid x) \, r(y \mid x). \tag{3}$$

This sum is astronomical: there are many possible token sequences that we cannot enumerate.

So in practice you replace it with a Monte Carlo estimate: sample a finite number of responses and average. That gives an unbiased estimate of the true expectation. Two knobs:

  1. How many prompts you sample per batch (estimating the outer expectation).
  2. How many responses you sample per prompt (estimating the inner one).

  The second knob is where different algorithms diverge:

  - Vanilla REINFORCE-style policy gradient: often just 1 response per prompt. Cheap, noisy.
  - Group-relative methods like GRPO: intentionally sample $G$ responses per prompt (e.g., 4, 8, 16) so you can compare them against each other and use the group mean as a baseline for
  variance reduction.

  Coming back to equation 2, we need to compute $\nabla_\theta J_{\theta_k}$, the gradient from samples.

### The Log-Derivative Trick

The gradient of the objective is

$$\nabla_\theta J_\theta = \nabla_\theta \, \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} [r(y \mid x)] = \nabla_\theta \sum_y \pi_\theta(y \mid x) \, r(y \mid x).$$

We can't just push the gradient inside the sum and Monte-Carlo it directly because the distribution we are sampling from itself depends on $\theta$. The standard trick is to multiply and divide by $\pi_\theta(y \mid x)$:

$$\nabla_\theta \pi_\theta(y \mid x) = \pi_\theta(y \mid x) \, \frac{\nabla_\theta \pi_\theta(y \mid x)}{\pi_\theta(y \mid x)} = \pi_\theta(y \mid x) \, \nabla_\theta \log \pi_\theta(y \mid x).$$

Plugging this back in:

$$\nabla_\theta J_\theta = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[\, r(y \mid x) \, \nabla_\theta \log \pi_\theta(y \mid x) \,\big]. \tag{4}$$

This is the REINFORCE estimator. The expectation is now over the same distribution on both sides, so we can replace it with a Monte Carlo average over sampled responses:

$$\nabla_\theta J_\theta \approx \frac{1}{N} \sum_{i=1}^N r(y_i \mid x) \, \nabla_\theta \log \pi_\theta(y_i \mid x), \quad y_i \sim \pi_\theta(\cdot \mid x).$$

Intuition: each sampled response $y_i$ contributes a gradient that pushes the model to make $y_i$ more likely, weighted by how good $y_i$ was. Good responses get amplified, bad ones get suppressed.

### Per-Token Decomposition

Since a response is a sequence of tokens, the log-probability factorizes:

$$\log \pi_\theta(y \mid x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}).$$

So the gradient becomes

$$\nabla_\theta \log \pi_\theta(y \mid x) = \sum_{t=1}^{|y|} \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t}).$$

Every token in the response gets the same scalar weight $r(y \mid x)$ — there is no per-token reward, only the final outcome. The reward is broadcast back to every token that produced it.

### The Variance Problem

REINFORCE is unbiased but extremely high-variance. Two reasons:

1. The reward $r(y \mid x) \in \{0, 1\}$ is a coarse signal — the gradient is either "push up" or do nothing.
2. The magnitude of $\nabla_\theta \log \pi_\theta(y \mid x)$ can be huge for long sequences.

The fix is a **baseline**. Subtracting any function $b(x)$ that depends only on the prompt (not on the action) leaves the estimator unbiased:

$$\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[\, b(x) \, \nabla_\theta \log \pi_\theta(y \mid x) \,\big] = b(x) \, \nabla_\theta \underbrace{\sum_y \pi_\theta(y \mid x)}_{= 1} = 0.$$

So the new estimator

$$\nabla_\theta J_\theta = \mathbb{E}\big[\, (r(y \mid x) - b(x)) \, \nabla_\theta \log \pi_\theta(y \mid x) \,\big]$$

has the same mean but lower variance, provided $b(x)$ is a good prediction of the expected reward at $x$. The quantity $A(x, y) = r(y \mid x) - b(x)$ is called the **advantage** — how much better this response was than average for this prompt.

In PPO, $b(x)$ is learned by a separate value network (the critic). That doubles the memory footprint and adds training complexity. GRPO's contribution is to skip the value network entirely.

### Group-Relative Advantage

GRPO's key idea: for each prompt, sample a group of $G$ responses and use the group's own mean (and std) as the baseline.

For each prompt $x$, sample $\{y_1, \dots, y_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)$ and compute the rewards $\{r_1, \dots, r_G\}$. The group-relative advantage for response $i$ is

$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \varepsilon}.$$

This is just z-scoring the rewards within the group. The mean serves as the baseline; the std normalizes so the magnitudes don't blow up.

Why this works for verifiable rewards: when all $G$ responses to a prompt are correct ($r_i = 1$ for all $i$), every advantage is $0$ — no gradient. Same when all are wrong. The model only learns from prompts where the group disagrees, which are precisely the prompts at the edge of its current ability. This is automatic curriculum learning.

In code:

```python
def group_relative_advantage(rewards):
    # rewards: tensor of shape (G,) for one prompt
    mean = rewards.mean()
    std = rewards.std() + 1e-8
    return (rewards - mean) / std
```
### Normalizations
#### Advantage Normalization
The next modification in GRPO, after subtracting the group mean, will be to divide by the standard deviation.

$$\hat{g} \leftarrow \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} \frac{r(y^{(i,j)} \mid x^{(i)}) - \mu_i}{\text{std}_i} \, \nabla_\theta \log \pi_\theta(y^{(i,j)} \mid x^{(i)}), \tag{25}$$

where,
$$\text{std}_i = \sqrt{\frac{1}{G} \sum_{j=1}^{G} \left( r(y^{(i,j)} \mid x^{(i)}) - \mu_i \right)^2}$$

(note that the default `torch.std` function uses a slightly different sample std expression to do bias correction; you should use the default `torch.std` in your implementation).

Dividing by the standard deviation no longer preserves the expectation of the estimator, so we are no longer doing gradient ascent on our original estimator $J_{\theta}$. One way to think about this modification is as a stability trick: assuming each gradient vector is iid Gaussian, dividing by the group std ensures that each group's gradient update is roughly equal in norm.

#### Sequence Normalization
There is one other detail in GRPO called sequence normalization that is inherited from PPO implementations.
GRPO adds an additional sequence-length normalization term.
Longer responses accumulate more log-probability terms, so their gradient contribution grows with length. To prevent long rollouts from dominating the update, GRPO averages the per-token log-probs over the response length before summing across rollouts:

$$\hat{g} \leftarrow \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} \frac{r(y^{(i,j)} \mid x^{(i)}) - \mu_i}{\text{std}_i} \, \nabla_\theta \left( \frac{1}{\text{len}(y^{(i,j)})} \sum_{t=1}^{\text{len}(y^{(i,j)})} \log \pi_\theta\big(y_t^{(i,j)} \mid x^{(i)}, y_{<t}^{(i,j)}\big) \right). \tag{27}$$

Each rollout's contribution is the *average* per-token log-prob rather than the *sum*, so a 500-token response and a 50-token response are weighted equally by the outer $\frac{1}{BG}$.



#### Implementing On-Policy GRPO
![On-Policy GRPO algorithm](image-2.png)

This is the complete on-policy GRPO implementation.

Putting the group-relative advantage and sequence normalization together, the on-policy GRPO objective is:

$$J_\theta^{\text{GRPO-on-policy}} = \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} \frac{1}{\text{len}(y^{(i,j)})} \sum_{t=1}^{\text{len}(y^{(i,j)})} \frac{r(y^{(i,j)} \mid x^{(i)}) - \mu_i}{\text{std}_i} \, \log \pi_\theta\big(y_t^{(i,j)} \mid x^{(i)}, y_{<t}^{(i,j)}\big). \tag{32}$$


### RL Algorithm Variants
#### Dr. GRPO
During the GRPO derivation, you may have noticed a series of choices that meant the GRPO policy gradient estimator was not "true/correct" with respect to the expected reward $J_\theta$. These two choices were adding the standard-deviation advantage normalization and sequence-length normalization. Removing these, we get the Dr. GRPO gradient estimator:

$$\hat{g} \leftarrow \frac{1}{Z} \sum_{i=1}^{B} \sum_{j=1}^{G} \sum_{t=1}^{\text{len}(y^{(i,j)})} \big(r(y^{(i,j)} \mid x^{(i)}) - \mu_i\big) \, \nabla_\theta \log \pi_\theta\big(y_t^{(i,j)} \mid x^{(i)}, y_{<t}^{(i,j)}\big),$$

where $Z$ is a normalization constant (typically chosen so the estimator has a stable scale, e.g. the total number of tokens across the batch, or $BG \cdot L_{\max}$ where $L_{\max}$ is the maximum response length).

Compared to standard GRPO, Dr. GRPO drops two things:

1. The per-group std $\text{std}_i$ in the denominator (no advantage-normalization stability trick).
2. The per-response length normalization $\frac{1}{\text{len}(y^{(i,j)})}$ (no sequence-length averaging).

Removing these keeps the estimator unbiased with respect to the true expected reward $J_\theta$, at the cost of the stability properties GRPO gained from them.

**When to use Dr. GRPO:**

- **You care about unbiased optimization of $J_\theta$.** Research settings, ablation studies, or anywhere you want the gradient to actually correspond to the reward you're maximizing.
- **You've observed length bias with standard GRPO.** The $\frac{1}{\text{len}(y)}$ term in standard GRPO makes long incorrect rollouts contribute less per-token, which can subtly bias the policy toward shorter (or longer) responses depending on the reward structure. Dr. GRPO removes this.
- **Rewards are already well-calibrated across prompts.** If prompt difficulty is roughly uniform, you don't need the std-normalization stability trick — the group mean baseline is enough.
- **Avoid** when reward variance across groups is very different (some prompts have all-1s or all-0s, some have mixed) — you'll want std normalization back for stability.

#### Rejection Fine-tuning
The next variant is a much simpler algorithm typically referred to as "rejection fine-tuning (RFT)" or "expert iteration." As the name suggests, the algorithm involves sampling a bunch of rollouts, keeping the ones that are correct, and doing supervised fine-tuning on them (i.e., minimizing next-word prediction log loss). The RFT gradient is

$$\hat{g} \leftarrow \nabla_\theta \left[ \frac{1}{Z} \sum_{i=1}^{B} \sum_{j=1}^{G} \mathbb{1}\!\left\{ r(y^{(i,j)} \mid x^{(i)}) = 1 \right\} \sum_{t=1}^{\text{len}(y^{(i,j)})} \log \pi_\theta\!\left(y_t^{(i,j)} \mid x^{(i)}, y_{<t}^{(i,j)}\right) \right],$$

where $\mathbb{1}\{\cdot\}$ is the indicator function that gates on whether the rollout was correct. Only successful rollouts contribute to the gradient — you're literally just doing SFT on the model's own correct samples. There is no baseline, no advantage, no clipping. That simplicity is the appeal: no critic, no reference model, no importance ratio, nothing to tune besides $Z$.

**When to use RFT:**

- **Sparse / binary rewards.** When most rollouts fail, the indicator throws them away cleanly — you don't waste gradient signal trying to "push down" bad responses. GRPO with mostly-failed groups produces near-zero advantages anyway; RFT skips the ceremony.
- **Bootstrapping / cold start.** Early in training, the model produces mostly garbage. RFT keeps the loop stable by only learning from the rare successes. Once success rate rises, you can switch to full GRPO.
- **Compute-constrained settings.** No reference model, no old policy snapshot, no KL estimator, no clipping. One forward pass per correct sample. Fits comfortably on smaller GPUs.
- **Strong verifier + iterative self-improvement.** If you can score rollouts cheaply and correctly (math, code, formal proofs), RFT + rejection sampling is a well-understood self-improvement loop (STaR, ReST, expert iteration).
- **Avoid** when success rate is very high — you throw away the signal in the small number of failures, which is often where the interesting gradient lives. Also avoid when the reward is a continuous score rather than pass/fail; RFT's indicator can't use graded signals.

#### MaxRL
The most recent method is Maximum Likelihood Reinforcement Learning, or MaxRL[4]. MaxRL proposes that instead of normalizing the advantage by the group std, we divide by the group mean. This gives the estimator

$$\hat{g} \leftarrow \frac{1}{Z} \sum_{i=1}^{B} \sum_{j=1}^{G} \sum_{t=1}^{\text{len}(y^{(i,j)})} \frac{r(y^{(i,j)} \mid x^{(i)}) - \mu_i}{\mu_i} \, \nabla_\theta \log \pi_\theta\!\left(y_t^{(i,j)} \mid x^{(i)}, y_{<t}^{(i,j)}\right).$$

Note that in the original MaxRL paper, the normalizer $Z$ is the total number of tokens in the batch $\sum_{i=1}^{B} \sum_{j=1}^{G} \text{len}(y^{(i,j)})$, which is batch-dependent. Using a constant $Z$ (as in this derivation) keeps the estimator on the same footing as the other baselines and isolates the effect of normalizing by $\mu_i$ instead of $\text{std}_i$.

**Reading the advantage $(r - \mu_i)/\mu_i$:**

This is a *relative improvement* — how much better a rollout did compared to the group average, as a fraction of the group average. Some concrete cases:

- **Hard group ($\mu_i = 0.1$, i.e. 10% success rate):** a correct rollout ($r = 1$) gets advantage $(1 - 0.1)/0.1 = 9$. A wrong rollout gets $(0 - 0.1)/0.1 = -1$. The correct rollout is amplified 9× because success on a hard prompt is a big deal.
- **Easy group ($\mu_i = 0.9$, 90% success rate):** a correct rollout gets $(1 - 0.9)/0.9 \approx 0.11$. A wrong rollout gets $-1$. Now the correct rollout barely matters — the informative signal is on the *failures*.

Compare to std normalization in standard GRPO: at $\mu_i = 0.5$, $\text{std}_i = 0.5$, so both denominators are similar. But away from 0.5, they diverge — $\mu_i$ shrinks (or grows) faster than $\text{std}_i$, so MaxRL applies more asymmetric weighting on skewed groups.

**When to use MaxRL:**

- **Bimodal / skewed groups.** When success rate within a group is far from 50%, MaxRL amplifies the rare outcome more than std normalization. Good for prompts where the model is either "mostly getting it" or "mostly missing it" — you push harder on the minority signal.
- **Hard-prompt emphasis.** If your dataset has a mix of easy and hard prompts, MaxRL naturally weights hard prompts more (small $\mu_i$ → large advantages). Useful when you want the policy to focus on frontier prompts.
- **You want a "percent improvement" interpretation.** The advantage is literally the fractional improvement over group mean, which is more interpretable than a z-score for some workflows.
- **Avoid** when $\mu_i \approx 0$ across many groups — the denominator blows up. Need to add an $\varepsilon$ or clip. Standard GRPO's std normalization is more graceful in this regime.
- **Avoid** with negative or unbounded rewards — the sign of $\mu_i$ affects the direction of the update, which is not what you want.

#### Choosing between variants

| Setting | Preferred |
|---|---|
| Verifiable reward, mid-range success rate (say 20–80%) | **Standard GRPO** — the std norm keeps groups with mixed outcomes stable |
| Very low success rate (< 10%) or cold-start training | **RFT** — most groups have zero informative gradient anyway |
| High success rate + you want to squeeze the last few % | **Dr. GRPO** — removes length bias so the tail improvements aren't washed out |
| Graded / continuous reward (e.g. reward model score) | **Standard GRPO** — RFT can't use the shading |
| Research / theoretical analysis | **Dr. GRPO** — unbiased, matches $\nabla J_\theta$ |
| Minimal infra, no reference/critic models | **RFT** — the whole loop is a fine-tune |
| Mixed easy/hard prompts, want to focus on hard ones | **MaxRL** — small $\mu_i$ amplifies hard-group gradients |
| Skewed groups (bimodal success rates far from 0.5) | **MaxRL** — asymmetric scaling emphasizes the minority outcome |


### Off-Policy RL

We have been talking about full on-policy RL, meaning every gradient estimator uses samples drawn from the model being updated. This means we took 1 training step per inference batch, and the training batch size was set equal to the inference batch size.

Off-policy RL, on the other hand, takes multiple training steps per inference batch and has the potential to speed up training. But this comes at the cost of potential instability and increased algorithm complexity.

#### Importance Sampling and Clipping

So far we have been sampling from the current policy $\pi_\theta$. But in practice we want to take multiple gradient steps on the same batch of rollouts (for sample efficiency). After the first step, $\pi_\theta \neq \pi_{\theta_{\text{old}}}$, so the samples are no longer from the right distribution.

Importance sampling corrects for this:

$$\mathbb{E}_{y \sim \pi_\theta} [f(y)] = \mathbb{E}_{y \sim \pi_{\theta_{\text{old}}}} \left[\, \frac{\pi_\theta(y \mid x)}{\pi_{\theta_{\text{old}}}(y \mid x)} \, f(y) \,\right].$$

What we are essentially doing is upweighting $y$'s that are still relevant for the current policy $\pi_\theta$, and downweighting the "stale" samples with low $\pi_0$.

Applying importance sampling per-token to the on-policy GRPO objective (equation 32) gives the off-policy version, without any clipping yet:

$$J_\theta^{\text{GRPO-off-policy-noclip}} = \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} \frac{1}{\text{len}(y^{(i,j)})} \sum_{t=1}^{\text{len}(y^{(i,j)})} \frac{r(y^{(i,j)} \mid x^{(i)}) - \mu_i}{\text{std}_i} \, \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_0(y_t \mid x, y_{<t})}, \tag{56}$$

where $\pi_0$ is the behavior policy that generated the rollouts (i.e. $\pi_{\theta_\text{old}}$).

The problem: the ratio $\rho_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_0(y_t \mid x, y_{<t})}$ can blow up when the new policy diverges from the behavior policy. A single token with $\rho_t = 20$ can dominate the whole batch's gradient.

PPO's solution — which GRPO inherits — is to clip the ratio. Let

$$A^{(i,j)} = \frac{r(y^{(i,j)} \mid x^{(i)}) - \mu_i}{\text{std}_i}, \qquad w_t^{(i,j)} = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_0(y_t \mid x, y_{<t})}$$

denote the advantage of response $j$ on prompt $i$ and the $t$-th token's importance ratio. The clipped off-policy objective becomes:

$$J_\theta^{\text{GRPO-off-policy-clip}} = \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} \frac{1}{\text{len}(y^{(i,j)})} \sum_{t=1}^{\text{len}(y^{(i,j)})} \min\!\Big( A^{(i,j)} w_t^{(i,j)}, \; A^{(i,j)} \, \text{clip}\big(w_t^{(i,j)}, [1-\epsilon, 1+\epsilon]\big) \Big). \tag{57}$$

The clipping caps how much any single token can pull the update, so a large per-token drift can't dominate the batch. Typical $\epsilon = 0.2$.

#### KL Regularization

Even with clipping, the policy can drift far from the original reference model (the SFT checkpoint) over many steps. This causes "alignment tax" — the model forgets things it knew. GRPO adds a KL penalty against a frozen reference policy $\pi_{\text{ref}}$:

$$\mathbb{D}_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big] = \mathbb{E}_{y \sim \pi_\theta} \left[\, \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \,\right].$$

In practice GRPO uses an unbiased low-variance estimator from Schulman:

$$\hat{\mathbb{D}}_{\text{KL}} = \frac{\pi_{\text{ref}}(y_{i,t} \mid x, y_{i,<t})}{\pi_\theta(y_{i,t} \mid x, y_{i,<t})} - \log \frac{\pi_{\text{ref}}(y_{i,t} \mid x, y_{i,<t})}{\pi_\theta(y_{i,t} \mid x, y_{i,<t})} - 1.$$

This is always $\geq 0$ and equals $0$ when $\pi_\theta = \pi_{\text{ref}}$.

#### The Full Off-Policy GRPO Objective

Putting it together, for a batch of prompts $\{x\}$ each with $G$ sampled responses $\{y_i\}_{i=1}^G$:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x, \{y_i\}} \left[\, \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \Big( \min(\rho_{i,t} A_i, \, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i) - \beta \, \hat{\mathbb{D}}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \Big) \,\right]$$

where

- $\rho_{i,t} = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}$ — the importance ratio at token $t$ of response $i$
- $A_i$ — the group-relative advantage of response $i$ (constant across tokens within a response)
- $\epsilon$ — clipping range (e.g. $0.2$)
- $\beta$ — KL penalty coefficient (e.g. $0.04$)

The training loop:

1. Sample a batch of prompts.
2. For each prompt, generate $G$ responses from $\pi_{\theta_{\text{old}}}$.
3. Score each response with the reward function (rule-based for math/code, or a reward model otherwise).
4. Compute group-relative advantages $A_i$ for each prompt.
5. Take several gradient steps on the GRPO objective with the same rollouts.
6. Copy $\theta \to \theta_{\text{old}}$ and repeat.


The trick is that we only sample once (from the old policy), then we score those exact tokens under both policies.

Step by step:

1. Rollout phase (using π_old): For each prompt x, sample a response y from π_old(· | x). This gives you concrete token sequences y_1, y_2, ..., y_T. You store them.
2. Freeze old_log_probs: Compute log π_old(y_t | x, y_<t) for those specific stored tokens. Save these — they're constants for the rest of training. Shape: (batch, seq).
3. Training loop (using π_θ): For those same tokens y from the rollout, compute log π_θ(y_t | x, y_<t) — i.e., "what probability would the current policy assign to the exact token that π_old sampled?" Shape: (batch, seq).
4. Importance ratio: π_θ(y_t) / π_old(y_t) — the ratio for the same token under the two policies.

So you're never comparing "what would π_θ have sampled" vs. "what π_old sampled." You're comparing "what probability π_θ assigns to π_old's sample" vs. "what probability π_old assigned to its own sample."

Why this is legit — the math:

We want the gradient of E_{y ~ π_θ}[R(y)]. But we only have samples from π_old. Importance sampling identity:

$$\mathbb{E}_{y \sim \pi_\theta}[R(y)] = \mathbb{E}_{y \sim \pi_{\text{old}}}\left[\frac{\pi_\theta(y)}{\pi_{\text{old}}(y)} R(y)\right]$$

So we can estimate an expectation under π_θ using samples from π_old, as long as we reweight by the ratio. That's why we need both log-probs for the same tokens.

Why this matters practically:

- Rollout is expensive (generation is autoregressive and slow). You want to reuse each rollout for multiple gradient steps.
- On the first gradient step, π_θ == π_old, so the ratio is exactly 1 everywhere and PPO reduces to vanilla policy gradient.
- As π_θ drifts from π_old over multiple update steps, the ratio deviates from 1 — that's when clipping kicks in to prevent trusting the importance-sampling estimate too far.
- Once π_θ has drifted too far (typically after a few epochs), you do a fresh rollout to get a new π_old.

### Why GRPO Works for Reasoning

For tasks with a verifiable reward (math, code), GRPO has three nice properties:

1. **No value network.** Halves the memory footprint vs PPO. Important when training 7B+ models.
2. **Self-normalizing.** The group baseline adapts to each prompt's difficulty — easy prompts produce small gradients, hard prompts produce informative ones.
3. **Robust to reward scale.** The std-normalization in the advantage makes the algorithm insensitive to whether rewards are $\{0, 1\}$ or $\{-10, 10\}$ — only the relative ordering within a group matters.

The main weakness is that GRPO needs $G$ rollouts per prompt, which is expensive when generation dominates the compute (long chain-of-thought responses). Picking $G$ is the main knob: too small and the baseline is noisy, too large and you waste compute.


### GSPO

In the GSPO paper[5], the authors find that GRPO is unstable and argue that we should do sequence-level reweighting instead. To get around the fact that the sequence-level ratio is a product of $L$ terms, they simply take the expression to the power $1/L$. They reweight by the geometric mean importance weight over the sequence rather than the product. The result is the following loss:

$$J_\theta^{\text{GRPO-off-policy-gspo}} = \frac{1}{BG} \sum_{i=1}^{B} \sum_{j=1}^{G} \min\!\Big( A^{(i,j)} s^{(i,j)}, \; A^{(i,j)} \, \text{clip}\big(s^{(i,j)}, [1-\epsilon, 1+\epsilon]\big) \Big), \tag{62}$$

where we now have a sequence-level importance weight $s^{(i,j)}$ that is shared across time steps:

$$s^{(i,j)} = \left( \prod_{t=1}^{\text{len}(y^{(i,j)})} \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_0(y_t \mid x, y_{<t})} \right)^{\!\frac{1}{\text{len}(y^{(i,j)})}}. \tag{63}$$

Note that without the power $\frac{1}{\text{len}(y^{(i,j)})}$ and without clipping, this objective becomes the sequence-level importance-reweighted policy gradient loss. Taking the geometric mean and clipping reduce variance at the cost of bias. One other note is that when taking the gradient, we naturally get sequence-length normalization due to the geometric mean power (clipping is ignored in the equation below for brevity).


References:

1. CS336n [Assignment 5](https://github.com/stanford-cs336/assignment5-alignment)
2. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models — [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300) (original GRPO paper)
3. Schulman et al., Proximal Policy Optimization Algorithms — [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
4. F. Tajwar et al., “Maximum Likelihood Reinforcement Learning.” [https://arxiv.org/abs/2602.02710](https://arxiv.org/abs/2602.02710)

