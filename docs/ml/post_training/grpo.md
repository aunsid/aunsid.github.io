# GRPO


###  Language Model as policies
In traditional RL, we think about a policy $π$ which takes in a state $s_t$, and outputs an action $a_t$. This action results in a reward $r_t = r(s_t, a_t)$, and a subsequent state $s_{t+1} \sim$ next_state $(s_t, a_t)$

A language model $π_\theta$ with parameters $\theta$ defines a probability distribution over the next token $y_t$ given the the current text prefix $y_{<t} = (y_1, \dots, y_{t-1})$.
In the context for RL, we can think of the next token $y_t$ as our action $a_t$ and the current text prefix $y<t$ as the state $s_t$. Hence, the LM is a categorical stochastic policy.
$$a_t \sim π_{\theta}(.|s_t) = [softmax(f_{\theta}(s_t))]_{a_{t}}$$

To optimize a policy with policy gradients, we need two operations:
1. Sampling from the policy: drawing an action $a_{t}$ from the distribution
2. Scoring the log-likelihood of an action: evaluatng $log π_{\theta}(a_t | s_t)$

$s_t$ is the partial completion and each $a_t$ is the next token of the solution. The episode ends with the <|end_of_text|> token.

###  Trajectories
In reinforcement learning, we start at an initial state drawn from starting distribtion. We then sample an action$a_t$ from the policy $π_t$, transistion to the next state according to some next state distribution $s_{t+1} \sim$ next_state $(s_t, a_t)$ and repeat. This series of states and actions then forms a trajectory.

$$ \tau = (s_0, a_0, s_1, a_1, \dots , s_T, a_T)$$

where T is the length of the trajectoru, i.e., $a_T$ is an end-of-text token or we have reached the maximum budget of tokens. Trajectories are also called episodes or rollouts. 


### Rewards and Return
In RL, a scalar reward  $r_t = r(s_t, a_t)$ judges the immediate quality of action $a_t$ taken at state $s_t$. But for our use case, we observe zero reward for intermediate reasoning steps, until we output the answer. We then receive a verified reward on the terminal action.

$$r_T = r(s_T, a_T) := \begin{cases} 1 & \text{if trajectory } \tau \text{ matches the ground-truth according to our reward function} \\ 0 & \text{otherwise} \end{cases}.$$

In our case, $r_t$ is 1 if our final answer is correct, and 0 otherwise. The Return$R(\tau)$ aggregates the rewards along the trajectory, and in our case it just $r_T$


The objective is to maximize the expected return
$$J_\theta = E_{\tau \sim π_{\theta}}[r(\tau)]$$

where $\tau \sim π_{\theta}$ is the distribution over trajectories where we first sample $s_0 \sim \rho $ and we then sample actions from the policy and the next states from the next state distribution.
In our case, this means we first sampling our math problem from out dataset, and then sampling tokens until we hit the end of our response. A process we can denote by $y \sim π_{\theta}(y|x)$. This objective leads to the optimization of 

$$\theta^* = \arg\max_\theta J_\theta$$

### Policy Gradients
Instead of using action $a_t$ and states $s_t$ we will use $(x, y)$ denoting prompts and responses. 

We want to optimize the model's expected reward(i.e accuracy), given by

$$J_\theta = \mathbb{E}_{x \sim \rho} \mathbb{E}_{y \sim \pi_\theta(y \mid x)} [r(y \mid x)]\tag{1}$$


where r(y|x) denotes whether the response y is correct for prompt x. Our approach will be to do gradient ascent on the objective -

$$\theta_{k+1} = \theta_k + \alpha \nabla_\theta J_{\theta_k},\tag{2}$$

The way that equation 1 is read, that we draw a prompt from our training dataset of prompts. The inner expectation, given the prompt, we draw a responsey the policy could reproduce from the policy's distribution. 
$$\mathbb{E}{y \sim \pi\theta(\cdot \mid x)}[r(y \mid x)] = \sum_{y} \pi_\theta(y \mid x) , r(y \mid x). \tag{3}$$

This sum is astronomical and there are many possible token sequences that we cannot ennumerate. 

So in practice you replace it with a Monte Carlo estimate: sample a finite number of responses and average. That gives an unbiased estimate of the true expectation. Two knobs:

  1. How many prompts you sample per batch (estimating the outer expectation).
  2. How many responses you sample per prompt (estimating the inner one).

  The second knob is where different algorithms diverge:

  - Vanilla REINFORCE-style policy gradient: often just 1 response per prompt. Cheap, noisy.
  - Group-relative methods like GRPO: intentionally sample $G$ responses per prompt (e.g., 4, 8, 16) so you can compare them against each other and use the group mean as a baseline for
  variance reduction.

  Coming back to equation 2, we need to compute $\nabla_\theta J_{\theta_k}$ the gradient from samples.

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

### Importance Sampling and Clipping

So far we have been sampling from the current policy $\pi_\theta$. But in practice we want to take multiple gradient steps on the same batch of rollouts (for sample efficiency). After the first step, $\pi_\theta \neq \pi_{\theta_{\text{old}}}$, so the samples are no longer from the right distribution.

Importance sampling corrects for this:

$$\mathbb{E}_{y \sim \pi_\theta} [f(y)] = \mathbb{E}_{y \sim \pi_{\theta_{\text{old}}}} \left[\, \frac{\pi_\theta(y \mid x)}{\pi_{\theta_{\text{old}}}(y \mid x)} \, f(y) \,\right].$$

The ratio $\rho_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})}$ can blow up when the new policy diverges from the old one. PPO's solution — which GRPO inherits — is to clip the ratio:

$$\mathcal{L}_{\text{clip}}(\theta) = \min\big(\rho_t(\theta) A_i, \; \text{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon) A_i\big).$$

The clipping caps how much the policy can update per step. Typical $\epsilon = 0.2$.

### KL Regularization

Even with clipping, the policy can drift far from the original reference model (the SFT checkpoint) over many steps. This causes "alignment tax" — the model forgets things it knew. GRPO adds a KL penalty against a frozen reference policy $\pi_{\text{ref}}$:

$$\mathbb{D}_{\text{KL}}\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big] = \mathbb{E}_{y \sim \pi_\theta} \left[\, \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \,\right].$$

In practice GRPO uses an unbiased low-variance estimator from Schulman:

$$\hat{\mathbb{D}}_{\text{KL}} = \frac{\pi_{\text{ref}}(y_{i,t} \mid x, y_{i,<t})}{\pi_\theta(y_{i,t} \mid x, y_{i,<t})} - \log \frac{\pi_{\text{ref}}(y_{i,t} \mid x, y_{i,<t})}{\pi_\theta(y_{i,t} \mid x, y_{i,<t})} - 1.$$

This is always $\geq 0$ and equals $0$ when $\pi_\theta = \pi_{\text{ref}}$.

### The Full GRPO Objective

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

### Why GRPO Works for Reasoning

For tasks with a verifiable reward (math, code), GRPO has three nice properties:

1. **No value network.** Halves the memory footprint vs PPO. Important when training 7B+ models.
2. **Self-normalizing.** The group baseline adapts to each prompt's difficulty — easy prompts produce small gradients, hard prompts produce informative ones.
3. **Robust to reward scale.** The std-normalization in the advantage makes the algorithm insensitive to whether rewards are $\{0, 1\}$ or $\{-10, 10\}$ — only the relative ordering within a group matters.

The main weakness is that GRPO needs $G$ rollouts per prompt, which is expensive when generation dominates the compute (long chain-of-thought responses). Picking $G$ is the main knob: too small and the baseline is noisy, too large and you waste compute.


References:

1. CS336n [Assignment 5](https://github.com/stanford-cs336/assignment5-alignment)
2. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models — [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300) (original GRPO paper)
3. Schulman et al., Proximal Policy Optimization Algorithms — [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

