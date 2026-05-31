# Deep Bluffing: Assignment 1

You will build three core algorithms: Value Iteration, Policy Iteration, and Q-Learning. You have been provided with three Python starter files. Your task is to complete the mathematical updates within the `TODO` blocks.

> **Note:** Do not modify the function signatures or the boilerplate environment setup. Focus solely on translating the Bellman equations into Python logic.

---

## Task 1: Value Iteration (Model-Based)

**File:** `value_iteration.py`

Value Iteration is a dynamic programming algorithm that computes the optimal value function $v_*(s)$ assuming we have perfect knowledge of the environment's transition probabilities $\mathcal{P}_{ss'}^a$ and rewards $\mathcal{R}_s^a$.

The algorithm iteratively updates the value of each state using the Bellman Optimality Equation until the maximum change across all states is less than a threshold $\theta$:


$$v_{\*}(s)=\max_{a}\left(\mathcal{R}_{s}^{a}+\gamma\sum_{s'\in S}\mathcal{P}_{ss'}^{a}v_{\*}(s')\right)$$

**Your Task:**
Locate the nested loops in `value_iteration.py`.

1. Compute the expected return (Q-value) for taking a specific action $a$ in state $s$.
2. Update the new value of state $s$ to be the maximum of these Q-values.

---

## Task 2: Policy Iteration (Model-Based)

**File:** `policy_iteration.py`

While Value Iteration iterates over values, Policy Iteration directly updates the policy itself. It consists of two repeating steps: evaluating the current policy, and strictly improving it.

**Your Task:**

1. **Policy Evaluation:** Calculate the expected value of following the *current* policy's chosen action.
2. **Policy Improvement:** For each state, check if there is an action that yields a higher expected return than the current policy's action. If so, update the policy to choose the new $\arg\max_a$ action.

---

## Task 3: Q-Learning on a Frozen Lake (Model-Free)

**File:** `frozen_lake_agent.py`

The agent is placed on a frozen lake and must learn the environment dynamics purely through trial and error. Because the agent does not know $\mathcal{P}_{ss'}^a$, it cannot calculate $V(s)$. It must learn the Action-Value function $Q(s,a)$ directly. You will guide an agent across a frozen lake to reach a treasure. After finishing the code, run `visualize_agent.py` to see it in action!

**Your Task:**

1. **Epsilon-Greedy Exploration:** Implement the logic to balance exploration (taking a random action) and exploitation (taking the best known action).
2. **Temporal-Difference Update:** Update the Q-Table using the off-policy Bellman equation:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\left[R+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]$$


## How to Submit
Simply push your code to the `main` branch. 
