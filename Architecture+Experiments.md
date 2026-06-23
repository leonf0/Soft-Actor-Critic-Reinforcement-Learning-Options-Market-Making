# Architecture and Experiments Analysis

## Introduction and Motivation

## Continuous Space Partially Observable Markov Decision Process

We define a countiuous space, partially observable, Markov Decision Process as the tuple POMDP 7-tuple: ($\mathcal{S}, \mathcal{O}, \mathcal{A}, \mathcal{P}, r, \gamma, \mathbb{S}$), where $\mathcal{S}$ is the set of states, $\mathcal{O}$ is the set of observations, $\mathcal{A}$ is the set of actions, $\mathcal{P} \colon \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$ is the transition kernel, $(s, a, s') \mapsto P(s' \mid s, a)$, with $\sum_{s' \in \mathcal{S}} P(s' \mid s, a) = 1$, $r \colon \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the reward function, $(s, a) \mapsto r(s, a)$, $\gamma \in [0,1)$ is the discount factor, $\mathbb{S} \in \Delta(\mathcal{S})$ is the initial state distribution, $\mathbb{S}(s) = \Pr(s_0 = s)$.

An episode will begine by sampling $s_{0}$ from $\mathbb{S}$, at every timestep $t$ the agent then takes an action $a_{t} = \pi(o_{t})$ according to its policy $\pi : \mathcal{O} \to \mathcal{A}$. The agent then gets a reward $r_{t} = r(s_{t}, a_{t})$ and transitions to $s_{t+1}$ according to $\mathcal{P}(s_{t}, a_{t}, \cdot)$.

The goal of the agent is to maximise its expected return 

$$\mathbb{E}_{\mathbb{S}}\left[ R_0 \mid \mathbb{S} \right] = \mathbb{E}_{s_0 \sim \mathbb{S}}\left[ \sum_{i=0}^{\infty} \gamma^{i} r_i \right]$$

Another relavent term to define is the Q-function, which is defined as $Q^{\pi}(s_t, a_t) = \mathbb{E}\left[ R_t \mid s_t, a_t \right]$, where in the case of partial observability the agent acts based on the partial observation $o_{t}$, meaning $a_{t} = \pi(o_{t})$.

## Introduction to Actor Critic Algorithms

## Asymmetric Actor Critic

## Soft Actor Critic Algorithm

## Set Attention Encoder

## Baseline, Training and Evaluation

## Experiment Results

## Discussion

