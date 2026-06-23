# Architecture and Experiments Analysis

## Continuous Space Partially Observable Markov Decision Process

We define a countiuous space, partially observable, Markov Decision Process as the tuple POMDP 7-tuple: ($\mathcal{S}, \mathcal{O}, \mathcal{A}, P, r, \gamma, \mathbb{S}$), where $\mathcal{S}$ is the set of states, $\mathcal{O}$ is the set of observations, $\mathcal{A}$ is the set of actions, $P \colon \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0,1]$ is the transition kernel, $(s, a, s') \mapsto P(s' \mid s, a)$, with $\sum_{s' \in \mathcal{S}} P(s' \mid s, a) = 1$, $r \colon \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ is the reward function, $(s, a) \mapsto r(s, a)$, $\gamma \in [0,1)$ is the discount factor, $\mathbb{S} \in \Delta(\mathcal{S})$ is the initial state distribution, $\mathbb{S}(s) = \Pr(s_0 = s)$.

An episode will begine by sampling $s_{0}$ from $\mathbb{S}$

## Introduction to Actor Critic Algorithms

## Asymmetric Actor Critic

## Soft Actor Critic Algorithm

## Set Attention Encoder

## Baseline, Training and Evaluation

## Experiment Results

## Discussion

