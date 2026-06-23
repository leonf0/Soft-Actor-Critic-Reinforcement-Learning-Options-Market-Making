# Architecture and Experiments Analysis

## Introduction and Motivation

A market maker posts a two-sided quote, offering a price they are willing to buy at (bid) and a price thet are willing to sell at (ask). By doing this they earns
the spread between the two values but in doing so inherits the book's risk. We define the following risk measurements:

**Delta**: The rate of change of the option's price with respect to a $1 change in the price of the underlying asset.

$$\Delta = \frac{\partial V}{\partial S}$$

**Gamma**: The rate of change of Delta with respect to a $1 change in the underlying asset's price. It is the second derivative of the option's value.

$$\Gamma = \frac{\partial \Delta}{\partial S} = \frac{\partial^2 V}{\partial S^2}$$

**Vega**: The rate of change of the option's price with respect to a 1% change in the implied volatility of the underlying asset.

$$\nu = \frac{\partial V}{\partial \sigma}$$

**Vanna**: The rate of change of Delta with respect to a 1% change in implied volatility, mathematically identical to the rate of change of Vega with respect to a $1 change in the underlying price.

$$\text{Vanna} = \frac{\partial \Delta}{\partial \sigma} = \frac{\partial \nu}{\partial S} = \frac{\partial^2 V}{\partial S \, \partial \sigma}$$

Whilst delta is hedgeable by taking positions in the underlying asset, this comes with transaction costs; on the other hand vega/gamma/vanna the book accumulates under stochastic volatility has method for offsetting risk, it can only be managed by how you quote (resevation price skew, and spread around this price).

The variance process that drives this risk is unobservable, meaning the market-maker acts on noisy proxies (such as order flow, IV estimates, realised-vol EWMAs). Another problem is adverse selection, Clustered, possibly informed flow can pick off one side of the quotes, so optimal quoting must take into account order flow and intensity, not just inventory. Many of the instruments an options market maker quotes share one underlying and overlapping vol exposure, this means quotes aren't separable, and a fill anywhere in any of these instruments will effect book-level Greeks.

Classic models like Avellaneda–Stoikov assume independent or single assets, known parametric dynamics, linear/quadratic risk, an observable state, and dont take microstructure into account. A reinforcement learning approach however learns a policy directly from simulator interaction with no closed-form value function, this allows it to develop an understanding of the complexity that breaks classical analytical methods.

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

