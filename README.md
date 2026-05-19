# Pricing Basket Options in the Black-Scholes Model

This repository contains an exploration and implementation of pricing techniques for a **two-asset basket option** within the multi-dimensional Black-Scholes framework. Since there is no closed-form analytical solution to value these types of options directly, this project focuses on implementing a standard closed-form approximation and evaluating its accuracy against various **Monte Carlo simulation** techniques.

This project was developed as part of the **PRB222 course**.

## 📌 Project Overview
The core objective is to price a basket option (Call or Put) whose underlying payoff depends on a weighted sum of two correlated assets, $\alpha S_1(T) + \beta S_2(T)$. The project evaluates the trade-offs between computational speed (analytical approximation) and numerical precision (Monte Carlo methods).

The asset dynamics follow a 2D Black-Scholes model:
$$dS_i(t) = S_i(t)(rdt + \sigma_i dW_i(t))$$
Where $W_1$ and $W_2$ are two Brownian motions with a correlation coefficient $\rho \in ]-1, 1[$.

## 🚀 Key Features & Solved Problems

### 1. Analytical Log-Normal Approximation
* **Moment Matching:** Approximates the weighted sum of log-normal assets with a single log-normal asset $S_B(t)$ by matching the first and second moments:
  $$\mathbb{E}[\alpha S_1(T) + \beta S_2(T)] = \mathbb{E}[S_B(T)]$$
  $$\mathbb{E}[(\alpha S_1(T) + \beta S_2(T))^2] = \mathbb{E}[(S_B(T))^2]$$
* **Closed-Form Pricing:** Implements the approximated option price analytically using the **Abramowitz & Stegun** numerical approximation for the cumulative distribution function (CDF) of the standard normal distribution.

### 2. Monte Carlo Simulation Framework
* **Standard Monte Carlo:** Simulates correlated Brownian motions $W(T)$ using only a uniform random number generator.
* **Variance Reduction Techniques:** To improve efficiency and speed up convergence, the following methods are implemented and compared:
  * **Conditioning:** Reducing variance by conditioning on a subset of the random variables.
  * **Control Variates (Put-Call Parity):** Utilizing analytical relationships to reduce estimator variance.
  * **Control Variates (Geometric Basket):** Using the analytical pricing of a geometric basket option as a control variate for the arithmetic basket option.

### 3. Empirical Analysis & Sensitivity (Greeks)
* **Convergence & Confidence Intervals:** Computes empirical variances and plots the estimators alongside their **90% asymptotic confidence intervals** against the number of simulation paths.
* **Parameter Sensitivity:** Benchmarks the Log-Normal approximation against the variance-reduced Monte Carlo method by plotting prices and error differences across variations of:
  * Asset correlation ($\rho \in ]-1, 1[$)
  * Asset initial values ($\alpha S_1,0 \in [0, 2]$)
  * Strike price ($K \in [1, 3]$)
* **Delta Risk Measure ($\Delta_1$):** Approximates the sensitivity of the option price with respect to the initial price of the first asset ($\Delta_1 = \frac{\partial P}{\partial S_{1,0}}$) using numerical differentiation combined with conditional Monte Carlo methods.

## 🛠️ Default Simulation Parameters
Unless specified otherwise in specific sensitivity tests, the model uses the following annual parameters:
* **Volatilities:** $\sigma = (0.35, 0.40)$
* **Initial Weights:** $\alpha S_{1,0} = \beta S_{2,0} = 1$
* **Correlation:** $\rho = 0.3$
* **Strike Price:** $K = 2$
* **Risk-free Rate:** $r = 0.01$
* **Maturity:** $T = 2$ years

## 📂 Repository Structure
* `/src` or `main script`: Contains the implementation of the simulation algorithms, moment matching formulas, and the Abramowitz & Stegun approximation.
* `/plots`: Graphical comparisons of pricing models, confidence intervals, and Delta sensitivities.
* `Basket_Options.pdf`: The original project assignment/guidelines.
