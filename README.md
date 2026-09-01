# Financial Time Series Generation using TBC-GAN

A deep learning framework for generating realistic synthetic financial time series using a hybrid WGAN-GP architecture with Transformer attention, BiLSTM sequence modeling, CNN-based temporal feature extraction, and automated hyperparameter optimization.

The project focuses on generating synthetic volatility time series based on the VSTOXX / EURO STOXX 50 volatility index while preserving important statistical and temporal characteristics of real financial data.

## Problem Statement

Financial time series are difficult to model because market behavior is highly non-linear and contains complex temporal dependencies.

Real financial data can exhibit:

- Volatility clustering
- Heavy-tailed distributions
- Non-linear relationships
- Short-term fluctuations
- Long-range temporal dependencies
- Extreme market movements

Traditional forecasting models can estimate future values, but they are not designed to generate large numbers of realistic alternative market scenarios.

This project addresses that problem by learning the underlying distribution and temporal structure of historical financial data and using that knowledge to generate new synthetic financial sequences.

Potential applications include:

- Risk analysis
- Portfolio simulation
- Stress testing
- Monte Carlo simulation
- Machine learning data augmentation
- Financial strategy evaluation

## Project Overview

The core model is a Transformer-based Conditional Generative Adversarial Network (TBC-GAN) built on the WGAN-GP framework.

The architecture combines:

- Transformer
- BiLSTM
- CNN
- WGAN-GP
- TimeGAN-inspired supervised learning
- Optuna-based hyperparameter optimization

The main idea is to combine different architectures so that each component captures a different aspect of financial time-series behavior.

```text
                   Historical Financial Data
                              |
                              v
                       Data Preprocessing
                              |
                              v
                     Sliding Window Sequences
                              |
                              v
                         Random Noise
                              |
                              v
                   +-----------------------+
                   |      Transformer      |
                   |    Self-Attention     |
                   | Global Dependencies   |
                   +-----------------------+
                              |
                              v
                   +-----------------------+
                   |        BiLSTM         |
                   |  Temporal Memory      |
                   | Long-Range Patterns   |
                   +-----------------------+
                              |
                              v
                   +-----------------------+
                   |         CNN           |
                   | Local Temporal        |
                   | Feature Extraction    |
                   +-----------------------+
                              |
                              v
                   Synthetic Time Series
                              |
                              v
                   +-----------------------+
                   |      Critic /         |
                   |     Discriminator     |
                   |     CNN + BiLSTM      |
                   +-----------------------+
                              |
                              v
                       WGAN-GP Feedback
                              |
                              v
                        Model Update
                              |
                              v
                    Improved Generator
