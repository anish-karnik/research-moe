# 🧠 Efficient Transformers via Mixture-of-Experts (MoE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

*A research implementation of sparsification techniques for Transformer models using dynamic expert routing*

## 📌 Table of Contents
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Gating Mechanisms](#-gating-mechanisms)
- [Results](#-results)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)

## 🚀 Key Features

### Implemented Gating Strategies
| Mechanism | Learnable | Load Balanced | Differentiable |
|-----------|-----------|---------------|----------------|
| Top-K     | ✓         | ✗             | ✗              |
| Noisy Top-K | ✓       | ✓             | ✗              |
| Soft MoE  | ✓         | ✓             | ✓              |
| Hash-based | ✗        | ✓             | ✗              |
| LSH-based | ✗         | ✓             | ✗              |
| GNN-based | ✓         | ✓             | ✓              |

### Benchmarking Toolkit
- FLOPs calculation
- Memory profiling
- Inference latency tracking
- Expert utilization visualization
