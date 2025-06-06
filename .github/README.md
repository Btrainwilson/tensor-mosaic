# Mosaic <br><sub><sup>(tensor-mosaic)</sup></sub>

<p align="center">
  <img alt="Mosaic logo" src="https://github.com/btrainwilson/tensor-mosaic/blob/main/.github/logo.png?raw=true" width="150">
</p>

<p align="center">
  <b>Flexible tensor view allocator and cache for deep learning and scientific computing.</b>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/tensor-mosaic">
    <img src="https://img.shields.io/pypi/v/tensor-mosaic.svg" alt="PyPI version">
  </a>
  <a href="https://pypi.python.org/pypi/tensor-mosaic">
    <img src="https://img.shields.io/pypi/pyversions/tensor-mosaic.svg" alt="Supported Python versions">
  </a>
  <a href="https://github.com/btrainwilson/mosaic/actions/workflows/test.yaml">
    <img src="https://github.com/btrainwilson/mosaic/actions/workflows/test.yaml/badge.svg" alt="Build Status">
  </a>
  <a href="https://tensor-mosaic.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/tensor-mosaic/badge/?version=latest" alt="Documentation Status">
  </a>
  <a href="https://coveralls.io/github/btrainwilson/mosaic?branch=master">
    <img src="https://coveralls.io/repos/github/btrainwilson/mosaic/badge.svg?branch=master" alt="Coverage Status">
  </a>
</p>

---

## Overview

**Mosaic** is a lightweight, extensible Python library for allocating and caching non-overlapping, named regions (views or slices) within large or dynamically constructed tensors.

It is especially useful for deep learning, scientific computing, and any application requiring efficient management and lookup of structured tensor subspaces.

- **Reserve and cache named slices:** Create stable, named regions within tensors for models, optimizers, or data pipelines.
- **Device-aware:** Automatically keeps cached indices and slices on the desired device (CPU/GPU).
- **Zero-copy views:** Retrieve direct views into the underlying tensor without data duplication.
- **Compatible with PyTorch and NumPy.**

---


## Installation

```bash
pip install tensor-mosaic
```


## Quick Start



---
```
import torch
import tensor_mosaic as mosaic  # Alias locally if you like

mosaic = mosaic.Mosaic()
mosaic.INPUTS = 16
mosaic.HIDDEN = 32
mosaic.OUTPUTS = 10

tensor = torch.randn(1, *mosaic.shape)  # Reserve total width

inputs_view = mosaic.slice_view(tensor, "inputs")
outputs_view = mosaic["outputs"]

print("Inputs:", inputs_view)
print("Outputs:", outputs_view)
```

