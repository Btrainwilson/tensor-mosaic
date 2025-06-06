# Mosaic (tensor-mosaic)

> **Flexible tensor view allocator and cache for deep learning and scientific computing.**

[![Latest Version on PyPI](https://img.shields.io/pypi/v/tensor-mosaic.svg)](https://pypi.python.org/pypi/tensor-mosaic/)
[![Supported Implementations](https://img.shields.io/pypi/pyversions/tensor-mosaic.svg)](https://pypi.python.org/pypi/tensor-mosaic/)
![Build Status](https://github.com/btrainwilson/mosaic/actions/workflows/test.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/tensor-mosaic/badge/?version=latest)](https://tensor-mosaic.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/btrainwilson/mosaic/badge.svg?branch=master)](https://coveralls.io/github/btrainwilson/mosaic?branch=master)

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

## Quick Start

import tensor_mosaic as Mosaic  # Alias locally if you like

mosaic = Mosaic()
mosaic.inputs = 16
mosaic.hidden = 32
mosaic.outputs = 10

import torch
tensor = torch.randn(1, mosaic.width)  # Reserve total width

inputs_view = mosaic.slice_view(tensor, "inputs")
outputs_view = mosaic["outputs"]

print("Inputs:", inputs_view)
print("Outputs:", outputs_view)

