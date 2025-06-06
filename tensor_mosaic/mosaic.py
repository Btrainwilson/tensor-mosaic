import torch
from typing import Dict, Tuple, Union, Optional, Callable, Any
from . import packers
from .cache import SpaceCache

class Mosaic:
    def __init__(self, device="cpu", cache=True, autocompile=True, strategy="greedy", batched=False):
        self.device = torch.device(device)
        self.requests: Dict[str, Tuple[int, ...]] = {}
        self.slices: Dict[str, Tuple[slice, ...]] = {}
        self.bin_shape: Optional[Tuple[int, ...]] = None
        self._compiled = False
        self._ndim: Optional[int] = None
        self.cache_indices = cache
        self.indices = SpaceCache(self.device) if cache else None
        self._packer_map: Dict[str, Callable] = {
            "greedy": packers.greedy_packer,
            # "knapsack": knapsack_packer,
        }
        self._strategy = strategy
        self._packer = self._packer_map[strategy]
        self.autocompile = autocompile
        self.batched = batched

    def add(self, name: str, shape: Union[int, Tuple[int, ...], list]):
        if isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, list):
            shape = tuple(shape)
        if self._ndim is None:
            self._ndim = len(shape)
        elif len(shape) != self._ndim:
            raise ValueError(f"All allocations must have dimension {self._ndim}, got {shape}")
        self.requests[name] = shape
        self._compiled = False
        if self.autocompile:
            self.compile()

    def __setattr__(self, name, value):
        if name in {
            "batched", "packer", "strategy", "device", "requests", "slices", "bin_shape", "_compiled", "_ndim",
            "cache_indices", "indices", "_packer_map", "_packer", "_strategy", "autocompile"
        }:
            super().__setattr__(name, value)
        elif isinstance(value, (int, tuple, list)):
            self.add(name, value)
        else:
            raise TypeError("Allocation must be int, tuple, or list")

    def __getitem__(self, name):
        if not self._compiled:
            self.compile()
        if self.cache_indices and self.indices and name in self.indices:
            return self.indices[name]
        return self.slices[name]

    def __getattr__(self, name):
        return self.__getitem__(name)

    def compile(self, packer: Optional[Callable] = None):
        packer = packer or self.packer
        allocs, bin_shape = packer(self.requests)
        self.slices = allocs
        self.bin_shape = bin_shape
        self._compiled = True
        if self.cache_indices and self.indices:
            self.indices.clear()
            for name, slices in allocs.items():
                idx_ranges = [torch.arange(s.start, s.stop, device=self.device) for s in slices]
                grid = torch.meshgrid(*idx_ranges, indexing="ij")
                idx_tensor = torch.stack([g.flatten() for g in grid], dim=-1)
                self.indices[name] = idx_tensor.squeeze()
                if self.batched:
                    self.indices[name] = self.indices[name].unsqueeze(0)

    def bin_tensor(self, fill_value=0, dtype=None):
        if not self._compiled:
            self.compile()
        dtype = dtype or torch.float
        return torch.full(self.bin_shape, fill_value, dtype=dtype, device=self.device)

    @property
    def shape(self):
        return self.bin_shape

    def pretty_print(self):
        print("\nMosaic Allocations:")
        for k, v in self.slices.items():
            print(f"{k:10}: {v}")
        print(f"Bin shape: {self.bin_shape}")

    def slice_view(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if not self._compiled:
            self.compile()
        return x[self.slices[name]]

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value: str):
        if value not in self._packer_map:
            raise ValueError(f"Unknown packing strategy: {value}")
        self._strategy = value
        self._packer = self._packer_map[value]
        self._compiled = False  # Force recompile on next access

    @property
    def packer(self):
        return self._packer

    @packer.setter
    def packer(self, value: Callable):
        self._packer = value
        self._compiled = False  # Force recompile on next access


