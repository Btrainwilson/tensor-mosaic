import torch
from typing import Dict, Tuple, Union, Optional, Callable, Any
from . import packers

class Mosaic:
    def __init__(self, cache=True, device='cpu', autocompile=True):
        self.cache = cache
        self.device = torch.device(device)
        self.alias_map: Dict[str, Tuple[slice, ...]] = {}
        self.tensor_map: Dict[str, torch.Tensor] = {}
        self.requests: Dict[str, Tuple[int, ...]] = {}
        self._ndim: Optional[int] = None
        self.strategy: str = "greedy"
        self._compiled = False
        self._bin_shape: Optional[Tuple[int, ...]] = None
        self._packer_map = {
            "greedy": packers.greedy_packer,
            "rectpack": packers.rectpack_packer,
            # Add more strategies here if desired
        }
        self._autocompile = autocompile

    def _normalize_shape(self, shape: Union[int, Tuple[int, ...], list]) -> Tuple[int, ...]:
        if isinstance(shape, int):
            return (shape,)
        elif isinstance(shape, tuple):
            return shape
        elif isinstance(shape, list):
            return tuple(shape)
        raise TypeError("Shape must be int, tuple, or list of ints.")

    def add(self, alias: str, shape: Union[int, Tuple[int, ...], list]):
        shape = self._normalize_shape(shape)
        if self._ndim is None:
            self._ndim = len(shape)
        elif len(shape) != self._ndim:
            raise ValueError(f"All requests must have the same number of dimensions ({self._ndim}); got {shape}")
        self.requests[alias] = shape
        self._compiled = False
        if self._autocompile:
            self.compile()

    def __setattr__(self, name, value):
        if name in {
            "cache", "device", "alias_map", "tensor_map", "requests", "_ndim", "strategy",
            "_compiled", "_bin_shape", "_packer_map", "_autocompile"
        }:
            super().__setattr__(name, value)
        elif isinstance(value, (int, tuple, list)):
            self.add(name, value)
        else:
            raise TypeError("Only int, tuple, or list supported for allocation.")

    def __getitem__(self, name) -> Union[Tuple[slice, ...], torch.Tensor]:
        return getattr(self, name)

    def __getattr__(self, name) -> Union[Tuple[slice, ...], torch.Tensor]:
        if not self._compiled:
            self.compile()
        if self.cache and name in self.tensor_map:
            return self.tensor_map[name]
        elif name in self.alias_map:
            return self.alias_map[name]
        raise AttributeError(f"'Mosaic' has no attribute '{name}'")

    def compile(self, strategy: Optional[str]=None, packer: Optional[Callable]=None):
        if strategy:
            self.strategy = strategy
        if packer is None:
            packer = self._packer_map.get(self.strategy)
            if packer is None:
                raise ValueError(f"Unknown packing strategy '{self.strategy}'")
        allocation, bin_shape = packer(self.requests)
        self.alias_map.clear()
        self.tensor_map.clear()
        self._bin_shape = bin_shape
        for alias, slices in allocation.items():
            self.alias_map[alias] = slices
            if self.cache:
                idx_ranges = [torch.arange(s.start, s.stop, device=self.device) for s in slices]
                grid = torch.meshgrid(*idx_ranges, indexing='ij')
                idx_tensor = torch.stack([g.flatten() for g in grid], dim=-1)
                self.tensor_map[alias] = idx_tensor

        self._compiled = True

    @property
    def bin_shape(self) -> Tuple[int, ...]:
        if not self._compiled:
            self.compile()
        return self._bin_shape

    def bin_tensor(self, fill_value=0, dtype=None) -> torch.Tensor:
        if not self._compiled:
            self.compile()
        dtype = dtype or torch.float
        return torch.full(self.bin_shape, fill_value, dtype=dtype, device=self.device)

    def pretty_print(self):
        print("\nMosaic Allocations:")
        for alias, slc in self.alias_map.items():
            print(f"{alias:10} : {slc}")
        print("Bin shape:", self.bin_shape)

    def slice_view(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if not self._compiled:
            self.compile()
        slc = self.alias_map[name]
        return x[slc]

# Example usage:
if __name__ == "__main__":
    mosaic = Mosaic(cache=True, device='cpu')
    mosaic.D = (2, 3)
    mosaic.E = [2, 3]
    mosaic.compile()

    mosaic.pretty_print()
    bin_tensor = mosaic.bin_tensor(fill_value=-1, dtype=torch.long)
    print("Bin tensor:\n", bin_tensor)
    print("Slice for D:\n", mosaic.slice_view(bin_tensor, "D"))
