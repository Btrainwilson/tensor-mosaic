from typing import Dict, Tuple, Union, Optional, Callable, Any, List
from .backend import TorchBackend, NumpyBackend, JaxBackend
from .packers import greedy_packer
from .slicemanager import BinManager

BACKEND_MAP = {
    "torch": TorchBackend,
    "numpy": NumpyBackend,
    "jax":   JaxBackend,
}

class Mosaic:

    def __init__(self, dim, backend="torch", device=None, cache=True, autocompile=True, strategy="greedy", batched=False):
        self.backend_name = backend
        self.backend = BACKEND_MAP[backend](device) if backend != "numpy" else BACKEND_MAP[backend]()
        self.device = device
        self.bin_manager = BinManager(dim=dim)
        self.cache_indices = cache
        self.indices = {}
        self._allocation_recipe: List[Dict] = []
        self._packer_map: Dict[str, Callable] = {
            "greedy": greedy_packer,
        }
        self._strategy = strategy
        self.autocompile = autocompile
        self.batched = batched

    # ---- BinManager Pass-Through Methods ----
    def add(self, name: str, shape=None, region=None):
        self.bin_manager.add(name, shape=shape, region=region)
        # Save recipe for serialization
        self._allocation_recipe.append({
            "name": name,
            "shape": shape if shape is not None else None,
            "region": region if region is not None else None,
        })
        if self.autocompile:
            self.compile()

    def __setattr__(self, name, value):
        # Allow normal setting for special/internal names
        if name in {
            "backend", "backend_name", "device", "bin_manager", "cache_indices", "indices",
            "_packer_map", "_strategy", "strategy", "packer", "autocompile", "batched", "_allocation_recipe"
        }:
            super().__setattr__(name, value)
        # Pass attribute assignments to BinManager
        elif hasattr(self, "bin_manager"):
            if isinstance(value, (int, tuple, list)):
                self.add(name, shape=value)
            else:
                self.add(name, region=value)
        else:
            super().__setattr__(name, value)

    def __getitem__(self, name):
        return self.bin_manager[name]

    def __getattr__(self, name):
        if "bin_manager" in self.__dict__:
            try:
                return getattr(self.bin_manager, name)
            except AttributeError:
                raise AttributeError(f"'Mosaic' object and its BinManager have no attribute '{name}'")
        else:
            raise AttributeError(f"'Mosaic' object has no attribute '{name}'")

    def compile(self, packer: Optional[Callable] = None):
        packer = packer or self._packer_map[self._strategy]
        self.bin_manager.compile(packer)
        # Build indices for each region
        if self.cache_indices:
            self.indices.clear()
            for name, region in self.bin_manager.slices.items():
                idx_ranges = [self.backend.arange(s.start, s.stop) for s in region]
                grid = self.backend.meshgrid(idx_ranges)
                idx_tensor = self.backend.stack([g.flatten() for g in grid], axis=-1)
                idx_tensor = idx_tensor.squeeze()
                if self.batched:
                    idx_tensor = self.backend.stack([idx_tensor], axis=0)
                self.indices[name] = idx_tensor

    def bin_tensor(self, fill_value=0, dtype=None):
        if not self.bin_manager._compiled:
            self.compile()
        return self.backend.full(self.bin_manager.shape, fill_value, dtype=dtype)

    @property
    def shape(self):
        return self.bin_manager.shape

    def pretty_print(self):
        print(f"\nMosaic Allocations (backend={self.backend_name}):")
        for k, v in self.bin_manager.slices.items():
            print(f"{k:10}: {v}")
        print(f"Bin shape: {self.shape}")

    def slice_view(self, x, name: str):
        if not self.bin_manager._compiled:
            self.compile()
        return x[self.bin_manager[name]]

    # --------- Serialization & Reload Support ---------
    def save_allocations(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self._allocation_recipe, f)

    @classmethod
    def load_allocations(cls, path, dim, backend="torch", device=None, **kwargs):
        import json
        with open(path, "r") as f:
            recipe = json.load(f)
        m = cls(dim=dim, backend=backend, device=device, **kwargs)
        for req in recipe:
            m.add(req["name"], shape=req.get("shape"), region=req.get("region"))
        return m

    def save_bin(self, x, path):
        self.backend.save(x, path)

    def load_bin(self, path):
        return self.backend.load(path)

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value: str):
        if value not in self._packer_map:
            raise ValueError(f"Unknown packing strategy: {value}")
        self._strategy = value
        self.bin_manager._compiled = False

    @property
    def packer(self):
        return self._packer_map[self._strategy]

    @packer.setter
    def packer(self, value: Callable):
        self._packer_map[self._strategy] = value
        self.bin_manager._compiled = False

