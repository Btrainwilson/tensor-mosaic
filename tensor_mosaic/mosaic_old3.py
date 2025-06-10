
class Mosaic:
    def __init__(self, backend="torch", device=None, cache=True, autocompile=True, strategy="greedy", batched=False):

        self.backend_name = backend
        self.backend = self.BACKEND_MAP[backend](device) if backend != "numpy" else self.BACKEND_MAP[backend]()
        self.device = device
        self.requests: Dict[str, Tuple[int, ...]] = {}
        self.slices: Dict[str, Tuple[slice, ...]] = {}
        self.bin_shape: Optional[Tuple[int, ...]] = None
        self._compiled = False
        self._ndim: Optional[int] = None
        self.cache_indices = cache
        self.indices = {}  # SpaceCache-like, just dict for generality
        self._allocation_recipe: List[Dict] = []
        self._packer_map: Dict[str, Callable] = {
            "greedy": self.default_packer,
        }
        self._strategy = strategy
        self.autocompile = autocompile
        self.batched = batched

    def add(self, name: str, shape=None, region=None):
        if region is not None:
            self.slices[name] = tuple(region) if not isinstance(region, tuple) else region
            self.requests.pop(name, None)
            self._allocation_recipe.append(dict(name=name, shape=None, region=[(s.start, s.stop, s.step) if isinstance(s, slice) else s for s in region]))
        elif shape is not None:
            if isinstance(shape, int):
                shape = (shape,)
            elif isinstance(shape, list):
                shape = tuple(shape)
            if self._ndim is None:
                self._ndim = len(shape)
            elif len(shape) != self._ndim:
                raise ValueError(f"All allocations must have dimension {self._ndim}, got {shape}")
            self.requests[name] = shape
            self.slices.pop(name, None)
            self._allocation_recipe.append(dict(name=name, shape=shape, region=None))
        else:
            raise ValueError("Must specify either shape or region")
        self._compiled = False
        if self.autocompile:
            self.compile()

    @property
    def shape(self):
        return self.bin_shape

    def __setattr__(self, name, value):
        if name in {
            "backend", "backend_name", "device", "requests", "slices", "bin_shape", "_compiled", "_ndim",
            "cache_indices", "indices", "_packer_map", "_strategy", "autocompile", "batched", "_allocation_recipe"
        }:
            super().__setattr__(name, value)
        elif isinstance(value, (int, tuple, list)):
            self.add(name, shape=value)
        elif isinstance(value, slice) or (isinstance(value, tuple) and all(isinstance(x, slice) for x in value)):
            self.add(name, region=value)
        else:
            raise TypeError("Allocation must be int, tuple, list, or slice/tuple of slices.")

    def __getitem__(self, name):
        if not self._compiled:
            self.compile()
        if self.cache_indices and name in self.indices:
            return self.indices[name]
        return self.slices[name]

    def __getattr__(self, name):
        return self.__getitem__(name)
