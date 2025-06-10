from typing import Union, Tuple, Optional, Callable, Dict, Any

class BinManager:
    def __init__(self, dim: int):
        self.requests: Dict[str, Tuple[int, ...]] = {}
        self.slices: Dict[str, Tuple[slice, ...]] = {}
        self.bin_shape: Optional[Tuple[int, ...]] = None
        self._compiled = False
        self.dim = dim


    def _as_shape(self, v) -> Tuple[int, ...]:

        if self.dim > 1:
            raise NotImplemented("Currently only dim=1 shapes (need a 2D+ packer). Otherwise, specify an explicit region.")

        if isinstance(v, int):
            return (v,) * self.dim
        elif isinstance(v, (list, tuple)):
            if all(isinstance(x, int) for x in v):
                if len(v) == 1 and self.dim > 1:
                    # (10,) for dim=2 -> (10,10)
                    return (v[0],) * self.dim
                elif len(v) != self.dim:
                    raise ValueError(f"Shape must have {self.dim} elements (got {v})")
                return tuple(v)
        raise TypeError(f"Could not interpret shape from {v}")

    def _as_region(self, v) -> Tuple[slice, ...]:
        # Accept slice, tuple of slices, or tuple of (start, stop) pairs
        if isinstance(v, slice):
            return (v,) * self.dim
        elif isinstance(v, (list, tuple)):
            # Tuple of slices?
            if all(isinstance(x, slice) for x in v):
                if len(v) != self.dim:
                    raise ValueError(f"Region must have {self.dim} slices (got {v})")
                return tuple(v)
            # Tuple of (start, stop) pairs for each dim
            if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in v):
                return tuple(slice(start, stop) for start, stop in v)
            # Single (start, stop) for 1D
            if self.dim == 1 and len(v) == 2 and all(isinstance(x, int) for x in v):
                return (slice(v[0], v[1]),)
        raise TypeError(f"Could not interpret region from {v}")

    def add(self, name: str, shape: Any = None, region: Any = None):
        if region is not None:
            region_tuple = self._as_region(region)
            self.slices[name] = region_tuple
            self.requests.pop(name, None)
        elif shape is not None:
            shape_tuple = self._as_shape(shape)
            self.requests[name] = shape_tuple
            self.slices.pop(name, None)
        else:
            raise ValueError("Either shape or region must be specified")
        self._compiled = False

    def __setattr__(self, name, value):
        if name in {"requests", "slices", "bin_shape", "_compiled", "dim"}:
            super().__setattr__(name, value)
        # Try explicit region first
        elif isinstance(value, slice) or (
            isinstance(value, (tuple, list)) and (
                all(isinstance(x, slice) for x in value) or
                all(isinstance(x, (tuple, list)) and len(x) == 2 for x in value) or
                (self.dim == 1 and len(value) == 2 and all(isinstance(x, int) for x in value))
            )
        ):
            self.add(name, region=value)
        # Otherwise treat as shape
        elif isinstance(value, (int, tuple, list)):
            self.add(name, shape=value)
        else:
            raise TypeError(f"Unsupported assignment: {name} = {value!r}")

    def __getitem__(self, name):
        if not self._compiled:
            raise RuntimeError("Call .compile(packer) first!")
        return self.slices[name]

    def __getattr__(self, name):
        return self.__getitem__(name)

    def compile(self, packer: Callable):
        allocs, bin_shape = packer(self.requests, self.slices)
        self.slices.update(allocs)
        self.bin_shape = bin_shape
        self._compiled = True

    @property
    def shape(self):
        return self.bin_shape

if __name__ == "__main__":
    sm = BinManager(dim=1)
    sm.add("FOO", shape=10)

    print("Requests:", sm.requests)
    print("Regions:", sm.slices)

