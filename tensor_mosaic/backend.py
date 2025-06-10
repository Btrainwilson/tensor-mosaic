
# --- Abstract Backends ---

class Backend:
    """Abstracts tensor ops for torch, numpy, jax."""
    def __init__(self, device=None):
        self.device = device
    def arange(self, start, stop):
        raise NotImplementedError
    def meshgrid(self, arrays):
        raise NotImplementedError
    def stack(self, arrays, axis=-1):
        raise NotImplementedError
    def asarray(self, x):
        raise NotImplementedError
    def full(self, shape, fill_value=0, dtype=None):
        raise NotImplementedError
    def move(self, x, device):
        raise NotImplementedError
    def save(self, x, path):
        raise NotImplementedError
    def load(self, path, map_location=None):
        raise NotImplementedError

# --- PyTorch Backend ---

class TorchBackend(Backend):
    def __init__(self, device="cpu"):
        import torch
        super().__init__(device)
        self.torch = torch
    def arange(self, start, stop):
        return self.torch.arange(start, stop, device=self.device)
    def meshgrid(self, arrays):
        return self.torch.meshgrid(*arrays, indexing="ij")
    def stack(self, arrays, axis=-1):
        return self.torch.stack(arrays, dim=axis)
    def asarray(self, x):
        return self.torch.as_tensor(x, device=self.device)
    def full(self, shape, fill_value=0, dtype=None):
        dtype = dtype or self.torch.float
        return self.torch.full(shape, fill_value, dtype=dtype, device=self.device)
    def move(self, x, device):
        return x.to(device)
    def save(self, x, path):
        self.torch.save(x, path)
    def load(self, path, map_location=None):
        return self.torch.load(path, map_location=map_location or self.device)

# --- NumPy Backend ---

class NumpyBackend(Backend):
    def __init__(self):
        import numpy as np
        super().__init__(None)
        self.np = np
    def arange(self, start, stop):
        return self.np.arange(start, stop)
    def meshgrid(self, arrays):
        return self.np.meshgrid(*arrays, indexing="ij")
    def stack(self, arrays, axis=-1):
        return self.np.stack(arrays, axis=axis)
    def asarray(self, x):
        return self.np.asarray(x)
    def full(self, shape, fill_value=0, dtype=None):
        dtype = dtype or self.np.float32
        return self.np.full(shape, fill_value, dtype=dtype)
    def move(self, x, device):
        return x  # no-op for numpy
    def save(self, x, path):
        self.np.save(path, x)
    def load(self, path, map_location=None):
        return self.np.load(path + ".npy")

# --- JAX Backend ---

class JaxBackend(Backend):
    def __init__(self, device=None):
        import jax
        import jax.numpy as jnp
        super().__init__(device)
        self.jnp = jnp
    def arange(self, start, stop):
        return self.jnp.arange(start, stop)
    def meshgrid(self, arrays):
        return self.jnp.meshgrid(*arrays, indexing="ij")
    def stack(self, arrays, axis=-1):
        return self.jnp.stack(arrays, axis=axis)
    def asarray(self, x):
        return self.jnp.asarray(x)
    def full(self, shape, fill_value=0, dtype=None):
        dtype = dtype or self.jnp.float32
        return self.jnp.full(shape, fill_value, dtype=dtype)
    def move(self, x, device):
        return x  # usually not needed; handled by jax
    def save(self, x, path):
        import numpy as np
        np.save(path, np.array(x))
    def load(self, path, map_location=None):
        import numpy as np
        return self.jnp.asarray(np.load(path + ".npy"))

