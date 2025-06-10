import os
import numpy as np
import pytest

# Update the import path as needed:
from tensor_mosaic import Mosaic  # <-- Change to match your codebase

@pytest.fixture(params=["numpy"])
def mosaic(request):
    """Fixture for a Mosaic instance on the desired backend."""
    return Mosaic(backend=request.param)

def test_dynamic_and_explicit_alloc(mosaic):
    # Dynamic
    mosaic.FOO = (10,)
    mosaic.QUX = (5,)
    # Explicit
    mosaic.BAR = slice(10, 20)
    mosaic.BAZ = (slice(25, 30),)
    mosaic.compile()
    # Slices exist
    assert isinstance(mosaic.FOO, tuple) and isinstance(mosaic.BAR, tuple)
    assert mosaic.shape[0] >= 30
    # No overlaps
    used = np.zeros(mosaic.shape, dtype=bool)
    for name in ["FOO", "BAR", "BAZ", "QUX"]:
        s = mosaic[name][0]
        assert not used[s.start:s.stop].any(), f"Overlap in {name}"
        used[s.start:s.stop] = True

def test_bin_tensor_and_slice_view(mosaic):
    mosaic.FOO = (4,)
    mosaic.BAR = slice(4, 8)
    mosaic.compile()
    arr = mosaic.bin_tensor(fill_value=0)
    arr[mosaic.FOO] = 1
    arr[mosaic.BAR] = 2
    # slice_view returns correct subarrays
    foo_view = mosaic.slice_view(arr, "FOO")
    bar_view = mosaic.slice_view(arr, "BAR")
    assert np.all(foo_view == 1)
    assert np.all(bar_view == 2)

def test_attribute_access(mosaic):
    mosaic.alpha = (3,)
    mosaic.beta = slice(3, 6)
    mosaic.compile()
    # Attribute-style access
    s1 = mosaic.alpha
    s2 = mosaic.beta
    assert isinstance(s1, tuple) and isinstance(s2, tuple)

def test_serialization_and_reload(tmp_path, mosaic):
    mosaic.X = (2,)
    mosaic.Y = slice(2, 5)
    mosaic.compile()
    p = tmp_path / "alloc.json"
    mosaic.save_allocations(str(p))
    m2 = Mosaic.load_allocations(str(p), backend=mosaic.backend_name)
    print(m2._allocation_recipe)
    m2.compile()
    assert m2.slice_manager.slices == mosaic.slice_manager.slices

def test_indices_cache(mosaic):
    # Should populate .indices dict after compile
    mosaic.FOO = (3,)
    mosaic.BAR = slice(3, 6)
    mosaic.compile()
    assert "FOO" in mosaic.indices and "BAR" in mosaic.indices
    # Indices should cover all values in the range
    foo_indices = mosaic.indices["FOO"]
    bar_indices = mosaic.indices["BAR"]
    # For 1D, indices are just positions
    foo_set = set(foo_indices.flatten())
    bar_set = set(bar_indices.flatten())
    assert foo_set == set(range(*mosaic.FOO[0].indices(mosaic.shape[0])))
    assert bar_set == set(range(*mosaic.BAR[0].indices(mosaic.shape[0])))

def test_bin_save_and_load(tmp_path, mosaic):
    mosaic.FOO = (5,)
    mosaic.compile()
    arr = mosaic.bin_tensor(fill_value=9)
    path = tmp_path / "bin"
    mosaic.save_bin(arr, str(path))
    arr2 = mosaic.load_bin(str(path))
    # Should be equal (for numpy)
    assert np.all(arr == arr2)

def test_pretty_print(capsys, mosaic):
    mosaic.a = (2,)
    mosaic.b = slice(2, 4)
    mosaic.compile()
    mosaic.pretty_print()
    out = capsys.readouterr().out
    assert "Mosaic Allocations" in out
    assert "Bin shape" in out

# Optionally test .strategy setter/getter
def test_strategy_setter(mosaic):
    old = mosaic.strategy
    mosaic.strategy = "greedy"
    assert mosaic.strategy == "greedy"
    mosaic.strategy = old

# Optionally test .packer setter/getter
def test_packer_setter(mosaic):
    def dummy_packer(req, static): return ({}, (0,))
    mosaic.packer = dummy_packer
    assert mosaic.packer is dummy_packer

