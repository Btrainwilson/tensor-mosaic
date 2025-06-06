# test_mosaic.py

import torch
import pytest

from tensor_mosaic import SpaceCache, Mosaic
from tensor_mosaic.packers import greedy_packer
# from your_module import rectpack_packer, knapsack_packer   # Uncomment if implemented

# ---- SpaceCache Tests ----

def test_spacecache_set_and_get():
    sc = SpaceCache()
    sc.a = [1, 2, 3]
    assert torch.equal(sc.a, torch.tensor([1, 2, 3]))
    sc['b'] = torch.arange(5)
    assert torch.equal(sc['b'], torch.arange(5))

def test_spacecache_device_move():
    sc = SpaceCache()
    sc.vec = [1, 2, 3]
    assert sc.vec.device.type == "cpu"
    if torch.cuda.is_available():
        sc.to("cuda")
        assert sc.vec.device.type == "cuda"

def test_spacecache_del_and_clear():
    sc = SpaceCache()
    sc.x = [0]
    assert 'x' in sc
    del sc['x']
    assert 'x' not in sc
    sc.y = [1]
    sc.z = [2]
    sc.clear()
    assert len(list(sc.keys())) == 0

def test_spacecache_normalize():
    sc = SpaceCache()
    arr = sc.normalize([3, 4])
    assert torch.equal(arr, torch.tensor([3, 4]))

# ---- Mosaic Tests ----

def make_1d_mosaic(**kwargs):
    m = Mosaic(autocompile=False)
    for k, v in kwargs.items():
        setattr(m, k, v)
    m.compile()
    return m

def make_2d_mosaic(**kwargs):
    m = Mosaic(autocompile=False)
    for k, v in kwargs.items():
        setattr(m, k, v)
    m.compile()
    return m

def test_mosaic_basic_1d_allocation():
    m = make_1d_mosaic(a=5, b=3, c=7)
    assert m.bin_shape == (15,)
    assert m.slices["a"] == (slice(0, 5),)
    assert m.slices["b"] == (slice(5, 8),)
    assert m.slices["c"] == (slice(8, 15),)
    t = m.bin_tensor()
    t.fill_(42)
    assert torch.equal(m.slice_view(t, "a"), torch.full((5,), 42))

def test_mosaic_autocompile():
    m = Mosaic(autocompile=True)
    m.foo = 4
    m.bar = 6
    assert m._compiled
    assert m.bin_shape[0] >= 10
    # Remove an alias and check recompile
    m.requests.pop("foo")
    m._compiled = False
    m.compile()
    assert "foo" not in m.slices

def test_mosaic_dimension_check():
    m = Mosaic(autocompile=False)
    m.x = (2, 3)
    with pytest.raises(ValueError):
        m.y = 5  # Should error: dimension mismatch

def test_mosaic_bin_tensor_dtype_and_device():
    m = make_1d_mosaic(a=3)
    t = m.bin_tensor(fill_value=1, dtype=torch.long)
    assert t.dtype == torch.long
    assert t.device == m.device

def test_mosaic_set_strategy_property():
    m = make_1d_mosaic(a=5, b=7)
    old_slices = m.slices.copy()
    m.strategy = "greedy"
    m.compile()
    assert m.slices == old_slices

def test_mosaic_set_custom_packer():
    def reverse_packer(requests):
        # Packs backwards for test
        total = sum(x[0] for x in requests.values())
        allocations = {}
        start = total
        for k, v in requests.items():
            start -= v[0]
            allocations[k] = (slice(start, start + v[0]),)
        return allocations, (total,)
    m = Mosaic(autocompile=False)
    m.a = 2
    m.b = 3
    m.packer = reverse_packer
    m.compile()
    assert m.slices["a"] == (slice(3, 5),)
    assert m.slices["b"] == (slice(0, 3),)

def test_mosaic_pretty_print_and_slice_view(capsys):
    m = make_2d_mosaic(x=(2, 3), y=(3, 2))
    m.pretty_print()
    out = capsys.readouterr().out
    assert "Bin shape" in out
    t = m.bin_tensor(fill_value=9)
    arr = m.slice_view(t, "x")
    assert arr.shape == (2, 3)

# ---- Packers Integration ----

def test_greedy_packer_output():
    reqs = {"a": (3,), "b": (4,), "c": (2,)}
    alloc, shape = greedy_packer(reqs)
    assert shape == (9,)
    assert alloc["a"] == (slice(0, 3),)
    assert alloc["b"] == (slice(3, 7),)
    assert alloc["c"] == (slice(7, 9),)
