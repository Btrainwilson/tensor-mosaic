from tensor_mosaic import Mosaic

def try_import(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False

def test_mosaic_basic():
    print("Testing Mosaic with all backends...")
    backends = []
    if try_import("torch"):
        backends.append("torch")
    if try_import("numpy"):
        backends.append("numpy")
    if try_import("jax"):
        backends.append("jax")
    if not backends:
        print("No backend available, install torch/numpy/jax to run tests.")
        return

    for backend in backends:
        print(f"\n[Backend: {backend}]")
        m = Mosaic(backend=backend)
        m.add("a", (5,))
        m.add("b", (3,))
        m.compile()
        print("Shape:", m.shape)
        print("Slices:", m.slices)
        bin_tensor = m.bin_tensor(fill_value=42)
        print("Bin tensor:", bin_tensor)
        # View checks (for torch: Tensor, numpy: ndarray, jax: DeviceArray)
        view_a = m.slice_view(bin_tensor, "a")
        view_b = m.slice_view(bin_tensor, "b")
        print("View a:", view_a)
        print("View b:", view_b)
        # Should have correct shapes
        assert view_a.shape[0] == 5, f"{backend}: view_a wrong shape"
        assert view_b.shape[0] == 3, f"{backend}: view_b wrong shape"
        # Serialization
        m.save_allocations(f"test_alloc_{backend}.json")
        m2 = Mosaic.load_allocations(f"test_alloc_{backend}.json", backend=backend)
        m2.compile()
        assert m2.slices == m.slices, f"{backend}: loaded slices do not match"
        print("Serialization test passed.")

if __name__ == "__main__":
    test_mosaic_basic()
