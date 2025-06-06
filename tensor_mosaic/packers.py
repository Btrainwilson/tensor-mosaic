from typing import Dict, Tuple, Union, Optional, Callable, Any

def greedy_packer(requests: Dict[str, Tuple[int, ...]]) -> Dict[str, Tuple[slice, ...]]:
    """
    Naive greedy allocation: Place each shape sequentially, growing the bin as needed along the first axis.
    Returns a dict of {alias: tuple of slices} and the bin size.
    """
    allocations = {}
    pos = [0] * (len(next(iter(requests.values()))) if requests else 1)
    max_dims = [0] * len(pos)
    for alias, shape in requests.items():
        slices = []
        for i, dim in enumerate(shape):
            start = pos[i] if i == 0 else 0
            stop = start + dim
            slices.append(slice(start, stop))
            if stop > max_dims[i]:
                max_dims[i] = stop
        allocations[alias] = tuple(slices)
        pos[0] += shape[0]
    return allocations, tuple(max_dims)
