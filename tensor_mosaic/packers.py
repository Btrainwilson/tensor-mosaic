from typing import Dict, Tuple, Union, Optional, Callable, Any

def greedy_packer(requests: Dict[str, Tuple[int, ...]], static) -> Dict[str, Tuple[slice, ...]]:
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


def greedy_gap_packer(requests, static):
    """
    allocates 1d regions for requested shapes without overlap, filling any available gaps between static allocations.
    """
    # gather all intervals (static) as [start, stop)
    intervals = []
    for slices in static.values():
        s = slices[0]
        intervals.append((s.start, s.stop))

    allocs = {}
    for k, shape in requests.items():
        length = shape[0]
        # intervals must be sorted before checking for gaps
        intervals.sort()
        prev_end = 0
        placed = False
        for start, end in intervals:
            if start - prev_end >= length:
                # place in gap [prev_end, start)
                allocs[k] = (slice(prev_end, prev_end + length),)
                # insert and re-sort on next loop
                intervals.append((prev_end, prev_end + length))
                placed = True
                break
            prev_end = end
        if not placed:
            # place at the end
            max_end = max([end for _, end in intervals], default=0)
            allocs[k] = (slice(max_end, max_end + length),)
            intervals.append((max_end, max_end + length))
            # sort is not needed now; will happen on next iteration
    # the bin shape is up to the last occupied slot
    all_ends = [stop for _, stop in intervals]
    bin_shape = (max(all_ends),)
    return allocs, bin_shape
