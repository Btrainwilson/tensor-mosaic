import torch
from tensor_mosaic import SliceManager

sm = SliceManager(dim=1)

sm.FOO = 10                 # Dynamic allocation, shape=(10,)
sm.BAR = (10, 20)           # Explicit region
sm.BAZ = (20, 30)           # Explicit region (tuple)
sm.QUX = 5                  # Dynamic allocation, shape=(5,)
sm.SLICE = slice(4, 10, 2)  # Slice Handling

def greedy_gap_packer(requests, static):
    """
    Allocates 1D regions for requested shapes without overlap, filling any available gaps between static allocations.
    """
    # Gather all intervals (static) as [start, stop)
    intervals = []
    for slices in static.values():
        s = slices[0]
        intervals.append((s.start, s.stop))
    intervals.sort()

    allocs = {}
    for k, shape in requests.items():
        length = shape[0]
        # Intervals must be sorted before checking for gaps
        intervals.sort()
        prev_end = 0
        placed = False
        for start, end in intervals:
            if start - prev_end >= length:
                # Place in gap [prev_end, start)
                allocs[k] = (slice(prev_end, prev_end + length),)
                # Insert and re-sort on next loop
                intervals.append((prev_end, prev_end + length))
                placed = True
                break
            prev_end = end
        if not placed:
            # Place at the end
            max_end = max([end for _, end in intervals], default=0)
            allocs[k] = (slice(max_end, max_end + length),)
            intervals.append((max_end, max_end + length))
            # sort is not needed now; will happen on next iteration
    # The bin shape is up to the last occupied slot
    all_ends = [stop for _, stop in intervals]
    bin_shape = (max(all_ends),)
    return allocs, bin_shape

sm.compile(greedy_gap_packer)

print(sm.slices)
print(sm.BAR)
print(sm.BAZ)
print(sm.QUX)
print(sm.FOO)

print("Shape:", sm.shape)
