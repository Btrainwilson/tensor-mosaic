from tensor_mosaic import Mosaic
import torch

lspace = Mosaic(cache=True, device='cpu')

lspace.DIST = (2, )
lspace.AUX = [3, ]

lspace.pretty_print()
# Build the bin tensor
x = lspace.bin_tensor()

print("DIST view:", x[lspace.DIST])

# Example: plugging a custom packer
def my_packer(requests):
    # Must match the packer interface!
    # Just allocates everything at (0,0,...), for demo only!
    allocs = {alias: tuple(slice(0, s) for s in shape) for alias, shape in requests.items()}
    ndims = max([len(shape) for alias, shape in requests.items()])
    return allocs, ndims

lspace.compile(packer=my_packer)
lspace.pretty_print()
