import torch
import numpy as np

x_torch = torch.zeros(16)
x_torch[(slice(2, 5),)] = 7
print(x_torch)  # works!

x_torch2 = torch.zeros((2, 16))
x_torch2[:, slice(2, 5)] = 5
print(x_torch2)  # works!

x_np = np.zeros(16)
x_np[(slice(2, 5),)] = 7
print(x_np)  # works!

x_np2 = np.zeros((2, 16))
x_np2[:, slice(2, 5)] = 5
print(x_np2)  # works!
from tensor_mosaic import Mosaic

# Suppose I just want to specify the SIZES of spaces.
m = Mosaic(dim=1)

m.STATE  = 12
m.REWARD = 14
m.ACTION = 14

print(m.shape)

x = m.bin_tensor(1)
print(m.STATE)

x[m.STATE] = 1.
x[m.REWARD] = 2.
x[m.ACTION] = 3.

print(x)

