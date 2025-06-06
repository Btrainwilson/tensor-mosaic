import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from tensor_mosaic import Mosaic
from tensor_mosaic.plot import plot_mosaic

# Example usage after allocation:
if __name__ == "__main__":
    # Assume Mosaic class and allocation has already been done...
    mosaic = Mosaic(device='cpu')
    mosaic.A = 10
    mosaic.B = [5]
    mosaic.C = (8,)
    plot_mosaic(mosaic)

    # For 2D:
    mosaic2 = Mosaic(device='cpu')
    mosaic2.A = (2, 3)
    mosaic2.B = (3, 4)
    mosaic2.C = (4, 2)
    plot_mosaic(mosaic2)


    mosaic = Mosaic()
    mosaic2.A = (2, 3)
    mosaic2.B = (3, 4)
    mosaic2.C = (4, 2)
    mosaic.pretty_print()
    # Change strategy (after registering more packers)
    # mosaic.strategy = "rectpack"
