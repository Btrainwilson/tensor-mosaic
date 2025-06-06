import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_slices(slices: dict, bin_shape=None, title="Allocation Plot", color_map="tab20"):
    """
    Universal plotting for {name: tuple of slices}.
    Supports 1D and 2D; falls back to sensible default for missing bin_shape.
    """
    # Auto-detect bin shape if not supplied
    if bin_shape is None:
        n_dim = len(next(iter(slices.values())))
        maxes = [0] * n_dim
        for v in slices.values():
            for i, s in enumerate(v):
                maxes[i] = max(maxes[i], s.stop)
        bin_shape = tuple(maxes)
    else:
        n_dim = len(bin_shape)

    cmap = plt.get_cmap(color_map)
    fig, ax = plt.subplots(figsize=(8, 2) if n_dim == 1 else (8, 8))
    ax.set_title(title)

    if n_dim == 1:
        y = 0.5
        for i, (alias, slc) in enumerate(slices.items()):
            color = cmap(i % cmap.N)
            x0 = slc[0].start
            width = slc[0].stop - slc[0].start
            rect = patches.Rectangle((x0, y-0.3), width, 0.6, facecolor=color, edgecolor="black")
            ax.add_patch(rect)
            ax.text(x0 + width/2, y, alias, color='black', ha='center', va='center', fontsize=12)
        ax.set_xlim(0, bin_shape[0])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Index")

    elif n_dim == 2:
        for i, (alias, slc) in enumerate(slices.items()):
            color = cmap(i % cmap.N)
            x0, y0 = slc[1].start, slc[0].start
            width = slc[1].stop - slc[1].start
            height = slc[0].stop - slc[0].start
            rect = patches.Rectangle((x0, y0), width, height, facecolor=color, edgecolor="black", alpha=0.7)
            ax.add_patch(rect)
            ax.text(x0 + width/2, y0 + height/2, alias, color='black', ha='center', va='center', fontsize=12)
        ax.set_xlim(0, bin_shape[1])
        ax.set_ylim(0, bin_shape[0])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.invert_yaxis()
    else:
        raise NotImplementedError("plot_slices only supports 1D and 2D slices at present.")

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    slices = {
        "M": (slice(0, 4), slice(0, 4)),       # Large, top-left
        "O": (slice(2, 3), slice(4, 5)),       # Tall, off-center right
        "S": (slice(3, 4), slice(4, 7)),      # Thin, top-right
        "A": (slice(0, 2), slice(5, 7)),      # Big, bottom-left
        "I": (slice(7, 10), slice(4, 6)),      # Small, bottom center-left
        "C": (slice(7, 10), slice(6, 12)),     # Wide, bottom right
    }
    bin_shape = (10, 10)

# Now plot it:
    plot_slices(slices, bin_shape=bin_shape, title="Irregular MOSAIC", color_map="Spectral")

