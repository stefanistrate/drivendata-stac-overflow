"""Plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Colormap
from matplotlib.colors import Normalize


def plot_images_sxs(fig: plt.figure.FigureBase,
                    images: list[np.ndarray],
                    cmaps: list[str | Colormap] = None,
                    bad_color: str = "magenta",
                    norms: list[None | Normalize] = None,
                    interpolation: str = "none",
                    titles: list[str] = None) -> None:
    """Plots multiple images side-by-side."""

    num_images = len(images)
    if cmaps is None:
        cmaps = ["gray"] * num_images
    cmaps = [cmap or "gray" for cmap in cmaps]
    cmaps = [
        plt.get_cmap(cmap).with_extremes(bad=bad_color) if isinstance(cmap, str)
        else cmap.with_extremes(bad=bad_color)
        for cmap in cmaps
    ]  # yapf: disable
    if norms is None:
        norms = [None] * num_images

    axs = fig.subplots(1, num_images, squeeze=False)[0]
    for idx in range(num_images):
        axs[idx].imshow(images[idx],
                        cmap=cmaps[idx],
                        norm=norms[idx],
                        interpolation=interpolation)
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
        if titles:
            axs[idx].set_title(titles[idx])
