import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_value_histogram(decoded: np.ndarray, title: str, save_path: str | Path | None = None):
    vals, counts = np.unique(decoded, return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(vals, counts, width=0.15, color='#4C72B0')
    ax.set_title(title)
    ax.set_xlabel('value')
    ax.set_ylabel('count')
    ax.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def plot_matrix_heatmap(matrix: np.ndarray, title: str, downsample: int = 4,
                          save_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 6))
    small = matrix[::downsample, ::max(1, downsample // 2)]
    im = ax.imshow(small, cmap='RdBu_r', vmin=-1.75, vmax=1.75, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('col (input dim)')
    ax.set_ylabel('row (output dim)')
    plt.colorbar(im, ax=ax, label='value')
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig


def plot_comparison(stats_dict: dict, groups: dict[str, list[str]],
                      save_path: str | Path | None = None):
    n_groups = len(groups)
    n_cols = 2
    n_rows = (n_groups + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5.5 * n_rows))
    axes = np.array(axes).flatten()

    for ax, (group_title, names) in zip(axes, groups.items()):
        for name in names:
            r = stats_dict[name]
            vals, counts = np.unique(r.decoded, return_counts=True)
            freq = counts / counts.sum() * 100
            ax.plot(vals, freq, marker='o', label=name, linewidth=2)
        ax.set_title(group_title, fontsize=11)
        ax.set_xlabel('value')
        ax.set_ylabel('% of values')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[len(groups):]:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig
