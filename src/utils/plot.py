import numpy as np
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def plot_scores(scores, window=10, name="score_curve.png"):
    if len(scores) == 0:
        return
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    steps = np.concatenate([np.full(len(b), i) for i, b in enumerate(scores)])
    vals = np.concatenate(scores)
    step_means = np.array([np.mean(b) for b in scores])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    y_limit = np.percentile(vals, 99)
    base_rgb = mcolors.to_rgb("#3498db")
    colors = np.zeros((len(vals), 4))
    colors[:, :3] = base_rgb
    colors[:, 3] = np.clip(0.1 + 0.5 * (vals / (y_limit + 1e-6)), 0.05, 0.4)

    ax.scatter(steps, vals, c=colors, s=5, edgecolors="none", label="Sample Score")

    if len(step_means) >= window:
        kernel = np.ones(window) / window
        smooth_y = np.convolve(step_means, kernel, mode="valid")
        smooth_x = np.arange(window - 1, len(step_means))
        ax.plot(smooth_x, smooth_y, color="#2c3e50", lw=2, label=f"Trend (MA-{window})")
    ax.set_title("Reward Score Distribution and Trend", fontweight="bold")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Score")
    ax.set_ylim(min(0, vals.min()), y_limit * 1.1)

    legend = ax.legend(loc="upper right", frameon=True, shadow=True)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    plt.tight_layout()
    plt.savefig(name, bbox_inches="tight")
    plt.close()
