"""Render a block diagram of the CIFAR-10 CNN to results/model_architecture.png.

Uses only matplotlib (no graphviz/pydot dependency) so it runs anywhere the
training code does. Layer output shapes are read from the actual built model so
the diagram never drifts from model.py.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model

# Group the model into logical stages for a readable diagram.
STAGES = [
    ("Input", "32 x 32 x 3", "#cfd8dc"),
    ("Conv Block 1\n2x [Conv2D 32 + BN] -> MaxPool -> Dropout 0.2", "16 x 16 x 32", "#90caf9"),
    ("Conv Block 2\n2x [Conv2D 64 + BN] -> MaxPool -> Dropout 0.3", "8 x 8 x 64", "#64b5f6"),
    ("Conv Block 3\n2x [Conv2D 128 + BN] -> MaxPool -> Dropout 0.4", "4 x 4 x 128", "#42a5f5"),
    ("GlobalAveragePooling2D", "128", "#a5d6a7"),
    ("Dense 128 + BN + Dropout 0.5", "128", "#81c784"),
    ("Dense 10 (softmax)", "10", "#ffcc80"),
]


def main():
    model = build_model()
    total_params = model.count_params()

    fig, ax = plt.subplots(figsize=(7, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(STAGES) * 1.6 + 1)
    ax.axis("off")

    box_w, box_h = 6.2, 1.0
    x_center = 5.0
    y = len(STAGES) * 1.6

    centers = []
    for title, shape, color in STAGES:
        box = FancyBboxPatch(
            (x_center - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.4, edgecolor="#37474f", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x_center, y, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#102027")
        ax.text(x_center + box_w / 2 + 0.25, y, shape, ha="left", va="center",
                fontsize=9, style="italic", color="#455a64")
        centers.append(y)
        y -= 1.6

    for top, bottom in zip(centers[:-1], centers[1:]):
        arrow = FancyArrowPatch(
            (x_center, top - box_h / 2), (x_center, bottom + box_h / 2),
            arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color="#37474f",
        )
        ax.add_patch(arrow)

    ax.set_title(
        f"CIFAR-10 CNN Architecture\n{total_params:,} trainable parameters",
        fontsize=13, fontweight="bold", pad=16,
    )

    os.makedirs("results", exist_ok=True)
    out = "results/model_architecture.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
