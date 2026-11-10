"""Automation-against-agreement figure (deterministic PNG)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

FIGURE = C.RESULTS / "automation_curve.png"
STYLE = {"tfidf_lr": ("#7a7a7a", "TF-IDF + LR"), "v1": ("#c26b2b", "judge v1"),
         "v2": ("#2b6cb0", "judge v2"), "guarded": ("#2f855a", "guarded pipeline")}


def automation_figure(curves: dict, target: float, path=FIGURE) -> str:
    """curves: {split: {system: [(coverage, agreement), ...]}}"""
    splits = [s for s in ("dev", "test") if curves.get(s)]
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.2), sharey=True, squeeze=False)
    for ax, split in zip(axes[0], splits):
        for name, pts in curves[split].items():
            color, label = STYLE[name]
            ax.step([p[0] for p in pts], [p[1] for p in pts], where="post", color=color,
                    label=label, linewidth=1.6)
        ax.axhline(target, color="#444444", linestyle=":", linewidth=1)
        ax.set_title(f"{split.upper()}: agreement on automated cases")
        ax.set_xlabel("share of complaints decided automatically")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.5, 1.0)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("agreement with recorded outcome")
    axes[0][-1].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, metadata={"Software": None})
    plt.close(fig)
    return path.relative_to(C.ROOT).as_posix()
