"""Shared plotting conventions."""

import re
from pathlib import Path
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def configure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": plt.cycler(color=COLORS),
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def filename(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")


def save_figure(fig, output, name):
    fig.savefig(Path(output) / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(Path(output) / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def curve(ax, frame, x="minute", label=None, **kwargs):
    frame = frame.sort_values(x)
    xx, yy = frame[x].to_numpy(float), frame["mean"].to_numpy(float)
    (line,) = ax.plot(xx, yy, label=label, **kwargs)
    if "std" in frame:
        sd = frame["std"].to_numpy(float)
        ax.fill_between(xx, yy - sd, yy + sd, color=line.get_color(), alpha=0.15)
    return line


def metric_lines(
    summary, output, prefix, x="horizon", label="model", ylabel=None, bars=False
):
    configure()
    for metric, group in summary.groupby("metric", sort=False):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        horizons = sorted(group[x].unique())
        models = list(group[label].unique())
        width = 0.8 / len(models)
        for index, (name, values) in enumerate(group.groupby(label, sort=False)):
            values = values.sort_values(x)
            position = values[x].to_numpy(float)
            if bars:
                position = (
                    np.array([horizons.index(h) for h in position])
                    - 0.4
                    + width * (index + 0.5)
                )
                ax.bar(position, values["mean"], width=width, label=str(name))
            else:
                ax.plot(position, values["mean"], marker="o", label=str(name))
            valid = values["std"].notna().to_numpy()
            ax.errorbar(
                position[valid],
                values["mean"].to_numpy()[valid],
                yerr=values["std"].to_numpy()[valid],
                fmt="none",
                color="black",
                capsize=3,
            )
        if bars:
            ax.set_xticks(np.arange(len(horizons)), horizons)
        else:
            ax.set_xticks(horizons)
        ax.set(
            xlabel=(
                "Time (min)"
                if x in ("horizon", "minute")
                else f"{x.replace('_', ' ').capitalize()} (min)"
            ),
            ylabel=ylabel or (
                f"{metric} (mg/dL)" if metric in ("MAE", "RMSE") else str(metric)
            ),
        )
        if metric == "PICP":
            ax.axhline(
                0.80,
                color="black",
                linestyle="--",
                linewidth=1,
                label="Nominal coverage (0.80)",
            )
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
        save_figure(fig, output, f"{prefix}_{filename(metric)}")
