"""Analyses for Fig. 4, Table 2, and Supplementary Fig. 4."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import (
    MODEL_HORIZONS,
    SUBJECT,
    WINDOW,
    read_frame,
    read_summary,
    dynamics_metrics,
    clarke_zones,
    check_time_grid,
    require_paired_models,
    subject_summary,
    write_csv,
    require_test_rows,
    require_metrics,
)
from .plot_style import configure, filename, metric_lines, save_figure


def load_results(path, paired=False):
    required = [*WINDOW, "model", "minute", "ground_truth_cgm", "modeled_cgm"]
    frame = read_frame(path, required, ["minute", "ground_truth_cgm", "modeled_cgm"])
    require_test_rows(frame)
    check_time_grid(frame, [*WINDOW, "model"], end_by_model=MODEL_HORIZONS)
    require_paired_models(frame, identical_windows=paired)
    if ("lower_cgm" in frame) != ("upper_cgm" in frame):
        raise ValueError("Supply both lower_cgm and upper_cgm.")
    if "lower_cgm" in frame:
        for col in ("lower_cgm", "upper_cgm"):
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        uncertainty = frame.loc[frame.model == "GluCause"]
        if not uncertainty.empty:
            dynamics_metrics(
                uncertainty.ground_truth_cgm,
                uncertainty.modeled_cgm,
                uncertainty.lower_cgm,
                uncertainty.upper_cgm,
            )
    return frame


def subject_metrics(frame, context=(), uncertainty=True):
    """Compute each subject's errors over all points through each horizon."""
    rows = []
    keys = [*context, *SUBJECT, "model"]
    for key, group in frame.groupby(keys, sort=True):
        info = dict(zip(keys, key))
        for horizon in range(30, MODEL_HORIZONS[info["model"]] + 1, 30):
            part = group.loc[group.minute <= horizon]
            bounds = (None, None)
            if uncertainty and info["model"] == "GluCause" and "lower_cgm" in part:
                bounds = (part.lower_cgm, part.upper_cgm)
            values = dynamics_metrics(part.ground_truth_cgm, part.modeled_cgm, *bounds)
            values.pop("MBE")
            rows.append({**info, "horizon": horizon, **values})
    return pd.DataFrame(rows)


def clarke_table(frame):
    rows = []
    for model, group in frame.groupby("model", sort=False):
        if model == "GluCause (Non-personalized)":
            continue
        for horizon in range(30, MODEL_HORIZONS[model] + 1, 30):
            part = group.loc[group.minute == horizon]
            zones = clarke_zones(part.ground_truth_cgm, part.modeled_cgm)
            row = {"model": model, "minute": horizon, "n_pairs": len(zones)}
            row.update({z: float(100 * (zones == z).mean()) for z in "ABCDE"})
            row["A+B"] = row["A"] + row["B"]
            rows.append(row)
    return pd.DataFrame(rows, columns=["model", "minute", "n_pairs", *"ABCDE", "A+B"])


def plot_clarke(frame, output):
    configure()
    colors = dict(zip("ABCDE", ["#009E73", "#0072B2", "#E69F00", "#D55E00", "#CC79A7"]))
    grid = np.linspace(0, 400, 801)
    xx, yy = np.meshgrid(grid, grid)
    background = clarke_zones(xx, yy)
    for model, group in frame.groupby("model", sort=False):
        horizons = (
            range(30, 121, 30) if model == "GluCause" else [MODEL_HORIZONS[model]]
        )
        if model == "GluCause (Non-personalized)":
            continue
        for horizon in horizons:
            part = group.loc[group.minute == horizon]
            zone = clarke_zones(part.ground_truth_cgm, part.modeled_cgm)
            fig, ax = plt.subplots(figsize=(5, 5))
            for z in "ABCDE":
                ax.contour(
                    xx,
                    yy,
                    (background == z).astype(float),
                    levels=[0.5],
                    colors="0.5",
                    linewidths=0.6,
                )
                keep = zone == z
                ax.scatter(
                    part.loc[keep, "ground_truth_cgm"],
                    part.loc[keep, "modeled_cgm"],
                    s=4,
                    alpha=0.35,
                    color=colors[z],
                    label=f"Zone {z}",
                    rasterized=True,
                )
            ax.plot([0, 400], [0, 400], "k--", linewidth=0.7)
            ax.set(
                xlim=(0, 400),
                ylim=(0, 400),
                xlabel="Reference CGM (mg/dL)",
                ylabel="Modeled CGM (mg/dL)",
                title=f"{model}, {horizon} min",
            )
            ax.legend(markerscale=2, fontsize=8)
            prefix = "fig4c" if model == "GluCause" else "supp_fig4"
            save_figure(fig, output, f"{prefix}_{filename(model)}_{horizon}min")


def render(summary, output):
    if not set(summary.model).issubset(MODEL_HORIZONS):
        raise ValueError("Unknown model in dynamics summary.")
    for row in summary.itertuples():
        if row.horizon not in range(30, MODEL_HORIZONS[row.model] + 1, 30):
            raise ValueError("Invalid model horizon in summary.")
        if row.metric in ("PICP", "MPIW") and row.model != "GluCause":
            raise ValueError("Fig. 4b contains GluCause uncertainty metrics only.")
    errors = summary.loc[summary.metric.isin(["MAE", "RMSE"])]
    intervals = summary.loc[summary.metric.isin(["PICP", "MPIW"])]
    require_metrics(errors, ["model", "horizon"], {"MAE", "RMSE"})
    require_metrics(intervals, ["model", "horizon"], {"PICP", "MPIW"})
    metric_lines(
        summary.loc[summary.metric.isin(["MAE", "RMSE"])],
        output,
        "fig4a",
        bars=True,
    )
    for metric, label in (("PICP", "PICP"), ("MPIW", "MPIW (mg/dL)")):
        metric_lines(
            summary.loc[summary.metric == metric], output, "fig4b", ylabel=label
        )


def run(args):
    if args.input:
        frame = load_results(args.input)
        subject = subject_metrics(frame)
        summary = subject_summary(
            subject,
            ["model", "horizon"],
            [c for c in ("MAE", "RMSE", "PICP", "MPIW") if c in subject],
        )
        summary = summary.loc[summary.n_subjects > 0].reset_index(drop=True)
        write_csv(subject, args.output, "dynamics_subject_metrics")
        write_csv(clarke_table(frame), args.output, "table2_clarke_zones")
        plot_clarke(frame, args.output)
    else:
        summary = read_summary(
            args.summary, ["model", "horizon"], {"MAE", "RMSE", "PICP", "MPIW"}
        )
        if args.clarke_summary:
            columns = ["model", "minute", "n_pairs", *"ABCDE", "A+B"]
            zones = read_frame(
                args.clarke_summary, columns, columns[1:], ["model", "minute"]
            )
            require_test_rows(zones)
            if not np.allclose(
                zones[list("ABCDE")].sum(axis=1), 100, rtol=0, atol=0.01
            ):
                raise ValueError(
                    "Clarke Error Grid zone percentages must sum to 100 (allowing rounding)."
                )
            if not np.allclose(zones["A+B"], zones.A + zones.B, rtol=0, atol=0.01):
                raise ValueError("Inconsistent A+B percentage.")
            if ((zones[list("ABCDE")] < 0) | (zones[list("ABCDE")] > 100)).any().any():
                raise ValueError("Clarke Error Grid zone percentages must be in [0,100].")
            for row in zones.itertuples():
                if row.model == "GluCause (Non-personalized)":
                    raise ValueError(
                        "Table 2 does not report non-personalized Clarke Error Grid zones."
                    )
                if row.model not in MODEL_HORIZONS or row.minute not in range(
                    30, MODEL_HORIZONS[row.model] + 1, 30
                ):
                    raise ValueError("Invalid model or endpoint in Clarke Error Grid summary.")
                if row.n_pairs < 1 or row.n_pairs % 1:
                    raise ValueError("n_pairs must be a positive integer.")
            write_csv(zones, args.output, "table2_clarke_zones")
    write_csv(summary, args.output, "fig4_metric_summary")
    render(summary, args.output)
