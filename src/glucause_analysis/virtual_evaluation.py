"""Analyses for Table 3 and Supplementary Fig. 5."""

import pandas as pd
import matplotlib.pyplot as plt

from .dynamics_evaluation import subject_metrics
from .metrics import (
    MODEL_HORIZONS,
    SUBJECT,
    WINDOW,
    read_frame,
    read_summary,
    check_time_grid,
    require_paired_models,
    subject_summary,
    write_csv,
    require_test_rows,
    require_metrics,
)
from .plot_style import configure, curve, save_figure

CONTEXT = ["intervention", "amount"]


def validate_conditions(frame):
    allowed = {("CHO", 20.0), ("CHO", 40.0), ("INS", 2.0), ("INS", 4.0)}
    if not set(map(tuple, frame[CONTEXT].drop_duplicates().to_numpy())).issubset(
        allowed
    ):
        raise ValueError(
            "Virtual interventions are carbohydrate (CHO) 20/40 g "
            "or bolus insulin (INS) 2/4 U."
        )


def trajectory_summary(frame):
    keys = [*CONTEXT, *SUBJECT, "model", "minute"]
    modeled = (
        frame.groupby(keys, as_index=False)
        .modeled_cgm.mean()
        .rename(columns={"modeled_cgm": "CGM"})
    )
    # Repeated ground truth across models is counted once per window/time point.
    reference = frame.drop_duplicates([*CONTEXT, *WINDOW, "minute"])
    reference = reference.groupby(
        [*CONTEXT, *SUBJECT, "minute"], as_index=False
    ).ground_truth_cgm.mean()
    reference = reference.rename(columns={"ground_truth_cgm": "CGM"}).assign(
        model="Ground truth"
    )
    return subject_summary(
        pd.concat([modeled, reference]), [*CONTEXT, "model", "minute"], ["CGM"]
    )


def render(curves, output):
    configure()
    for (intervention, amount), group in curves.groupby(CONTEXT):
        fig, ax = plt.subplots(figsize=(7, 4))
        for model, values in group.groupby("model", sort=False):
            curve(
                ax,
                values,
                label=model,
                linestyle="--" if model == "Ground truth" else "-",
            )
        unit = "g" if intervention == "CHO" else "U"
        ax.set(
            xlabel="Time (min)",
            ylabel="CGM (mg/dL)",
            title=f"{intervention}: {amount:g} {unit}",
        )
        ax.legend(fontsize=8)
        save_figure(fig, output, f"supp_fig5_{intervention}_{amount:g}")


def run(args):
    curves = None
    if args.input:
        required = [
            *CONTEXT,
            *WINDOW,
            "model",
            "minute",
            "ground_truth_cgm",
            "modeled_cgm",
        ]
        frame = read_frame(
            args.input,
            required,
            ["amount", "minute", "ground_truth_cgm", "modeled_cgm"],
        )
        require_test_rows(frame)
        validate_conditions(frame)
        check_time_grid(
            frame, [*CONTEXT, *WINDOW, "model"], end_by_model=MODEL_HORIZONS, start=0
        )
        require_paired_models(frame, CONTEXT)
        initial = frame.loc[frame.minute == 0]
        if not (abs(initial.modeled_cgm - initial.ground_truth_cgm) <= 1e-6).all():
            raise ValueError(
                "Virtual modeled/reference CGM must agree at the observed present."
            )
        subject = subject_metrics(
            frame.loc[frame.minute > 0], CONTEXT, uncertainty=False
        )
        summary = subject_summary(
            subject, [*CONTEXT, "model", "horizon"], ["MAE", "RMSE"]
        )
        curves = trajectory_summary(frame)
        write_csv(subject, args.output, "virtual_subject_metrics")
    else:
        summary = read_summary(
            args.summary, [*CONTEXT, "model", "horizon"], {"MAE", "RMSE"}
        )
        validate_conditions(summary)
        if args.curves:
            curves = read_summary(args.curves, [*CONTEXT, "model", "minute"], {"CGM"})
            validate_conditions(curves)
    for row in summary.itertuples():
        if row.model not in MODEL_HORIZONS or row.horizon not in range(
            30, MODEL_HORIZONS[row.model] + 1, 30
        ):
            raise ValueError("Invalid model or horizon in virtual summary.")
    require_metrics(summary, [*CONTEXT, "model", "horizon"], {"MAE", "RMSE"})
    if curves is not None:
        for (intervention, amount), group in curves.groupby(CONTEXT):
            models = set(group.model) - {"Ground truth"}
            if (
                not models
                or not models.issubset(MODEL_HORIZONS)
                or "Ground truth" not in set(group.model)
            ):
                raise ValueError(
                    "Virtual curve summaries require reported models and Ground truth."
                )
            limits = {
                **MODEL_HORIZONS,
                "Ground truth": max(MODEL_HORIZONS[m] for m in models),
            }
            check_time_grid(group, ["model"], end_by_model=limits, start=0)
            if group.n_subjects.nunique() != 1:
                raise ValueError(
                    "Paired virtual curves require constant subject counts."
                )
            condition = summary.loc[
                (summary.intervention == intervention) & (summary.amount == amount)
            ]
            for model in sorted(models):
                metrics = condition.loc[condition.model == model]
                if metrics.empty:
                    raise ValueError(
                        f"Virtual curves for {intervention} {amount:g}, {model} "
                        "require matching condition/model metrics."
                    )
                if not metrics.n_subjects.eq(group.n_subjects.iloc[0]).all():
                    raise ValueError(
                        f"Virtual metrics and curves for {intervention} {amount:g}, "
                        f"{model} must have the same subject count."
                    )
    write_csv(summary, args.output, "table3_virtual_interventions")
    if curves is not None:
        write_csv(curves, args.output, "supp_fig5_trajectory_summary")
        render(curves, args.output)
