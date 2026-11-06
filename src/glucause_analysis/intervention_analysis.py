"""Intervention analyses for Fig. 2–3 and Supplementary Fig. 1–2/Table 1–2."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
)
from .plot_style import configure, curve, filename, save_figure

PRESENT = {f"CHO_{d}g_at_0min" for d in (30, 60, 90)} | {
    f"INS_{d}U_at_0min" for d in (3, 6, 9)
}
FUTURE = (
    {f"CHO_50g_at_{t}min" for t in (20, 40, 60)}
    | {f"INS_5U_at_{t}min" for t in (20, 40, 60)}
    | {"CHO_50g_at_10min_INS_5U_at_50min", "INS_5U_at_10min_CHO_50g_at_50min"}
)
ABLATION = {f"CHO_{d}g_at_30min" for d in (30, 60, 90)} | {
    f"INS_{d}U_at_30min" for d in (3, 6, 9)
}
SCENARIOS = {"present": PRESENT, "future": FUTURE, "ablation": ABLATION}
MINIMUM_AMOUNT_COLUMNS = {
    "RMSE": "amount_min_mean_RMSE",
    "MBE": "amount_min_abs_mean_MBE",
}
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "ground_truth": "Ground truth",
    **{
        f"CHO_{amount}g_at_{minute}min": f"+{amount} g"
        for minute in (0, 30)
        for amount in (30, 60, 90)
    },
    **{
        f"INS_{amount}U_at_{minute}min": f"+{amount} U"
        for minute in (0, 30)
        for amount in (3, 6, 9)
    },
    **{
        f"{intervention}_{amount}_at_{minute}min": f"+{minute} min"
        for intervention, amount in (("CHO", "50g"), ("INS", "5U"))
        for minute in (20, 40, 60)
    },
    "CHO_50g_at_10min_INS_5U_at_50min": "CHO → INS",
    "INS_5U_at_10min_CHO_50g_at_50min": "INS → CHO",
}


def load_trajectories(path, kind, models):
    required = [
        *WINDOW,
        "model",
        "scenario",
        "minute",
        "ground_truth_cgm",
        "modeled_cgm",
    ]
    frame = read_frame(path, required, ["minute", "ground_truth_cgm", "modeled_cgm"])
    require_test_rows(frame)
    allowed = {"baseline"} | SCENARIOS[kind]
    if not set(frame.scenario).issubset(allowed):
        raise ValueError(f"Allowed {kind} scenario labels: {sorted(allowed)}")
    check_time_grid(frame, [*WINDOW, "model", "scenario"], end_by_model=models, start=0)
    require_paired_models(frame, ["scenario"], identical_windows=False)
    for model, group in frame.groupby("model"):
        if set(group.scenario) != allowed:
            raise ValueError(f"{model} requires baseline and all {kind} scenarios.")
        windows = [
            set(map(tuple, g[WINDOW].drop_duplicates().to_numpy()))
            for _, g in group.groupby("scenario")
        ]
        if any(w != windows[0] for w in windows[1:]):
            raise ValueError("Intervention scenarios must contain identical windows.")
        reference = group.groupby([*WINDOW, "minute"]).ground_truth_cgm.agg(
            ["min", "max"]
        )
        if not np.allclose(reference["min"], reference["max"], rtol=0, atol=1e-6):
            raise ValueError(
                "Observed reference CGM must be identical across added interventions."
            )
    initial = frame.loc[frame.minute == 0]
    if not np.allclose(
        initial.ground_truth_cgm, initial.modeled_cgm, rtol=0, atol=1e-6
    ):
        raise ValueError("At t=0, modeled and reference CGM must equal recorded CGM.")
    return frame


def trajectory_summary(frame):
    keys = [*WINDOW, "model", "scenario"]
    anchor = frame.loc[frame.minute == 0, keys + ["ground_truth_cgm"]].rename(
        columns={"ground_truth_cgm": "cgm0"}
    )
    values = frame.merge(anchor, on=keys, validate="many_to_one")
    values["cgm_change_from_present"] = values.modeled_cgm - values.cgm0
    subject = values.groupby(
        [*SUBJECT, "model", "scenario", "minute"], as_index=False
    ).cgm_change_from_present.mean()
    observed = values.loc[values.scenario == "baseline"].copy()
    observed["cgm_change_from_present"] = observed.ground_truth_cgm - observed.cgm0
    observed = observed.groupby(
        [*SUBJECT, "model", "minute"], as_index=False
    ).cgm_change_from_present.mean()
    observed["scenario"] = "ground_truth"
    return subject_summary(
        pd.concat([subject, observed]),
        ["model", "scenario", "minute"],
        ["cgm_change_from_present"],
    )


def render_trajectories(summary, kind, output, prefix):
    configure()
    for model, model_frame in summary.groupby("model", sort=False):
        panels = ["CHO", "INS", "dual"] if kind == "future" else ["CHO", "INS"]
        for intervention in panels:
            fig, ax = plt.subplots(figsize=(7, 4))
            for scenario, values in model_frame.groupby("scenario", sort=False):
                dual = "CHO" in scenario and "INS" in scenario
                keep = scenario in ("baseline", "ground_truth") or (
                    dual
                    if intervention == "dual"
                    else intervention in scenario and not dual
                )
                if keep:
                    curve(
                        ax,
                        values,
                        label=SCENARIO_LABELS[scenario],
                        linestyle=(
                            "--" if scenario in ("baseline", "ground_truth") else "-"
                        ),
                    )
            times = (
                [0]
                if kind == "present"
                else (
                    [30]
                    if kind == "ablation"
                    else ([10, 50] if intervention == "dual" else [20, 40, 60])
                )
            )
            for t in times:
                ax.axvline(
                    t,
                    color="0.5",
                    linestyle="--",
                    linewidth=0.7,
                    label=(
                        "Intervention timing"
                        if t == times[0] and kind != "present"
                        else None
                    ),
                )
            title = f"{model}\n{intervention}"
            if kind == "future":
                intervention_label = {
                    "CHO": "CHO (50 g)",
                    "INS": "INS (5 U)",
                    "dual": "CHO (50 g), INS (5 U)",
                }[intervention]
                title = f"{model}\n{intervention_label}"
            ax.set(
                xlabel="Time (min)",
                ylabel=r"CGM$_t$ − CGM$_0$ (mg/dL)",
                title=title,
            )
            ax.legend(fontsize=7)
            save_figure(fig, output, f"{prefix}_{filename(model)}_{intervention}")


def validate_curve_summary(summary, kind, models):
    allowed = SCENARIOS[kind] | {"baseline", "ground_truth"}
    if not set(summary.scenario).issubset(allowed) or not set(summary.model).issubset(
        models
    ):
        raise ValueError("Unexpected model or scenario in trajectory summary.")
    check_time_grid(summary, ["model", "scenario"], end_by_model=models, start=0)
    for model, part in summary.groupby("model"):
        if set(part.scenario) != allowed or part.n_subjects.nunique() != 1:
            raise ValueError(
                f"{model} requires every reported scenario and constant subject counts."
            )
    initial = summary.loc[summary.minute == 0]
    if not np.allclose(initial["mean"], 0, atol=1e-8, rtol=0) or not np.allclose(
        initial["std"].fillna(0), 0, atol=1e-8, rtol=0
    ):
        raise ValueError("CGM change and its SD at the present must be zero.")


def sweep_grid(intervention):
    if intervention not in ("CHO", "INS"):
        raise ValueError("Replacement intervention must be CHO or INS (bolus insulin).")
    return np.arange(301, dtype=float) / (10 if intervention == "INS" else 1)


def _check_amounts(frame, keys):
    for _, group in frame.groupby(keys, sort=False):
        amounts = np.sort(group.amount.unique())
        grid = sweep_grid(group.intervention.iloc[0])
        if amounts.shape != grid.shape or not np.allclose(
            amounts, grid, rtol=0, atol=2e-6
        ):
            raise ValueError(
                "Replacement analysis requires all 301 replacement amounts."
            )
    # Round validated replacement amounts to the CHO/INS grid.
    frame["amount"] = frame.amount.astype(float)
    for intervention in frame.intervention.unique():
        selected = frame.intervention == intervention
        frame.loc[selected, "amount"] = frame.loc[selected, "amount"].round(
            1 if intervention == "INS" else 0
        )


def compute_sweeps(path):
    keys = ["model", "intervention", "amount", *WINDOW]
    frame = read_frame(
        path,
        keys + ["minute", "ground_truth_cgm", "modeled_cgm", "recorded_amount"],
        ["amount", "minute", "ground_truth_cgm", "modeled_cgm", "recorded_amount"],
    )
    require_test_rows(frame)
    if "GluCause (Non-personalized)" in set(frame.model):
        raise ValueError(
            "Non-personalized replacement error profiles are not reported in the paper."
        )
    check_time_grid(frame, keys, end_by_model=MODEL_HORIZONS)
    _check_amounts(frame, ["model", "intervention", *WINDOW])
    require_paired_models(frame, ["intervention", "amount"], identical_windows=False)
    for _, group in frame.groupby(["model", "intervention", *WINDOW], sort=False):
        if group.recorded_amount.nunique() != 1 or group.recorded_amount.iloc[0] <= 0:
            raise ValueError(
                "Each replacement window must have one positive recorded intervention amount."
            )
        if (group.groupby("minute").ground_truth_cgm.nunique() != 1).any():
            raise ValueError(
                "Replacement analysis must retain identical reference trajectories."
            )
    frame["error"] = frame.modeled_cgm - frame.ground_truth_cgm
    frame["squared_error"] = frame.error**2
    subjects = []
    for model, model_frame in frame.groupby("model"):
        horizons = [30, 60, 90, 120] if model == "GluCause" else [MODEL_HORIZONS[model]]
        for horizon in horizons:
            windows = (
                model_frame.loc[model_frame.minute <= horizon]
                .groupby(keys, as_index=False)
                .agg(
                    mse=("squared_error", "mean"),
                    MBE=("error", "mean"),
                    recorded_mean=("recorded_amount", "first"),
                )
            )
            windows["RMSE"] = np.sqrt(windows.mse)
            # This experiment averages window RMSE within subject before averaging subjects.
            subject = windows.groupby(
                ["model", "intervention", "amount", *SUBJECT], as_index=False
            )[["RMSE", "MBE", "recorded_mean"]].mean()
            subject["horizon"] = horizon
            subjects.append(subject)
    subject = pd.concat(subjects, ignore_index=True)
    summary = subject_summary(
        subject, ["model", "intervention", "amount", "horizon"], ["RMSE", "MBE"]
    )
    observed = subject.groupby(
        ["model", "intervention", "amount", "horizon"], as_index=False
    ).recorded_mean.mean()
    summary = summary.merge(
        observed,
        on=["model", "intervention", "amount", "horizon"],
        validate="many_to_one",
    )
    return subject, summary


def sweep_minima(summary):
    """Report the first minimum on the grid. Leave full-range constants blank."""
    rows = []
    for key, group in summary.groupby(["model", "intervention", "horizon"]):
        if group.recorded_mean.nunique() != 1:
            raise ValueError(
                "recorded_mean must be constant across a replacement error profile."
            )
        if group.recorded_mean.iloc[0] <= 0 or group.n_subjects.nunique() != 1:
            raise ValueError(
                "Replacement error profiles require positive recorded amounts and constant subject counts."
            )
        row = dict(zip(["model", "intervention", "horizon"], key))
        row["recorded_mean"] = group.recorded_mean.iloc[0]
        for metric in ("RMSE", "MBE"):
            part = group.loc[group.metric == metric].sort_values("amount")
            if len(part) != 301:
                raise ValueError("Supply both full RMSE and MBE profiles.")
            target = part["mean"].to_numpy(float)
            if metric == "MBE":
                target = np.abs(target)
            tied = np.isclose(target, target.min(), rtol=0, atol=1e-8)
            if tied.all():
                row[MINIMUM_AMOUNT_COLUMNS[metric]] = np.nan
            else:
                candidates = part.amount.to_numpy()[tied]
                # The ascending grid identifies the start of a minimum plateau.
                row[MINIMUM_AMOUNT_COLUMNS[metric]] = float(candidates[0])
        rows.append(row)
    return pd.DataFrame(rows)


def render_sweeps(summary, minima, output):
    configure()
    for key, group in summary.groupby(["model", "intervention", "horizon"]):
        model, intervention, horizon = key
        marks = minima.loc[
            (minima.model == model)
            & (minima.intervention == intervention)
            & (minima.horizon == horizon)
        ].iloc[0]
        fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
        for ax, metric in zip(axes, ("RMSE", "MBE")):
            curve(ax, group.loc[group.metric == metric], x="amount", label=metric)
            ax.axvline(
                marks.recorded_mean,
                linestyle="--",
                color="#009E73",
                label="Mean recorded amount",
            )
            minimum = marks[MINIMUM_AMOUNT_COLUMNS[metric]]
            if np.isfinite(minimum):
                ax.axvline(
                    minimum,
                    linestyle="--",
                    color="#D55E00",
                    label=(
                        "Minimum mean RMSE"
                        if metric == "RMSE"
                        else "Minimum |mean MBE|"
                    ),
                )
            if metric == "MBE":
                ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
            ax.set_ylabel(f"{metric} (mg/dL)")
            ax.legend(fontsize=8)
        axes[0].set_title(f"{model}, through {horizon} min")
        axes[-1].set_xlabel(f"{intervention} ({'g' if intervention == 'CHO' else 'U'})")
        prefix = (
            ("fig2" if horizon == 120 else "supp_fig1")
            if model == "GluCause"
            else "supp_fig2"
        )
        save_figure(
            fig, output, f"{prefix}_{filename(model)}_{intervention}_{horizon}min"
        )


def run(args):
    if args.kind == "replacement":
        if args.input:
            subject, summary = compute_sweeps(args.input)
            write_csv(subject, args.output, "replacement_subject_metrics")
        else:
            summary = read_summary(
                args.summary,
                ["model", "intervention", "amount", "horizon"],
                {"RMSE", "MBE"},
            )
            if (
                "recorded_mean" not in summary
                or not np.isfinite(summary.recorded_mean).all()
            ):
                raise ValueError(
                    "Replacement summaries require recorded_mean in g for CHO or U for INS."
                )
            _check_amounts(summary, ["model", "intervention", "horizon", "metric"])
        for row in summary[["model", "horizon"]].drop_duplicates().itertuples():
            allowed = (
                [30, 60, 90, 120]
                if row.model == "GluCause"
                else [MODEL_HORIZONS.get(row.model)]
            )
            if row.model == "GluCause (Non-personalized)" or row.horizon not in allowed:
                raise ValueError(
                    "Replacement model/horizon is not reported in the paper."
                )
        minima = sweep_minima(summary)
        write_csv(
            minima.loc[minima.model == "GluCause"],
            args.output,
            "supp_table1_replacement_amounts",
        )
        write_csv(
            minima.loc[minima.model != "GluCause"],
            args.output,
            "supp_table2_replacement_amounts",
        )
        write_csv(summary, args.output, "replacement_profile_summary")
        render_sweeps(summary, minima, args.output)
    else:
        models = {
            m: h
            for m, h in MODEL_HORIZONS.items()
            if m != "GluCause (Non-personalized)"
        }
        if args.kind == "future":
            models = {m: models[m] for m in ("GluCause", "BGDeepAR-TA (He)")}
        if args.input:
            summary = trajectory_summary(
                load_trajectories(args.input, args.kind, models)
            )
        else:
            summary = read_summary(
                args.summary,
                ["model", "scenario", "minute"],
                {"cgm_change_from_present"},
            )
            validate_curve_summary(summary, args.kind, models)
        prefix = "fig2" if args.kind == "present" else "fig3"
        write_csv(summary, args.output, f"{prefix}_trajectory_summary")
        render_trajectories(summary, args.kind, args.output, prefix)
