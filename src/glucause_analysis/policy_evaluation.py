"""Bolus insulin and rescue carbohydrate policy evaluation for Fig. 6."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import (
    SUBJECT,
    RANGE_METRICS,
    read_frame,
    read_summary,
    check_time_grid,
    glycemic_metrics,
    subject_summary,
    write_csv,
    require_test_rows,
    require_metrics,
)
from .plot_style import configure, save_figure, COLORS

EPISODE = [*SUBJECT, "episode_id"]
DETECTION = ("sensitivity", "specificity", "prevention_rate")
DURATION = "hypoglycemia_duration_minutes"


def _same_cases(frame, keys, group="policy"):
    cases = [
        set(map(tuple, part[keys].drop_duplicates().to_numpy()))
        for _, part in frame.groupby(group)
    ]
    if any(s != cases[0] for s in cases[1:]):
        raise ValueError(
            "All policies, including baseline, must cover identical cases."
        )


def bolus_insulin_subject_metrics(path):
    frame = read_frame(path, [*SUBJECT, "policy", "minute", "cgm"], ["minute", "cgm"])
    require_test_rows(frame)
    check_time_grid(frame, [*SUBJECT, "policy"], end=14 * 24 * 60 - 5, start=0)
    if "Baseline" not in set(frame.policy):
        raise ValueError("Include the recorded test set CGM as policy='Baseline'.")
    _same_cases(frame, SUBJECT)
    rows = []
    for key, group in frame.groupby([*SUBJECT, "policy"]):
        rows.append(
            {**dict(zip([*SUBJECT, "policy"], key)), **glycemic_metrics(group.cgm)}
        )
    return pd.DataFrame(rows)


def rescue_carbohydrate_evaluation(cgm, onset):
    cgm = np.asarray(cgm, dtype=float)
    if cgm.shape != (78,) or not np.isfinite(cgm).all() or onset not in range(5, 31, 5):
        raise ValueError(
            "Rescue carbohydrate policy trajectories require 78 finite points "
            "(5..390 min), onset=5..30 min."
        )
    start = int(onset) // 5 - 1
    period = cgm[start : start + 72]
    recovery = np.flatnonzero(period >= 70)
    duration = 5.0 * (int(recovery[0]) if recovery.size else 72)
    return {
        **glycemic_metrics(period),
        DURATION: duration,
        "prevention_rate": float(100 * (not np.all(cgm[start : start + 3] < 70))),
    }


def rescue_carbohydrate_subject_metrics(trajectory_path, detection_path):
    """Use all recorded hypoglycemia events, including undetected events, for prevention."""
    if not detection_path:
        raise ValueError(
            "Rescue carbohydrate policy evaluation from trajectories requires --detections."
        )
    detections = read_frame(
        detection_path,
        [*EPISODE, "policy", "has_onset", "detected"],
        ["has_onset", "detected"],
        [*EPISODE, "policy"],
    )
    require_test_rows(detections)
    if "onset_minute" not in detections:
        raise ValueError(
            "Detection records require onset_minute (blank for negative cases)."
        )
    detections["onset_minute"] = pd.to_numeric(detections.onset_minute, errors="raise")
    if (
        not detections.has_onset.isin([0, 1]).all()
        or not detections.detected.isin([0, 1]).all()
    ):
        raise ValueError("has_onset and detected must be 0 or 1.")
    if "Baseline" in set(detections.policy):
        raise ValueError("Baseline is a reference trajectory, not a detector.")
    positive = detections.has_onset == 1
    if not detections.loc[positive, "onset_minute"].isin(range(5, 31, 5)).all():
        raise ValueError(
            "Positive cases require a baseline onset at 5, 10, ..., 30 minutes."
        )
    if detections.loc[~positive, "onset_minute"].notna().any():
        raise ValueError("Negative cases must have a blank onset_minute.")
    _same_cases(detections, EPISODE)
    for _, group in detections.groupby(EPISODE):
        if group.has_onset.nunique() != 1 or group.onset_minute.nunique() > 1:
            raise ValueError("Baseline event labels must agree across policies.")
    truth = detections.loc[positive, [*EPISODE, "onset_minute"]].drop_duplicates()
    if truth.empty:
        raise ValueError(
            "No hypoglycemia events are available for rescue carbohydrate policy evaluation."
        )
    trajectories = read_frame(
        trajectory_path, [*EPISODE, "policy", "minute", "cgm"], ["minute", "cgm"]
    )
    require_test_rows(trajectories)
    check_time_grid(trajectories, [*EPISODE, "policy"], end=390)
    if set(trajectories.policy) != set(detections.policy) | {"Baseline"}:
        raise ValueError(
            "Supply a baseline and every detected/undetected policy outcome."
        )
    expected = set(map(tuple, truth[EPISODE].to_numpy()))
    for _, part in trajectories.groupby("policy"):
        if set(map(tuple, part[EPISODE].drop_duplicates().to_numpy())) != expected:
            raise ValueError(
                "Rescue carbohydrate policy outcomes must cover all and only recorded hypoglycemia events."
            )
    trajectories = trajectories.merge(truth, on=EPISODE, validate="many_to_one")
    rows = []
    for key, group in trajectories.groupby([*EPISODE, "policy"]):
        cgm = group.sort_values("minute").cgm.to_numpy()
        onset = int(group.onset_minute.iloc[0])
        metrics = rescue_carbohydrate_evaluation(cgm, onset)
        if key[-1] == "Baseline":
            starts = [i for i in range(6) if np.all(cgm[i : i + 3] < 70)]
            if not starts or (starts[0] + 1) * 5 != onset:
                raise ValueError(
                    "Recorded onset is inconsistent with the baseline trajectory."
                )
            metrics.pop("prevention_rate")
        rows.append({**dict(zip([*EPISODE, "policy"], key)), **metrics})
    episodes = pd.DataFrame(rows)
    subject = episodes.groupby([*SUBJECT, "policy"], as_index=False)[
        [*RANGE_METRICS, DURATION, "prevention_rate"]
    ].mean()
    detection_rows = []
    for key, part in detections.groupby([*SUBJECT, "policy"]):
        positive = part.has_onset == 1
        sensitivity = (
            float(100 * part.loc[positive, "detected"].mean())
            if positive.any()
            else np.nan
        )
        specificity = (
            float(100 * (1 - part.loc[~positive, "detected"]).mean())
            if (~positive).any()
            else np.nan
        )
        detection_rows.append(
            {
                **dict(zip([*SUBJECT, "policy"], key)),
                "sensitivity": sensitivity,
                "specificity": specificity,
            }
        )
    subject = subject.merge(
        pd.DataFrame(detection_rows),
        on=[*SUBJECT, "policy"],
        how="outer",
        validate="one_to_one",
    )
    return subject


def _range_plot(summary, output, name):
    frame = summary.loc[summary.metric.isin(RANGE_METRICS)]
    policies = list(frame.policy.unique())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(policies)
    x = np.arange(len(RANGE_METRICS))
    for i, policy in enumerate(policies):
        part = (
            frame.loc[frame.policy == policy].set_index("metric").reindex(RANGE_METRICS)
        )
        positions = x - 0.4 + width / 2 + i * width
        ax.bar(
            positions, part["mean"], width, label=policy, color=COLORS[i % len(COLORS)]
        )
        good = part["std"].notna().to_numpy()
        ax.errorbar(
            positions[good],
            part["mean"].to_numpy()[good],
            yerr=part["std"].to_numpy()[good],
            fmt="none",
            color="black",
            linewidth=0.8,
            capsize=2,
        )
    ax.set_xticks(
        x, ["TIR", r"TBR$_{70}$", r"TBR$_{54}$", r"TAR$_{180}$", r"TAR$_{250}$"]
    )
    ax.set_ylabel("Percentage (%)")
    ax.legend(fontsize=8)
    save_figure(fig, output, name)


def render(summary, kind, output):
    configure()
    _range_plot(
        summary,
        output,
        (
            "fig6a_bolus_insulin"
            if kind == "bolus_insulin"
            else "fig6d_rescue_carbohydrate"
        ),
    )
    if kind == "bolus_insulin":
        return
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    for policy, part in summary.loc[summary.metric.isin(DETECTION)].groupby(
        "policy", sort=False
    ):
        values = part.set_index("metric").reindex(DETECTION)["mean"].to_numpy()
        closed_angles, closed = np.r_[angles, angles[0]], np.r_[values, values[0]]
        (line,) = ax.plot(closed_angles, closed, label=policy)
        ax.fill(closed_angles, closed, alpha=0.08, color=line.get_color())
    ax.set_xticks(angles, ["Sensitivity", "Specificity", "Prevention rate"])
    ax.tick_params(axis="x", pad=18)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)
    save_figure(fig, output, "fig6b_rescue_carbohydrate")
    duration = summary.loc[summary.metric == DURATION]
    baseline = duration.loc[duration.policy == "Baseline"]
    part = duration.loc[duration.policy != "Baseline"]
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(part))
    ax.barh(y, part["mean"], color=COLORS[0])
    good = part["std"].notna().to_numpy()
    ax.errorbar(
        part["mean"].to_numpy()[good],
        y[good],
        xerr=part["std"].to_numpy()[good],
        fmt="none",
        color="black",
        capsize=3,
    )
    labels = list(part.policy)
    if not baseline.empty:
        ref = baseline.iloc[0]
        ax.axvline(ref["mean"], color="black", linestyle="--")
        ax.errorbar(
            [ref["mean"]],
            [len(part)],
            xerr=None if np.isnan(ref["std"]) else [ref["std"]],
            fmt="o",
            color="black",
            capsize=3,
        )
        labels.append("Baseline")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Hypoglycemia duration (min)")
    save_figure(fig, output, "fig6c_rescue_carbohydrate")


def run(args):
    metrics = (
        list(RANGE_METRICS)
        if args.kind == "bolus_insulin"
        else [*RANGE_METRICS, *DETECTION, DURATION]
    )
    if args.input:
        subject = (
            bolus_insulin_subject_metrics(args.input)
            if args.kind == "bolus_insulin"
            else rescue_carbohydrate_subject_metrics(args.input, args.detections)
        )
        summary = subject_summary(subject, ["policy"], metrics)
        summary = summary.loc[summary.n_subjects > 0].reset_index(drop=True)
        write_csv(subject, args.output, f"{args.kind}_subject_metrics")
    else:
        summary = read_summary(args.summary, ["policy"], set(metrics))
    if "Baseline" not in set(summary.policy):
        raise ValueError("Include baseline summary metrics.")
    if set(summary.policy) == {"Baseline"}:
        raise ValueError(
            "Policy evaluation requires at least one policy and its baseline."
        )
    for policy, part in summary.groupby("policy"):
        required = set(RANGE_METRICS)
        if args.kind == "rescue_carbohydrate":
            required.add(DURATION)
            if policy != "Baseline":
                required.update(DETECTION)
        require_metrics(part, [], required)
        values = part.set_index("metric")["mean"]
        if not np.isclose(
            values.TIR_70_180 + values.TBR_below_70 + values.TAR_above_180,
            100,
            atol=0.01,
            rtol=0,
        ):
            raise ValueError("TIR + TBR below 70 + TAR above 180 must sum to 100%.")
        if (
            values.TBR_below_54 > values.TBR_below_70 + 0.01
            or values.TAR_above_250 > values.TAR_above_180 + 0.01
        ):
            raise ValueError("Nested TBR/TAR percentages are inconsistent.")
    ranges = summary.loc[summary.metric.isin(RANGE_METRICS)]
    if ranges.n_subjects.nunique() != 1:
        raise ValueError(
            "Range metrics must describe the same virtual subjects under all policies."
        )
    if args.kind == "rescue_carbohydrate":
        positive = summary.loc[summary.metric != "specificity"]
        if positive.n_subjects.nunique() != 1:
            raise ValueError(
                "Rescue carbohydrate policy range metrics, hypoglycemia duration, "
                "sensitivity, and prevention rate must have the same positive-case subject count."
            )
        specificity = summary.loc[summary.metric == "specificity"]
        if specificity.n_subjects.nunique() != 1:
            raise ValueError(
                "Specificity must have the same negative-case subject count "
                "across policies. It may differ from the positive-case count."
            )
    rate = summary.metric != DURATION
    if not summary.loc[rate, "mean"].between(0, 100).all():
        raise ValueError(
            "Rates and time-in-range metrics must be percentages in [0,100]."
        )
    if not summary.loc[~rate, "mean"].between(0, 360).all():
        raise ValueError("Hypoglycemia duration must be in [0,360] minutes.")
    write_csv(summary, args.output, f"fig6_{args.kind}_summary")
    render(summary, args.kind, args.output)
