"""Latent representation analyses for Fig. 5a–g and Supplementary Note 2."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from .metrics import (
    SUBJECT,
    WINDOW,
    read_frame,
    require_columns,
    require_unique,
    check_time_grid,
    subject_summary,
    write_csv,
    validate_statistics,
    require_test_rows,
)
from .plot_style import configure, curve, save_figure

MODEL = "GluCause (Non-personalized)"
KEY = ["split", *WINDOW]
ENCODER_REPRESENTATION_COLUMNS = [f"encoder_representation_{i:03d}" for i in range(256)]
TARGETS = [
    "cgm0",
    "past_cho_exposure_proxy",
    "past_ins_exposure_proxy",
    "cho_response_score",
    "ins_response_score",
]
TARGET_LABELS = {
    "cgm0": "Present CGM",
    "past_cho_exposure_proxy": "Past CHO exposure proxy",
    "past_ins_exposure_proxy": "Past INS exposure proxy",
    "cho_response_score": "Carbohydrate response score",
    "ins_response_score": "Insulin response score",
    "cho_ins_balance": "Carbohydrate–insulin balance",
}
PROBE_LABELS = {
    **TARGET_LABELS,
    "cho_response_score": "Response to present CHO",
    "ins_response_score": "Response to present INS",
}
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
TDD_EXPECTED_RECORDS_PER_DAY = 288
TDD_MIN_DAY_COVERAGE = 0.95
TDD_MIN_VALID_DAYS = 2
# Eligibility criteria for Spearman rank correlation analysis.
RANK_CORRELATION_MIN_SUBJECTS_PER_DATASET = 3
POOLED_MIN_DATASETS = 2
POOLED_MIN_SUBJECTS = 6


def subject_weights(frame):
    count = frame.groupby(SUBJECT).subject_id.transform("size").to_numpy(float)
    weights = 1.0 / count
    return weights * len(weights) / weights.sum()


def exposure_proxy(amounts, half_life):
    """Columns must correspond to lags 5, 10, ..., 715 min, in that order."""
    amounts = np.asarray(amounts, dtype=float)
    if (
        amounts.ndim != 2
        or amounts.shape[1] != 143
        or not np.isfinite(amounts).all()
        or (amounts < 0).any()
    ):
        raise ValueError(
            "Exposure inputs require 143 finite, nonnegative past amounts."
        )
    return amounts @ np.power(2.0, -np.arange(5, 716, 5) / half_life)


def weighted_r2(actual, estimate, weight):
    mean = np.average(actual, weights=weight)
    total = np.sum(weight * (actual - mean) ** 2)
    return (
        float(1 - np.sum(weight * (actual - estimate) ** 2) / total)
        if total > 1e-8
        else np.nan
    )


def weighted_quantile(values, weight, fraction):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weight[order])
    index = min(
        np.searchsorted(cumulative, fraction * cumulative[-1], side="left"),
        len(order) - 1,
    )
    return float(np.asarray(values)[order[index]])


def _require_non_personalized(frame, test_only=False):
    require_columns(frame, ["model", "split"])
    if set(frame.model) != {MODEL}:
        raise ValueError(
            "Latent analyses require the same GluCause (Non-personalized) model throughout."
        )
    allowed = {"Test"} if test_only else {"Training", "Validation", "Test"}
    if not set(frame.split).issubset(allowed):
        raise ValueError(f"Allowed splits: {sorted(allowed)}")


def load_representations(feature_path, response_path):
    if not response_path:
        raise ValueError("Panels a–d require --responses as well as --input.")
    cho = [f"cho_lag_{t}" for t in range(5, 716, 5)]
    ins = [f"ins_lag_{t}" for t in range(5, 716, 5)]
    numeric = ["window_index", "cgm0", *ENCODER_REPRESENTATION_COLUMNS, *cho, *ins]
    frame = read_frame(feature_path, [*KEY, "model", *numeric], numeric, KEY)
    _require_non_personalized(frame)
    if set(frame.split) != {"Training", "Validation", "Test"}:
        raise ValueError(
            "Mamba Encoder representation analysis requires all three splits."
        )
    if (frame.window_index < 0).any() or (frame.window_index % 1 != 0).any():
        raise ValueError(
            "window_index must be a nonnegative integer within subject and split."
        )
    require_unique(frame, ["split", *SUBJECT, "window_index"])
    frame["past_cho_exposure_proxy"] = exposure_proxy(frame[cho].to_numpy(), 60)
    frame["past_ins_exposure_proxy"] = exposure_proxy(frame[ins].to_numpy(), 180)
    responses = read_frame(
        response_path,
        [*KEY, "model", "minute", "baseline_cgm", "cho_cgm", "ins_cgm"],
        ["minute", "baseline_cgm", "cho_cgm", "ins_cgm"],
    )
    _require_non_personalized(responses)
    check_time_grid(responses, KEY, end=120)
    responses["cho_response_score"] = responses.cho_cgm - responses.baseline_cgm
    responses["ins_response_score"] = responses.baseline_cgm - responses.ins_cgm
    scores = responses.groupby(KEY, as_index=False)[
        ["cho_response_score", "ins_response_score"]
    ].mean()
    if set(map(tuple, frame[KEY].to_numpy())) != set(
        map(tuple, scores[KEY].to_numpy())
    ):
        raise ValueError(
            "Representations and responses must cover identical windows and splits."
        )
    return frame.drop(columns=cho + ins).merge(scores, on=KEY, validate="one_to_one")


def _sample(frame, cap):
    rng = np.random.default_rng(42)
    selected = []
    # Traverse subjects by ascending subject_id, using dataset to break ties.
    ordered = frame.sort_values(["subject_id", "dataset", "window_index"])
    for _, part in ordered.groupby(SUBJECT, sort=False):
        candidates = part.loc[part.window_index % 12 == 0].index.to_numpy()
        if len(candidates) > cap:
            candidates = np.sort(rng.choice(candidates, cap, replace=False))
        selected.extend(candidates)
    return frame.loc[selected].copy()


def project_representations(frame):
    training = frame.loc[frame.split == "Training"]
    fit = _sample(training, 1000)
    test_source = frame.loc[frame.split == "Test"]
    test = _sample(test_source, 100)
    if len(test[SUBJECT].drop_duplicates()) != len(
        test_source[SUBJECT].drop_duplicates()
    ):
        raise ValueError(
            "Each subject in the test set needs at least one PCA window whose original "
            "window_index is divisible by 12. Supply those windows without "
            "renumbering the original indices."
        )
    if len(fit) < 2 or test.empty:
        raise ValueError("Not enough every-twelfth training/test set windows for PCA.")
    scaler = StandardScaler().fit(fit[ENCODER_REPRESENTATION_COLUMNS].to_numpy())
    pca = PCA(n_components=2, svd_solver="full").fit(
        scaler.transform(fit[ENCODER_REPRESENTATION_COLUMNS].to_numpy())
    )
    coordinates = pca.transform(
        scaler.transform(test[ENCODER_REPRESENTATION_COLUMNS].to_numpy())
    )
    projection = test[[*SUBJECT, "window_id", *TARGETS]].copy()
    projection[["pc1", "pc2"]] = coordinates
    projection["pc1_variance_pct"] = 100 * pca.explained_variance_ratio_[0]
    projection["pc2_variance_pct"] = 100 * pca.explained_variance_ratio_[1]
    weight = subject_weights(training)
    standardized = []
    for target in ("cho_response_score", "ins_response_score"):
        mean = np.average(training[target], weights=weight)
        variance = np.average((training[target] - mean) ** 2, weights=weight)
        if variance <= 0:
            raise ValueError(
                "A response score has zero training set variance. "
                "Carbohydrate–insulin balance is undefined."
            )
        standardized.append((test[target].to_numpy() - mean) / np.sqrt(variance))
    projection["cho_ins_balance"] = standardized[0] - standardized[1]
    return projection


def fit_ridge_probes(frame):
    """Fit ridge probes to Mamba Encoder representations.

    Select each penalty on the validation set, refit on the combined training
    and validation sets, and evaluate on the test set.
    """
    training, validation, test = [
        frame.loc[frame.split == split] for split in ("Training", "Validation", "Test")
    ]
    combined = pd.concat([training, validation], ignore_index=True)
    wt, wv, ws, wc = [
        subject_weights(x) for x in (training, validation, test, combined)
    ]
    scaler = StandardScaler().fit(
        training[ENCODER_REPRESENTATION_COLUMNS].to_numpy(), sample_weight=wt
    )
    xt, xv = [
        scaler.transform(x[ENCODER_REPRESENTATION_COLUMNS].to_numpy())
        for x in (training, validation)
    ]
    final_scaler = StandardScaler().fit(
        combined[ENCODER_REPRESENTATION_COLUMNS].to_numpy(), sample_weight=wc
    )
    xc, xs = [
        final_scaler.transform(x[ENCODER_REPRESENTATION_COLUMNS].to_numpy())
        for x in (combined, test)
    ]
    probes, states = [], []
    for target in TARGETS:
        candidates = []
        for alpha in ALPHAS:
            probe = Ridge(alpha=alpha, solver="cholesky").fit(
                xt, training[target], sample_weight=wt
            )
            score = weighted_r2(validation[target].to_numpy(), probe.predict(xv), wv)
            candidates.append((alpha, score))
        valid = [(alpha, score) for alpha, score in candidates if np.isfinite(score)]
        if not valid:
            raise ValueError(f"Validation R² is undefined for target {target}.")
        best = max(score for _, score in valid)
        alpha = max(a for a, score in valid if abs(score - best) <= 1e-12)
        final = Ridge(alpha=alpha, solver="cholesky").fit(
            xc, combined[target], sample_weight=wc
        )
        estimate = final.predict(xs)
        probes.append(
            {"target": target, "r2": weighted_r2(test[target].to_numpy(), estimate, ws)}
        )
        if target.endswith("response_score"):
            cuts = [weighted_quantile(estimate, ws, q) for q in (1 / 3, 2 / 3)]
            assigned = np.asarray(["Low", "Middle", "High"])[
                np.searchsorted(cuts, estimate, side="left")
            ]
            part = (
                test[[*SUBJECT, target]]
                .rename(columns={target: "response_score"})
                .copy()
            )
            part["state"] = assigned
            part = part.groupby(
                [*SUBJECT, "state"], as_index=False
            ).response_score.mean()
            part["intervention"] = "CHO" if target.startswith("cho") else "INS"
            states.append(part)
    return pd.DataFrame(probes), pd.concat(states, ignore_index=True)


def dynamics_head_displacement_summary(path):
    columns = {
        name: [f"{name}_{i:03d}" for i in range(128)]
        for name in ("baseline", "cho", "ins")
    }
    vectors = sum(columns.values(), [])
    frame = read_frame(
        path, [*WINDOW, "split", "model", "minute", *vectors], ["minute", *vectors]
    )
    _require_non_personalized(frame, test_only=True)
    check_time_grid(frame, WINDOW, end=120)
    baseline = frame[columns["baseline"]].to_numpy()
    cho, ins = [frame[columns[name]].to_numpy() - baseline for name in ("cho", "ins")]
    cho_norm, ins_norm = np.linalg.norm(cho, axis=1), np.linalg.norm(ins, axis=1)
    cosine = np.full(len(frame), np.nan)
    valid = (cho_norm > 1e-8) & (ins_norm > 1e-8)
    cosine[valid] = np.clip(
        np.sum(cho[valid] * ins[valid], axis=1) / (cho_norm[valid] * ins_norm[valid]),
        -1,
        1,
    )
    values = frame[[*SUBJECT, "minute"]].copy()
    values["cho_l2"], values["ins_l2"], values["cosine"] = cho_norm, ins_norm, cosine
    subject = values.groupby([*SUBJECT, "minute"], as_index=False)[
        ["cho_l2", "ins_l2", "cosine"]
    ].mean()
    return subject_summary(subject, ["minute"], ["cho_l2", "ins_l2", "cosine"], ddof=1)


def training_tdd(path):
    """Median daily total insulin in the training set, requiring 95% coverage and two valid days."""
    frame = read_frame(
        path,
        [*SUBJECT, "split", "date", "Basal", "Bolus", "INS"],
        ["Basal", "Bolus", "INS"],
    )
    if set(frame.split) != {"Training"}:
        raise ValueError("TDD input must contain only canonical training set records.")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if (frame.date != frame.date.dt.floor("5min")).any():
        raise ValueError("TDD timestamps must lie on the 5-minute grid.")
    require_unique(frame, [*SUBJECT, "date"])
    if (frame[["Basal", "Bolus", "INS"]] < 0).any().any() or not np.allclose(
        frame.INS, frame.Basal + frame.Bolus, rtol=0, atol=1e-6
    ):
        raise ValueError(
            "TDD requires nonnegative canonical U/5min insulin and INS = Basal + Bolus."
        )
    frame["day"] = frame.date.dt.normalize()
    daily = frame.groupby([*SUBJECT, "day"], as_index=False).agg(
        INS=("INS", "sum"), unique_slots=("date", "nunique")
    )
    minimum_slots = int(np.ceil(TDD_EXPECTED_RECORDS_PER_DAY * TDD_MIN_DAY_COVERAGE))
    daily = daily.loc[daily.unique_slots >= minimum_slots]
    # Use recorded totals without imputing missing slots or scaling to 24 hours.
    tdd = daily.groupby(SUBJECT, as_index=False).agg(
        tdd=("INS", "median"), valid_days=("day", "size")
    )
    tdd = tdd.loc[tdd.valid_days >= TDD_MIN_VALID_DAYS, [*SUBJECT, "tdd"]].copy()
    if (tdd.tdd <= 0).any():
        raise ValueError("ISF proxy is undefined for zero median daily insulin.")
    tdd["isf_proxy"] = 1800.0 / tdd.tdd
    return tdd


def insulin_sensitivity_rank_analysis(response_path, training_path):
    if not training_path:
        raise ValueError(
            "Panels f–g require --training-insulin with canonical training set records."
        )
    frame = read_frame(
        response_path,
        [
            *WINDOW,
            "split",
            "model",
            "bolus_insulin_amount",
            "minute",
            "baseline_cgm",
            "modeled_cgm",
        ],
        ["bolus_insulin_amount", "minute", "baseline_cgm", "modeled_cgm"],
    )
    _require_non_personalized(frame, test_only=True)
    check_time_grid(frame, [*WINDOW, "bolus_insulin_amount"], end=120)
    for _, part in frame.groupby(WINDOW):
        if set(part.bolus_insulin_amount) != {0, 3, 6, 9}:
            raise ValueError(
                "Insulin response slope analysis requires bolus insulin amounts "
                "of 0, 3, 6, and 9 U for every window."
            )
        if (part.groupby("minute").baseline_cgm.nunique() != 1).any():
            raise ValueError(
                "Bolus insulin responses must share a baseline trajectory."
            )
        zero = part.loc[part.bolus_insulin_amount == 0]
        if not np.allclose(zero.baseline_cgm, zero.modeled_cgm, rtol=0, atol=1e-6):
            raise ValueError("The 0 U trajectory must equal baseline.")
    frame["score"] = frame.baseline_cgm - frame.modeled_cgm
    scores = frame.groupby(
        [*WINDOW, "bolus_insulin_amount"], as_index=False
    ).score.mean()
    matrix = scores.pivot(
        index=WINDOW, columns="bolus_insulin_amount", values="score"
    ).reindex(columns=[0, 3, 6, 9])
    centered_insulin_amount = np.array([0.0, 3.0, 6.0, 9.0]) - 4.5
    slopes = matrix.index.to_frame(index=False)
    slopes["insulin_response_slope"] = (
        matrix.to_numpy()
        @ centered_insulin_amount
        / np.square(centered_insulin_amount).sum()
    )
    subjects = slopes.groupby(SUBJECT, as_index=False).insulin_response_slope.median()
    subjects = subjects.merge(
        training_tdd(training_path), on=SUBJECT, how="left", validate="one_to_one"
    )
    if subjects.tdd.isna().any():
        raise ValueError(
            "Some subjects in the test set have no eligible training set TDD. "
            "Each requires at least two training set days with at least "
            "274 unique 5-minute records per day."
        )
    xcol, ycol = "isf_proxy_relative_rank", "insulin_response_slope_relative_rank"
    subjects[xcol] = np.nan
    subjects[ycol] = np.nan
    correlations = []
    for dataset, part in subjects.groupby("dataset"):
        n = len(part)
        x, y = part.isf_proxy, part.insulin_response_slope
        valid = (
            n >= RANK_CORRELATION_MIN_SUBJECTS_PER_DATASET
            and x.nunique() > 1
            and y.nunique() > 1
        )
        rho = spearmanr(x, y).statistic if valid else np.nan
        correlations.append({"dataset": dataset, "rho": rho, "n_subjects": n})
        # Only datasets with a defined correlation contribute relative ranks.
        if valid:
            for values, column in ((x, xcol), (y, ycol)):
                ranks = values.rank(method="average")
                subjects.loc[part.index, column] = (ranks - 1) / (n - 1)
    ranked = subjects.dropna(subset=[xcol, ycol]).copy()
    available = (
        ranked.dataset.nunique() >= POOLED_MIN_DATASETS
        and len(ranked) >= POOLED_MIN_SUBJECTS
    )
    if not available:
        ranked = ranked.iloc[0:0]
    rho = spearmanr(ranked[xcol], ranked[ycol]).statistic if available else np.nan
    ranked["bin"] = np.minimum((ranked[xcol] * 8).astype(int), 7)
    bins = []
    for index in range(8):
        part = ranked.loc[ranked.bin == index]
        bins.append(
            {
                "bin": index,
                "x_mean": part[xcol].mean(),
                "mean": part[ycol].mean(),
                "std": part[ycol].std(ddof=1),
                "n_subjects": len(part),
                "pooled_rho": rho,
            }
        )
    return subjects, pd.DataFrame(correlations), pd.DataFrame(bins)


def plot_abcd(projection, probes, states, output):
    configure()
    for panel, targets in (
        ("a", ["past_cho_exposure_proxy", "past_ins_exposure_proxy"]),
        ("b", ["cho_response_score", "ins_response_score", "cho_ins_balance"]),
    ):
        for target in targets:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            points = ax.scatter(
                projection.pc1,
                projection.pc2,
                c=projection[target],
                s=8,
                alpha=0.7,
                rasterized=True,
                cmap="coolwarm" if target == "cho_ins_balance" else "viridis",
            )
            fig.colorbar(points, ax=ax, label=TARGET_LABELS[target])
            ax.set(
                xlabel="PC1",
                ylabel="PC2",
            )
            save_figure(fig, output, f"fig5{panel}_{target}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([PROBE_LABELS[target] for target in probes.target], probes.r2)
    ax.set_ylabel("R²")
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, output, "fig5c_ridge_probes")
    for intervention, part in states.groupby("intervention"):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for i, state in enumerate(("Low", "Middle", "High")):
            values = part.loc[part.state == state, "response_score"].to_numpy()
            if len(values):
                offsets = (
                    np.linspace(-0.15, 0.15, len(values)) if len(values) > 1 else [0]
                )
                ax.scatter(i + np.asarray(offsets), values, alpha=0.6, s=18)
                ax.hlines(values.mean(), i - 0.3, i + 0.3, color="black")
        ax.set_xticks([0, 1, 2], ["Low", "Middle", "High"])
        ax.set(
            xlabel="Response state",
            ylabel=(
                "CGM increase (mg/dL)"
                if intervention == "CHO"
                else "CGM decrease (mg/dL)"
            ),
            title=intervention,
        )
        save_figure(fig, output, f"fig5d_{intervention}")


def plot_dynamics_head_displacements(summary, output):
    configure()
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ("cho_l2", "ins_l2"):
        curve(
            ax,
            summary.loc[summary.metric == metric],
            label={"cho_l2": "CHO", "ins_l2": "INS"}[metric],
        )
    second = ax.twinx()
    cosine = summary.loc[summary.metric == "cosine"].sort_values("minute")
    second.plot(cosine.minute, cosine["mean"], "k--", label="CHO–INS")
    second.set(ylabel="Cosine similarity", ylim=(-1, 1))
    ax.set(
        xlabel="Time (min)",
        ylabel="L2 norm",
        title="Dynamics Head displacement vectors",
    )
    a, b = ax.get_legend_handles_labels()
    c, d = second.get_legend_handles_labels()
    ax.legend(a + c, b + d, fontsize=8)
    save_figure(fig, output, "fig5e_dynamics_head_displacement")


def plot_insulin_sensitivity_ranks(correlations, bins, output):
    configure()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(correlations.dataset, correlations.rho)
    ax.set(
        ylabel="Spearman ρ",
        ylim=(-1, 1),
        title="ISF proxy and INS response slope",
    )
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, output, "fig5f_within_dataset_ranks")
    fig, ax = plt.subplots(figsize=(5, 5))
    curve(
        ax,
        bins.dropna(subset=["x_mean", "mean"]),
        x="x_mean",
        marker="o",
        label="Mean relative rank",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Identical relative ranks")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Relative rank of ISF proxy",
        ylabel="Relative rank of INS response slope",
        title=(
            f"Pooled Spearman ρ = {bins.pooled_rho.iloc[0]:.3f}"
            if np.isfinite(bins.pooled_rho.iloc[0])
            else "Pooled analysis not available"
        ),
    )
    ax.legend(fontsize=8)
    save_figure(fig, output, "fig5g_pooled_relative_ranks")


def run(args):
    if args.panel == "abcd":
        if args.input:
            frame = load_representations(args.input, args.responses)
            projection = project_representations(frame)
            probes, states = fit_ridge_probes(frame)
        else:
            root = Path(args.summary)
            projection = read_frame(
                root / "fig5ab_projection.csv",
                [
                    *SUBJECT,
                    "window_id",
                    *TARGETS,
                    "cho_ins_balance",
                    "pc1",
                    "pc2",
                    "pc1_variance_pct",
                    "pc2_variance_pct",
                ],
                [
                    *TARGETS,
                    "cho_ins_balance",
                    "pc1",
                    "pc2",
                    "pc1_variance_pct",
                    "pc2_variance_pct",
                ],
                WINDOW,
            )
            probes = read_frame(
                root / "fig5c_ridge_probes.csv", ["target"], unique=["target"]
            )
            states = read_frame(
                root / "fig5d_subject_states.csv",
                [*SUBJECT, "intervention", "state", "response_score"],
                ["response_score"],
                [*SUBJECT, "intervention", "state"],
            )
            for table in (projection, probes, states):
                require_test_rows(table)
            variance = projection[["pc1_variance_pct", "pc2_variance_pct"]]
            if (
                (variance.nunique() != 1).any()
                or (variance < 0).any().any()
                or (variance.sum(axis=1) > 100 + 1e-6).any()
            ):
                raise ValueError(
                    "PCA explained-variance percentages must be constant and sum to at most 100."
                )
            if "r2" not in probes:
                raise ValueError(
                    "Ridge probe summaries require r2 (blank when undefined)."
                )
            probes["r2"] = pd.to_numeric(probes.r2, errors="raise")
            if np.isinf(probes.r2).any() or (probes.r2 > 1 + 1e-8).any():
                raise ValueError("Invalid ridge probe R² values.")
            if (
                set(probes.target) != set(TARGETS)
                or not set(states.state).issubset({"Low", "Middle", "High"})
                or set(states.intervention) != {"CHO", "INS"}
            ):
                raise ValueError("Unexpected ridge probe targets or response states.")
        projection_subjects = set(
            map(tuple, projection[SUBJECT].drop_duplicates().to_numpy())
        )
        for intervention, part in states.groupby("intervention"):
            state_subjects = set(map(tuple, part[SUBJECT].drop_duplicates().to_numpy()))
            if state_subjects != projection_subjects:
                raise ValueError(
                    f"PCA projection and {intervention} response states must contain "
                    "the same (dataset, subject_id) cohort."
                )
        write_csv(projection, args.output, "fig5ab_projection")
        write_csv(probes, args.output, "fig5c_ridge_probes")
        write_csv(states, args.output, "fig5d_subject_states")
        plot_abcd(projection, probes, states, args.output)
    elif args.panel == "e":
        summary = (
            dynamics_head_displacement_summary(args.input)
            if args.input
            else read_frame(
                args.summary,
                ["minute", "metric", "n_subjects"],
                ["minute", "n_subjects"],
                ["minute", "metric"],
            )
        )
        if set(summary.metric) != {"cho_l2", "ins_l2", "cosine"} or not {
            "mean",
            "std",
        }.issubset(summary):
            raise ValueError(
                "Dynamics Head displacement summaries require cho_l2, ins_l2, cosine, mean, and std."
            )
        check_time_grid(summary, ["metric"], end=120)
        require_test_rows(summary)
        validate_statistics(summary, allow_empty=True)
        norms = summary.loc[summary.metric != "cosine"]
        cosines = summary.loc[summary.metric == "cosine"]
        if (
            (norms.n_subjects < 1).any()
            or (norms["mean"] < 0).any()
            or norms.n_subjects.nunique() != 1
        ):
            raise ValueError(
                "Displacement L2 norms must be nonnegative and use the same subject cohort throughout."
            )
        if (
            not cosines["mean"].dropna().between(-1, 1).all()
            or (cosines.n_subjects > norms.n_subjects.iloc[0]).any()
        ):
            raise ValueError(
                "Invalid cosine similarity between displacement vectors or contributing-subject count."
            )
        write_csv(summary, args.output, "fig5e_summary")
        plot_dynamics_head_displacements(summary, args.output)
    else:
        if args.input:
            subjects, correlations, bins = insulin_sensitivity_rank_analysis(
                args.input, args.training_insulin
            )
            write_csv(subjects, args.output, "fig5fg_subjects")
        else:
            root = Path(args.summary)
            correlations = read_frame(
                root / "fig5f_correlations.csv",
                ["dataset", "n_subjects"],
                ["n_subjects"],
                ["dataset"],
            )
            bins = read_frame(
                root / "fig5g_bins.csv",
                ["bin", "n_subjects"],
                ["bin", "n_subjects"],
                ["bin"],
            )
            if "rho" not in correlations or not {
                "x_mean",
                "mean",
                "std",
                "pooled_rho",
            }.issubset(bins):
                raise ValueError("Missing rank correlation summary statistics.")
            if set(bins.bin) != set(range(8)):
                raise ValueError("Panel g requires eight equal-width rank intervals.")
        for table in (correlations, bins):
            require_test_rows(table)
        validate_statistics(bins, allow_empty=True)
        for table, columns in (
            (correlations, ["rho"]),
            (bins, ["x_mean", "pooled_rho"]),
        ):
            for col in columns:
                table[col] = pd.to_numeric(table[col], errors="raise")
                if np.isinf(table[col]).any():
                    raise ValueError(
                        "Rank correlation summaries cannot contain infinite statistics."
                    )
        if (
            not correlations.rho.dropna().between(-1, 1).all()
            or not bins.pooled_rho.dropna().between(-1, 1).all()
        ):
            raise ValueError("Spearman correlations must be in [-1,1].")
        if (correlations.n_subjects < 1).any() or (
            correlations.n_subjects % 1 != 0
        ).any():
            raise ValueError("Invalid subject count for rank correlation analysis.")
        if (
            (correlations.n_subjects < RANK_CORRELATION_MIN_SUBJECTS_PER_DATASET)
            & correlations.rho.notna()
        ).any():
            raise ValueError(
                "A dataset correlation needs at least "
                f"{RANK_CORRELATION_MIN_SUBJECTS_PER_DATASET} subjects."
            )
        if bins.pooled_rho.nunique(dropna=False) != 1:
            raise ValueError("Every rank bin must report the same pooled correlation.")
        present = bins.n_subjects > 0
        if (
            bins.loc[present, "x_mean"].isna().any()
            or bins.loc[~present, "x_mean"].notna().any()
        ):
            raise ValueError(
                "Rank-bin horizontal means must match their subject counts."
            )
        if not bins.loc[present, "mean"].between(0, 1).all():
            raise ValueError("Mean relative ranks must be in [0,1].")
        for row in bins.loc[present].itertuples():
            if not row.bin / 8 - 1e-10 <= row.x_mean <= (row.bin + 1) / 8 + 1e-10:
                raise ValueError("A horizontal rank mean lies outside its bin.")
        valid = correlations.rho.notna()
        pooled_subjects = correlations.loc[valid, "n_subjects"].sum()
        available = (
            valid.sum() >= POOLED_MIN_DATASETS
            and pooled_subjects >= POOLED_MIN_SUBJECTS
        )
        if bins.n_subjects.sum() != (pooled_subjects if available else 0) or (
            bins.pooled_rho.notna().all() != available
        ):
            raise ValueError(
                "Rank bins must cover the subjects of datasets with defined "
                f"correlations, with at least {POOLED_MIN_DATASETS} datasets and "
                f"{POOLED_MIN_SUBJECTS} subjects. Otherwise they remain blank."
            )
        write_csv(correlations, args.output, "fig5f_correlations")
        write_csv(bins, args.output, "fig5g_bins")
        plot_insulin_sensitivity_ranks(correlations, bins, args.output)
