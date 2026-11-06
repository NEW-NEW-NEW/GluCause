"""Shared numerical definitions and table readers."""

from pathlib import Path

import numpy as np
import pandas as pd

SUBJECT = ["dataset", "subject_id"]
WINDOW = SUBJECT + ["window_id"]
MODEL_HORIZONS = {
    "GluCause": 120,
    "GluCause (Non-personalized)": 120,
    "BG-BERT": 60,
    "BGDeepAR-TA (He)": 90,
    "Diabetes-SeqT": 60,
}
RANGE_METRICS = (
    "TIR_70_180",
    "TBR_below_70",
    "TBR_below_54",
    "TAR_above_180",
    "TAR_above_250",
)


def read_frame(path, required=(), numeric=(), unique=()):
    """Read CSV or Parquet without evaluating serialized Python objects."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(
            path, dtype={c: "string" for c in (*SUBJECT, "window_id", "episode_id")}
        )
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError("Input must be a .csv or .parquet table.")
    # Keep observed values without category metadata creating empty groups.
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            frame[col] = frame[col].astype(object)
    require_columns(frame, required)
    if frame.empty:
        raise ValueError(f"Input table is empty: {path}")
    for col in (*SUBJECT, "window_id", "episode_id"):
        if col in frame:
            if (
                frame[col].isna().any()
                or frame[col].astype(str).str.strip().eq("").any()
            ):
                raise ValueError(f"Missing identifier: {col}")
            frame[col] = frame[col].astype(str)
    if numeric:
        numeric = list(dict.fromkeys(numeric))
        values = frame[numeric].apply(pd.to_numeric, errors="raise")
        valid = np.isfinite(values.to_numpy(dtype=float)).all(axis=0)
        if not valid.all():
            bad = [col for col, finite in zip(numeric, valid) if not finite]
            raise ValueError(f"Nonfinite values in {bad}")
        frame = pd.concat([frame.drop(columns=numeric), values], axis=1)
    if unique:
        require_unique(frame, unique)
    return frame


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame[list(columns)].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    for col in columns:
        if (
            pd.api.types.is_string_dtype(frame[col])
            and frame[col].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"Required column contains blank values: {col}")


def require_unique(frame, keys):
    if frame.duplicated(list(keys)).any():
        raise ValueError(f"Duplicate records for {list(keys)}")


def finite_array(values):
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("A nonempty finite array is required.")
    return result


def dynamics_metrics(reference, modeled, lower=None, upper=None):
    """MAE, RMSE and MBE in mg/dL, with PICP (0–1) and MPIW (mg/dL)."""
    reference, modeled = finite_array(reference), finite_array(modeled)
    if reference.shape != modeled.shape:
        raise ValueError("Reference and modeled CGM shapes differ.")
    error = modeled - reference
    result = {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt(np.square(error).mean())),
        "MBE": float(error.mean()),
    }
    if (lower is None) != (upper is None):
        raise ValueError("Supply both interval bounds.")
    if lower is not None:
        lower, upper = finite_array(lower), finite_array(upper)
        if (
            lower.shape != reference.shape
            or upper.shape != reference.shape
            or np.any(lower > upper)
        ):
            raise ValueError("Invalid uncertainty interval shape or ordering.")
        result.update(
            PICP=float(((reference >= lower) & (reference <= upper)).mean()),
            MPIW=float((upper - lower).mean()),
        )
    return result


def clarke_zones(reference, modeled):
    """Assign Clarke Error Grid zones in precedence order A, E, C, D, B."""
    r, m = finite_array(reference), finite_array(modeled)
    if r.shape != m.shape:
        raise ValueError("Clarke Error Grid input shapes differ.")
    z = np.full(r.shape, "B", dtype="U1")
    a = (np.abs(m - r) <= 0.2 * r) | ((r <= 70) & (m <= 70))
    z[a] = "A"
    e = (~a) & (((r >= 180) & (m <= 70)) | ((r <= 70) & (m >= 180)))
    z[e] = "E"
    c = (~a & ~e) & (
        ((r >= 70) & (r <= 290) & (m >= r + 110))
        | ((r >= 130) & (r <= 180) & (m <= 1.4 * r - 182))
    )
    z[c] = "C"
    d = (~a & ~e & ~c) & (
        ((r >= 240) & (m >= 70) & (m <= 180))
        | ((r <= 175 / 3) & (m >= 70) & (m <= 180))
        | ((r >= 175 / 3) & (r <= 70) & (m >= 1.2 * r))
    )
    z[d] = "D"
    return z


def glycemic_metrics(values):
    """TIR, TBR, and TAR percentages used in Fig. 6a,d."""
    cgm = finite_array(values)
    masks = ((cgm >= 70) & (cgm <= 180), cgm < 70, cgm < 54, cgm > 180, cgm > 250)
    return {name: float(100 * mask.mean()) for name, mask in zip(RANGE_METRICS, masks)}


def subject_summary(frame, groups, values, ddof=1):
    """Summarize one value per subject and group with equal subject weights."""
    require_unique(frame, list(groups) + SUBJECT)
    rows = []
    grouped = (
        frame.groupby(list(groups), sort=True, dropna=False)
        if groups
        else [((), frame)]
    )
    for key, group in grouped:
        key = key if isinstance(key, tuple) else (key,)
        for metric in values:
            array = group[metric].dropna().to_numpy(dtype=float)
            if np.isinf(array).any():
                raise ValueError(f"Infinite {metric} values")
            rows.append(
                {
                    **dict(zip(groups, key)),
                    "metric": metric,
                    "mean": float(array.mean()) if len(array) else np.nan,
                    "std": float(array.std(ddof=ddof)) if len(array) > ddof else np.nan,
                    "n_subjects": len(array),
                }
            )
    return pd.DataFrame(rows)


def require_test_rows(frame):
    """Require test set outcomes when labeled. Unlabeled inputs follow the schema contract."""
    if "split" in frame and (
        frame["split"].isna().any() or set(frame["split"]) != {"Test"}
    ):
        raise ValueError("Outcome evaluation requires test set records only.")


def validate_statistics(frame, allow_empty=False):
    """Validate mean/SD/count triples, including explicitly undefined statistics."""
    if not {"mean", "std", "n_subjects"}.issubset(frame):
        raise ValueError("Summary inputs require mean, std, and n_subjects.")
    for col in ("mean", "std", "n_subjects"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if np.isinf(frame[col]).any():
            raise ValueError(f"Infinite summary values in {col}.")
    n = frame.n_subjects
    if n.isna().any() or (n % 1 != 0).any() or (n < (0 if allow_empty else 1)).any():
        raise ValueError("Invalid contributing-subject count.")
    if (frame["mean"].isna() & (n > 0)).any() or (
        frame["mean"].notna() & (n == 0)
    ).any():
        raise ValueError(
            "Summary means must be finite for nonempty groups and blank for empty groups."
        )
    if (frame["std"] < 0).any() or (frame["std"].isna() & (n > 1)).any():
        raise ValueError("Invalid or missing standard deviation.")
    if ((n == 0) & frame["std"].notna()).any() or (
        (n == 1) & frame["std"].fillna(0).ne(0)
    ).any():
        raise ValueError(
            "An empty group has no SD. A single-subject SD is zero or undefined."
        )
    return frame


def require_metrics(frame, groups, expected):
    grouped = frame.groupby(list(groups), dropna=False) if groups else [((), frame)]
    for key, part in grouped:
        if set(part.metric) != set(expected):
            raise ValueError(
                f"Incomplete metrics for {key}. Expected {sorted(expected)}."
            )
        if set(expected) in ({"MAE", "RMSE"}, {"PICP", "MPIW"}):
            if part.n_subjects.nunique() != 1:
                raise ValueError("Paired metrics require the same subject count.")
            if "MAE" in set(expected):
                values = part.set_index("metric")["mean"]
                if values.RMSE + 1e-8 < values.MAE:
                    raise ValueError(
                        "RMSE cannot be smaller than MAE for the same observations."
                    )


def read_summary(path, groups, allowed_metrics):
    """Read already aggregated values. Never treat their SD as a standard error."""
    frame = read_frame(
        path,
        [*groups, "metric", "mean", "n_subjects"],
        ["mean", "n_subjects"],
        [*groups, "metric"],
    )
    require_test_rows(frame)
    if not set(frame.metric).issubset(allowed_metrics):
        raise ValueError(f"Allowed metrics: {sorted(allowed_metrics)}")
    validate_statistics(frame)
    positive = frame.metric.isin(
        ["MAE", "RMSE", "MPIW", "cho_l2", "ins_l2", "hypoglycemia_duration_minutes"]
    )
    if (frame.loc[positive, "mean"] < 0).any():
        raise ValueError(
            "Error magnitudes, widths, norms, and durations cannot be negative."
        )
    if not frame.loc[frame.metric == "PICP", "mean"].between(0, 1).all():
        raise ValueError("PICP must be a probability in [0,1], as in Fig. 4b.")
    return frame


def check_time_grid(frame, keys, end_by_model=None, end=None, start=5):
    """Require complete 5-minute trajectories, never extrapolate short horizons."""
    require_unique(frame, [*keys, "minute"])
    for key, group in frame.groupby(list(keys), sort=False, dropna=False):
        stop = end
        if end_by_model is not None:
            model = group.model.iloc[0]
            if model not in end_by_model:
                raise ValueError(f"Unknown model: {model}")
            stop = end_by_model[model]
        expected = np.arange(start, stop + 1, 5)
        if not np.array_equal(np.sort(group.minute.to_numpy()), expected):
            raise ValueError(
                f"Incomplete or invalid time grid for {key}. Expected {start}..{stop} by 5."
            )


def require_paired_models(frame, context=(), identical_windows=True):
    """Check reference CGM consistency and window pairing across models."""
    grouped = frame.groupby(list(context), dropna=False) if context else [((), frame)]
    for _, group in grouped:
        sets = [
            set(map(tuple, g[WINDOW].drop_duplicates().to_numpy()))
            for _, g in group.groupby("model")
        ]
        if identical_windows and any(s != sets[0] for s in sets[1:]):
            raise ValueError("Models must be evaluated on identical windows.")
        if "ground_truth_cgm" in group:
            span = group.groupby([*WINDOW, "minute"]).ground_truth_cgm.agg(
                ["min", "max"]
            )
            if not np.allclose(span["min"], span["max"], rtol=0, atol=1e-6):
                raise ValueError("Paired models have different ground truth CGM.")


def write_csv(frame, output, name):
    path = Path(output) / f"{name}.csv"
    frame.to_csv(path, index=False, float_format="%.10g")
    return path
