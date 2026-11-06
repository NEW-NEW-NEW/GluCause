"""Convert HUPA-UCM source records to the canonical schema."""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_COLUMNS = ["date", "CGM", "is_interpolated", "Basal", "Bolus", "CHO", "INS"]
SOURCE_COLUMNS = [
    "time",
    "glucose",
    "calories",
    "heart_rate",
    "steps",
    "basal_rate",
    "bolus_volume_delivered",
    "carb_input",
]
GRID_FREQ = "5min"
MAX_CONSECUTIVE_MISSING_CGM = 4
SUBJECT_OFFSET = 400


def max_consecutive_true(values: pd.Series) -> int:
    max_run = 0
    current_run = 0
    for value in values.to_numpy(dtype=bool):
        if value:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def extract_subject_id(path: Path) -> int:
    match = re.search("HUPA(\\d+)P", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot extract subject id from {path}")
    return int(match.group(1))


def subject_sort_key(path: Path) -> tuple[int, str]:
    subject_id = extract_subject_id(path)
    return (subject_id, path.as_posix())


def discover_csv_files(input_dir: Path) -> list[Path]:
    csv_paths = sorted(input_dir.glob("HUPA*P.csv"), key=subject_sort_key)
    if not csv_paths:
        raise FileNotFoundError(f"No HUPA-UCM CSV files found under {input_dir}")
    return csv_paths


def require_source_columns(df: pd.DataFrame, source: Path) -> None:
    missing = [column for column in SOURCE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def load_source_frame(source: Path) -> pd.DataFrame:
    source_df = pd.read_csv(source, sep=";")
    require_source_columns(source_df, source)
    frame = source_df.copy()
    frame["timestamp"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["CGM"] = pd.to_numeric(frame["glucose"], errors="coerce")
    frame["Basal"] = pd.to_numeric(frame["basal_rate"], errors="coerce")
    frame["Bolus"] = pd.to_numeric(
        frame["bolus_volume_delivered"], errors="coerce"
    ).fillna(0.0)
    frame["CHO"] = (
        pd.to_numeric(frame["carb_input"], errors="coerce").fillna(0.0) * 10.0
    )
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    frame["slot"] = frame["timestamp"].dt.round(GRID_FREQ)
    frame["source_order"] = np.arange(len(frame), dtype=np.int64)
    return frame[["timestamp", "slot", "CGM", "Basal", "Bolus", "CHO", "source_order"]]


def build_cgm_series(source_df: pd.DataFrame) -> pd.Series:
    cgm_df = source_df.dropna(subset=["CGM"])
    if cgm_df.empty:
        return pd.Series(dtype=np.float64)
    cgm_series = (
        cgm_df.sort_values(["slot", "timestamp", "source_order"])
        .groupby("slot", as_index=False)
        .tail(1)
        .set_index("slot")["CGM"]
        .astype(np.float64)
        .sort_index()
    )
    return cgm_series


def interpolate_day(day_cgm: pd.Series) -> tuple[pd.DataFrame | None, bool]:
    grid = pd.date_range(day_cgm.index.min(), day_cgm.index.max(), freq=GRID_FREQ)
    cgm_on_grid = day_cgm.reindex(grid)
    missing_mask = cgm_on_grid.isna()
    if max_consecutive_true(missing_mask) > MAX_CONSECUTIVE_MISSING_CGM:
        return (None, True)
    if missing_mask.any():
        if cgm_on_grid.notna().sum() < 4:
            return (None, True)
        filled_cgm = cgm_on_grid.interpolate(method="cubic", limit_area="inside")
    else:
        filled_cgm = cgm_on_grid
    if filled_cgm.isna().any():
        return (None, True)
    return (
        pd.DataFrame(
            {
                "date": grid,
                "CGM": filled_cgm.to_numpy(dtype=np.float64),
                "is_interpolated": missing_mask.to_numpy(dtype=bool),
            }
        ),
        False,
    )


def build_cgm_frame(cgm_series: pd.Series) -> tuple[pd.DataFrame, int, int]:
    if cgm_series.empty:
        return (pd.DataFrame(columns=["date", "CGM", "is_interpolated"]), 0, 0)
    output_parts: list[pd.DataFrame] = []
    dropped_days = 0
    days = sorted(cgm_series.index.normalize().unique())
    for day in days:
        day_cgm = cgm_series[cgm_series.index.normalize() == day]
        day_frame, dropped = interpolate_day(day_cgm)
        if dropped:
            dropped_days += 1
            continue
        output_parts.append(day_frame)
    if output_parts:
        return (pd.concat(output_parts, ignore_index=True), len(days), dropped_days)
    return (
        pd.DataFrame(columns=["date", "CGM", "is_interpolated"]),
        len(days),
        dropped_days,
    )


def build_basal_values(
    date_index: pd.DatetimeIndex, source_df: pd.DataFrame
) -> np.ndarray:
    if len(date_index) == 0:
        return np.zeros(0, dtype=np.float64)
    date_frame = pd.DataFrame({"date": pd.DatetimeIndex(date_index).sort_values()})
    basal_source = (
        source_df.dropna(subset=["Basal"])
        .sort_values(["slot", "timestamp", "source_order"])
        .groupby("slot", as_index=False)
        .tail(1)
        .sort_values("slot")
        .rename(columns={"slot": "date"})
    )
    if basal_source.empty:
        return np.zeros(len(date_frame), dtype=np.float64)
    aligned = pd.merge_asof(
        date_frame, basal_source[["date", "Basal"]], on="date", direction="backward"
    )
    return aligned["Basal"].fillna(0.0).to_numpy(dtype=np.float64)


def build_bolus_series(source_df: pd.DataFrame) -> pd.Series:
    if source_df.empty:
        return pd.Series(dtype=np.float64)
    return source_df.groupby("slot")["Bolus"].sum().astype(np.float64).sort_index()


def build_cho_series(source_df: pd.DataFrame) -> pd.Series:
    if source_df.empty:
        return pd.Series(dtype=np.float64)
    return source_df.groupby("slot")["CHO"].sum().astype(np.float64).sort_index()


def convert_subject(source: Path, output_dir: Path) -> Path:
    source_subject_id = extract_subject_id(source)
    output_subject_id = source_subject_id + SUBJECT_OFFSET
    source_df = load_source_frame(source)
    cgm_series = build_cgm_series(source_df)
    cgm_frame, candidate_days, dropped_days = build_cgm_frame(cgm_series)
    if cgm_frame.empty:
        output_df = pd.DataFrame(columns=TARGET_COLUMNS)
    else:
        date_index = pd.DatetimeIndex(cgm_frame["date"])
        basal_values = build_basal_values(date_index, source_df)
        bolus_series = build_bolus_series(source_df)
        cho_series = build_cho_series(source_df)
        output_df = cgm_frame.copy()
        output_df["Basal"] = basal_values
        output_df["Bolus"] = (
            bolus_series.reindex(date_index).fillna(0.0).to_numpy(dtype=np.float64)
        )
        output_df["CHO"] = (
            cho_series.reindex(date_index).fillna(0.0).to_numpy(dtype=np.float64)
        )
        output_df["INS"] = output_df["Basal"] + output_df["Bolus"]
        output_df = output_df[TARGET_COLUMNS].sort_values("date").reset_index(drop=True)
    output_path = output_dir / f"New_Subject{output_subject_id}.csv"
    output_to_write = output_df.copy()
    if not output_to_write.empty:
        output_to_write["date"] = pd.to_datetime(output_to_write["date"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        output_to_write["is_interpolated"] = np.where(
            output_to_write["is_interpolated"], "TRUE", "FALSE"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_to_write.to_csv(output_path, index=False, float_format="%.10g")
    return output_path


def convert(input_dir: Path, output_dir: Path) -> list[Path]:
    """Write canonical CSV files to a new or empty directory."""
    from .common import require_empty_directory

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    require_empty_directory(output_dir)

    return [convert_subject(p, output_dir) for p in discover_csv_files(input_dir)]
