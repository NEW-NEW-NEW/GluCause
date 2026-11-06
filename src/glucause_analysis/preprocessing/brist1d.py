"""Convert BrisT1D source records to the canonical schema."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_COLUMNS = ["date", "CGM", "is_interpolated", "Basal", "Bolus", "CHO", "INS"]
SOURCE_COLUMNS = ["timestamp", "bg", "insulin", "carbs"]
GRID_FREQ = "5min"
BG_TO_CGM_FACTOR = 18.0
MAX_CONSECUTIVE_MISSING_CGM = 4
INSULIN_BOLUS_THRESHOLD = 0.5
PUMP_BOLUS_INCREMENT = 0.1


def parse_subject_id(input_path: Path) -> str:
    """Convert a BrisT1D file stem such as P01 to subject ID 201."""
    stem = input_path.stem
    if len(stem) < 2 or not stem.startswith("P") or (not stem[1:].isdigit()):
        raise ValueError(
            f"Expected a BrisT1D file name like P01.csv, got {input_path.name}"
        )
    return str(200 + int(stem[1:]))


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


def split_total_insulin(total_insulin: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Derive Basal and Bolus while preserving total insulin in INS."""
    total = total_insulin.fillna(0.0).to_numpy(dtype=np.float64)
    basal = total.copy()
    bolus = np.zeros_like(total)
    high_dose_mask = total > INSULIN_BOLUS_THRESHOLD
    if high_dose_mask.any():
        high_total = total[high_dose_mask]
        high_bolus = (
            np.floor((high_total + 1e-09) / PUMP_BOLUS_INCREMENT) * PUMP_BOLUS_INCREMENT
        )
        high_basal = high_total - high_bolus
        high_basal = np.where(high_basal < 1e-10, 0.0, high_basal)
        basal[high_dose_mask] = high_basal
        bolus[high_dose_mask] = high_bolus
    return (basal, bolus)


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing_columns = [column for column in SOURCE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    for column in ["bg", "insulin", "carbs"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["slot"] = df["timestamp"].dt.round(GRID_FREQ)
    df = df.dropna(subset=["slot"]).sort_values(["slot", "timestamp"])
    return df


def aggregate_to_grid(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    bg_rows = df.loc[df["bg"].notna(), ["slot", "timestamp", "bg"]]
    bg_series = (
        bg_rows.sort_values(["slot", "timestamp"])
        .groupby("slot", as_index=False)
        .tail(1)
        .set_index("slot")["bg"]
        .sort_index()
        * BG_TO_CGM_FACTOR
    )
    cho_series = df.loc[df["carbs"].notna()].groupby("slot")["carbs"].sum().sort_index()
    insulin_series = (
        df.loc[df["insulin"].notna()].groupby("slot")["insulin"].sum().sort_index()
    )
    return (bg_series, cho_series, insulin_series)


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
    day_df = pd.DataFrame(
        {
            "date": grid,
            "CGM": filled_cgm.to_numpy(dtype=np.float64),
            "is_interpolated": missing_mask.to_numpy(dtype=bool),
        }
    )
    return (day_df, False)


def convert_file(input_path: Path, output_path: Path, subject_id: str) -> Path:
    source_df = load_source(input_path)
    bg_series, cho_series, insulin_series = aggregate_to_grid(source_df)
    output_parts: list[pd.DataFrame] = []
    dropped_days = 0
    for day in sorted(bg_series.index.normalize().unique()):
        day_cgm = bg_series[bg_series.index.normalize() == day]
        day_frame, dropped = interpolate_day(day_cgm)
        if dropped:
            dropped_days += 1
            continue
        output_parts.append(day_frame)
    if output_parts:
        output_df = pd.concat(output_parts, ignore_index=True)
    else:
        output_df = pd.DataFrame(columns=["date", "CGM", "is_interpolated"])
    date_index = pd.DatetimeIndex(output_df["date"])
    cho = cho_series.reindex(date_index).fillna(0.0).to_numpy(dtype=np.float64)
    insulin = insulin_series.reindex(date_index).fillna(0.0)
    basal, bolus = split_total_insulin(insulin)
    output_df["Basal"] = basal
    output_df["Bolus"] = bolus
    output_df["CHO"] = cho
    output_df["INS"] = insulin.to_numpy(dtype=np.float64)
    output_df = output_df[TARGET_COLUMNS]
    output_df["date"] = pd.to_datetime(output_df["date"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    output_df["is_interpolated"] = np.where(
        output_df["is_interpolated"], "TRUE", "FALSE"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, float_format="%.10g")
    return output_path


def convert(input_dir: Path, output_dir: Path) -> list[Path]:
    """Write canonical CSV files to a new or empty directory."""
    from .common import require_empty_directory

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    require_empty_directory(output_dir)

    paths = sorted(input_dir.glob("P*.csv"), key=lambda p: int(p.stem[1:]))
    if not paths:
        raise FileNotFoundError(f"No P*.csv files in {input_dir}")
    return [
        convert_file(
            p, output_dir / f"New_Subject{parse_subject_id(p)}.csv", parse_subject_id(p)
        )
        for p in paths
    ]
