"""Convert DiaTrend source records to the canonical schema."""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_COLUMNS = ["date", "CGM", "is_interpolated", "Basal", "Bolus", "CHO", "INS"]
GRID_FREQ = "5min"
MAX_CONSECUTIVE_MISSING_CGM = 4
FIVE_MINUTES = pd.Timedelta(minutes=5)
SUBJECT_OFFSET = 300


def parse_float(value: object) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


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


def discover_excel_files(input_dir: Path) -> list[Path]:
    excel_paths = sorted(input_dir.glob("Subject*.xlsx"), key=subject_sort_key)
    if not excel_paths:
        raise FileNotFoundError(f"No DiaTrend Excel files found under {input_dir}")
    return excel_paths


def subject_sort_key(path: Path) -> tuple[int, str]:
    subject_id = extract_subject_id(path)
    return (subject_id, path.as_posix())


def extract_subject_id(path: Path) -> int:
    match = re.search("Subject(\\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot extract subject id from {path}")
    return int(match.group(1))


def require_columns(
    df: pd.DataFrame, required_columns: list[str], source: Path, sheet: str
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} sheet {sheet} is missing columns: {missing}")


def load_cgm_events(source: Path) -> pd.DataFrame:
    cgm_df = pd.read_excel(source, sheet_name="CGM")
    require_columns(cgm_df, ["date", "mg/dl"], source, "CGM")
    cgm_df = cgm_df.copy()
    cgm_df["timestamp"] = pd.to_datetime(cgm_df["date"], errors="coerce")
    cgm_df["CGM"] = pd.to_numeric(cgm_df["mg/dl"], errors="coerce")
    cgm_df = cgm_df.dropna(subset=["timestamp", "CGM"]).reset_index(drop=True)
    cgm_df["slot"] = cgm_df["timestamp"].dt.round(GRID_FREQ)
    cgm_df["source_order"] = np.arange(len(cgm_df), dtype=np.int64)
    return cgm_df[["timestamp", "slot", "CGM", "source_order"]]


def load_bolus_events(source: Path) -> pd.DataFrame:
    bolus_df = pd.read_excel(source, sheet_name="Bolus")
    require_columns(bolus_df, ["date", "normal", "carbInput"], source, "Bolus")
    bolus_df = bolus_df.copy()
    bolus_df["timestamp"] = pd.to_datetime(bolus_df["date"], errors="coerce")
    bolus_df["Bolus"] = pd.to_numeric(bolus_df["normal"], errors="coerce").fillna(0.0)
    bolus_df["CHO"] = pd.to_numeric(bolus_df["carbInput"], errors="coerce").fillna(0.0)
    bolus_df = bolus_df.dropna(subset=["timestamp"]).reset_index(drop=True)
    bolus_df["slot"] = bolus_df["timestamp"].dt.round(GRID_FREQ)
    bolus_df["source_order"] = np.arange(len(bolus_df), dtype=np.int64)
    return bolus_df[["timestamp", "slot", "Bolus", "CHO", "source_order"]]


def load_basal_events(source: Path) -> pd.DataFrame:
    basal_df = pd.read_excel(source, sheet_name="Basal")
    require_columns(basal_df, ["date", "duration", "rate"], source, "Basal")
    basal_df = basal_df.copy()
    basal_df["start"] = pd.to_datetime(basal_df["date"], errors="coerce")
    basal_df["duration_ms"] = pd.to_numeric(basal_df["duration"], errors="coerce")
    basal_df["rate"] = pd.to_numeric(basal_df["rate"], errors="coerce").fillna(0.0)
    basal_df = basal_df.dropna(subset=["start", "duration_ms"]).reset_index(drop=True)
    basal_df = basal_df[basal_df["duration_ms"] > 0].reset_index(drop=True)
    basal_df["end"] = basal_df["start"] + pd.to_timedelta(
        basal_df["duration_ms"], unit="ms"
    )
    basal_df["source_order"] = np.arange(len(basal_df), dtype=np.int64)
    return basal_df[["start", "end", "rate", "duration_ms", "source_order"]]


def build_cgm_series(cgm_df: pd.DataFrame) -> pd.Series:
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
    date_index: pd.DatetimeIndex, basal_df: pd.DataFrame
) -> np.ndarray:
    if len(date_index) == 0 or basal_df.empty:
        return np.zeros(len(date_index), dtype=np.float64)
    sorted_dates = pd.DatetimeIndex(date_index).sort_values()
    basal_rate = np.zeros(len(sorted_dates), dtype=np.float64)
    for row in basal_df.sort_values(["start", "source_order"]).itertuples(index=False):
        mask = (sorted_dates >= row.start) & (sorted_dates < row.end)
        if mask.any():
            basal_rate[np.asarray(mask, dtype=bool)] = float(row.rate)
    return (basal_rate / 12.0).astype(np.float64)


def build_bolus_series(bolus_df: pd.DataFrame) -> pd.Series:
    if bolus_df.empty:
        return pd.Series(dtype=np.float64)
    return bolus_df.groupby("slot")["Bolus"].sum().astype(np.float64).sort_index()


def build_cho_series(bolus_df: pd.DataFrame) -> pd.Series:
    if bolus_df.empty:
        return pd.Series(dtype=np.float64)
    return bolus_df.groupby("slot")["CHO"].sum().astype(np.float64).sort_index()


def convert_subject(source: Path, output_dir: Path) -> Path:
    source_subject_id = extract_subject_id(source)
    output_subject_id = source_subject_id + SUBJECT_OFFSET
    cgm_events = load_cgm_events(source)
    bolus_events = load_bolus_events(source)
    basal_events = load_basal_events(source)
    cgm_series = build_cgm_series(cgm_events)
    cgm_frame, candidate_days, dropped_days = build_cgm_frame(cgm_series)
    if cgm_frame.empty:
        output_df = pd.DataFrame(columns=TARGET_COLUMNS)
    else:
        date_index = pd.DatetimeIndex(cgm_frame["date"])
        basal_values = build_basal_values(date_index, basal_events)
        bolus_series = build_bolus_series(bolus_events)
        cho_series = build_cho_series(bolus_events)
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

    return [convert_subject(p, output_dir) for p in discover_excel_files(input_dir)]
