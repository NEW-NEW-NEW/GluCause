"""Convert T1D-UOM source records to the canonical schema."""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_COLUMNS = ["date", "CGM", "is_interpolated", "Basal", "Bolus", "CHO", "INS"]
GRID_FREQ = "5min"
MMOL_L_TO_MG_DL = 18.0
MAX_CONSECUTIVE_MISSING_CGM = 4
LONG_ACTING_DURATION = pd.Timedelta(hours=24)
RAPID_BASAL_CARRY_LIMIT = pd.Timedelta(hours=24)


def participant_id_from_path(path: Path) -> str:
    match = re.search("(\\d{4})", path.name)
    if match is None:
        raise ValueError(f"Unable to parse participant ID from {path}")
    return match.group(1)


def parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


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


def discover_participant_ids(input_dir: Path) -> list[str]:
    glucose_dir = input_dir / "Glucose Data"
    glucose_paths = sorted(glucose_dir.glob("UoMGlucose*.csv"))
    if not glucose_paths:
        raise FileNotFoundError(f"No T1D-UOM glucose CSV files found in {glucose_dir}")
    return [participant_id_from_path(path) for path in glucose_paths]


def load_glucose(input_dir: Path, participant_id: str) -> pd.Series:
    path = input_dir / "Glucose Data" / f"UoMGlucose{participant_id}.csv"
    df = pd.read_csv(path, low_memory=False)
    missing_columns = [
        column for column in ["bg_ts", "value"] if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    source = pd.DataFrame(
        {
            "timestamp": parse_datetime(df["bg_ts"]),
            "CGM": pd.to_numeric(df["value"], errors="coerce") * MMOL_L_TO_MG_DL,
        }
    ).dropna(subset=["timestamp", "CGM"])
    source["slot"] = source["timestamp"].dt.round(GRID_FREQ)
    source = source.dropna(subset=["slot"]).sort_values(["slot", "timestamp"])
    cgm_series = (
        source.groupby("slot", as_index=False)
        .tail(1)
        .set_index("slot")["CGM"]
        .sort_index()
    )
    cgm_series.attrs["glucose_rows"] = len(df)
    return cgm_series


def load_event_series(
    path: Path, timestamp_column: str, value_column: str
) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, low_memory=False)
    if timestamp_column not in df.columns or value_column not in df.columns:
        raise ValueError(f"{path} is missing {timestamp_column!r} or {value_column!r}")
    source = pd.DataFrame(
        {
            "timestamp": parse_datetime(df[timestamp_column]),
            "value": pd.to_numeric(df[value_column], errors="coerce"),
        }
    ).dropna(subset=["timestamp", "value"])
    if source.empty:
        return pd.Series(dtype=float)
    source["slot"] = source["timestamp"].dt.round(GRID_FREQ)
    return source.groupby("slot")["value"].sum().sort_index()


def load_cho(input_dir: Path, participant_id: str) -> pd.Series:
    path = input_dir / "Nutrition Data" / f"UoMNutrition{participant_id}.csv"
    return load_event_series(path, "meal_ts", "carbs_g")


def load_bolus(input_dir: Path, participant_id: str) -> pd.Series:
    path = input_dir / "Insulin Data" / "Bolus Data" / f"UoMBolus{participant_id}.csv"
    return load_event_series(path, "bolus_ts", "bolus_dose")


def load_basal(
    input_dir: Path, participant_id: str
) -> tuple[pd.Series, pd.Series, int, int]:
    path = input_dir / "Insulin Data" / "Basal Data" / f"UoMBasal{participant_id}.csv"
    if not path.exists():
        return (pd.Series(dtype=float), pd.Series(dtype=float), 0, 0)
    df = pd.read_csv(path, low_memory=False)
    missing_columns = [
        column
        for column in ["basal_ts", "basal_dose", "insulin_kind"]
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")
    source = pd.DataFrame(
        {
            "timestamp": parse_datetime(df["basal_ts"]),
            "basal_dose": pd.to_numeric(df["basal_dose"], errors="coerce"),
            "insulin_kind": df["insulin_kind"].astype(str).str.upper().str.strip(),
        }
    ).dropna(subset=["timestamp", "basal_dose"])
    if source.empty:
        return (pd.Series(dtype=float), pd.Series(dtype=float), 0, 0)
    source["slot"] = source["timestamp"].dt.round(GRID_FREQ)
    rapid_rows = source[source["insulin_kind"].eq("R")].sort_values(
        ["slot", "timestamp"]
    )
    if rapid_rows.empty:
        rapid_rate = pd.Series(dtype=float)
    else:
        rapid_rate = (
            rapid_rows.groupby("slot", as_index=False)
            .tail(1)
            .set_index("slot")["basal_dose"]
            .sort_index()
            / 12.0
        )
    long_rows = source[source["insulin_kind"].eq("L")].sort_values("slot")
    if long_rows.empty:
        long_events = pd.Series(dtype=float)
    else:
        long_events = long_rows.groupby("slot")["basal_dose"].sum().sort_index()
    return (rapid_rate, long_events, len(rapid_rows), len(long_rows))


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
    if not output_parts:
        return (
            pd.DataFrame(columns=["date", "CGM", "is_interpolated"]),
            len(days),
            dropped_days,
        )
    return (pd.concat(output_parts, ignore_index=True), len(days), dropped_days)


def align_rapid_basal(
    date_index: pd.DatetimeIndex, rapid_rate: pd.Series
) -> np.ndarray:
    if rapid_rate.empty or len(date_index) == 0:
        return np.zeros(len(date_index), dtype=np.float64)
    target = pd.DataFrame({"date": pd.DatetimeIndex(date_index).sort_values()})
    source = (
        rapid_rate.rename("rapid_basal").reset_index().rename(columns={"slot": "date"})
    )
    if "date" not in source.columns:
        source = source.rename(columns={source.columns[0]: "date"})
    aligned = pd.merge_asof(
        target,
        source[["date", "rapid_basal"]].sort_values("date"),
        on="date",
        direction="backward",
        tolerance=RAPID_BASAL_CARRY_LIMIT,
    )
    return aligned["rapid_basal"].fillna(0.0).to_numpy(dtype=np.float64)


def align_long_basal(
    date_index: pd.DatetimeIndex, long_events: pd.Series
) -> np.ndarray:
    if long_events.empty or len(date_index) == 0:
        return np.zeros(len(date_index), dtype=np.float64)
    basal = np.zeros(len(date_index), dtype=np.float64)
    date_values = pd.DatetimeIndex(date_index)
    slots_per_duration = int(LONG_ACTING_DURATION / pd.Timedelta(GRID_FREQ))
    for event_time, dose in long_events.items():
        mask = (date_values >= event_time) & (
            date_values < event_time + LONG_ACTING_DURATION
        )
        if mask.any():
            basal[mask] += float(dose) / slots_per_duration
    return basal


def convert_participant(input_dir: Path, output_dir: Path, participant_id: str) -> Path:
    cgm_series = load_glucose(input_dir, participant_id)
    output_df, candidate_days, dropped_days = build_cgm_frame(cgm_series)
    rapid_basal, long_basal, rapid_basal_rows, long_basal_events = load_basal(
        input_dir, participant_id
    )
    bolus_series = load_bolus(input_dir, participant_id)
    cho_series = load_cho(input_dir, participant_id)
    if output_df.empty:
        output_df = pd.DataFrame(columns=TARGET_COLUMNS)
    else:
        date_index = pd.DatetimeIndex(output_df["date"])
        rapid_basal_values = align_rapid_basal(date_index, rapid_basal)
        long_basal_values = align_long_basal(date_index, long_basal)
        output_df["Basal"] = rapid_basal_values + long_basal_values
        output_df["Bolus"] = (
            bolus_series.reindex(date_index).fillna(0.0).to_numpy(dtype=np.float64)
        )
        output_df["CHO"] = (
            cho_series.reindex(date_index).fillna(0.0).to_numpy(dtype=np.float64)
        )
        output_df["INS"] = output_df["Basal"] + output_df["Bolus"]
        output_df = output_df[TARGET_COLUMNS]
    if not output_df.empty:
        output_df["date"] = pd.to_datetime(output_df["date"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        output_df["is_interpolated"] = np.where(
            output_df["is_interpolated"], "TRUE", "FALSE"
        )
    output_path = output_dir / f"New_Subject{participant_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, float_format="%.10g")
    return output_path


def convert(input_dir: Path, output_dir: Path) -> list[Path]:
    """Write canonical CSV files to a new or empty directory."""
    from .common import require_empty_directory

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    require_empty_directory(output_dir)

    return [
        convert_participant(input_dir, output_dir, s)
        for s in discover_participant_ids(input_dir)
    ]
