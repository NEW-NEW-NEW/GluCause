"""Convert AZT1D source records to the canonical schema."""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_COLUMNS = ["date", "CGM", "is_interpolated", "Basal", "Bolus", "CHO", "INS"]
GRID_FREQ = "5min"
MAX_CONSECUTIVE_MISSING_CGM = 4
SUBJECT_ID_OFFSET = 100


def subject_number_from_path(path: Path) -> int:
    match = re.search("Subject\\s+(\\d+)", str(path))
    if match is None:
        raise ValueError(f"Unable to parse subject number from {path}")
    return int(match.group(1))


def discover_input_files(input_dir: Path) -> list[Path]:
    source_dir = input_dir
    if (source_dir / "CGM Records").is_dir():
        source_dir = source_dir / "CGM Records"
    csv_paths = sorted(
        source_dir.rglob("Subject *.csv"), key=lambda p: subject_number_from_path(p)
    )
    if not csv_paths:
        raise FileNotFoundError(f"No AZT1D subject CSV files found in {source_dir}")
    return csv_paths


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


def find_cgm_column(df: pd.DataFrame) -> str:
    if "CGM" in df.columns:
        return "CGM"
    if "Readings (CGM / BGM)" in df.columns:
        return "Readings (CGM / BGM)"
    raise ValueError("Missing CGM column: expected 'CGM' or 'Readings (CGM / BGM)'")


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "EventDateTime" not in df.columns:
        raise ValueError(f"{path} is missing required column: EventDateTime")
    cgm_column = find_cgm_column(df)
    source = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["EventDateTime"], errors="coerce"),
            "CGM": pd.to_numeric(df[cgm_column], errors="coerce"),
            "Basal": pd.to_numeric(df.get("Basal"), errors="coerce"),
            "Bolus": pd.to_numeric(
                df.get("TotalBolusInsulinDelivered"), errors="coerce"
            ),
            "CHO": pd.to_numeric(df.get("CarbSize"), errors="coerce"),
        }
    )
    source = source.dropna(subset=["timestamp"]).copy()
    duplicate_subset = ["timestamp", "CGM", "Basal", "Bolus", "CHO"]
    duplicate_rows_removed = int(
        source.duplicated(subset=duplicate_subset, keep="last").sum()
    )
    source = source.drop_duplicates(subset=duplicate_subset, keep="last")
    source["slot"] = source["timestamp"].dt.round(GRID_FREQ)
    source = source.dropna(subset=["slot"]).sort_values(["slot", "timestamp"])
    source.attrs["input_rows"] = len(df)
    source.attrs["duplicate_rows_removed"] = duplicate_rows_removed
    return source


def convert_basal_to_interval_amounts(basal: pd.Series) -> tuple[pd.Series, int]:
    basal_rate = pd.to_numeric(basal, errors="coerce").astype(float)
    integer_like = np.isclose(basal_rate, np.round(basal_rate), equal_nan=False)
    decimal_fix_mask = (basal_rate > 20.0) | (basal_rate >= 9.0) & integer_like
    corrected_rate = basal_rate.copy()
    corrected_rate.loc[decimal_fix_mask] = corrected_rate.loc[decimal_fix_mask] / 1000.0
    basal_interval_units = corrected_rate / 12.0
    return (basal_interval_units, int(decimal_fix_mask.sum()))


def aggregate_to_grid(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, int]:
    cgm_rows = df.loc[df["CGM"].notna(), ["slot", "timestamp", "CGM"]]
    cgm_series = (
        cgm_rows.sort_values(["slot", "timestamp"])
        .groupby("slot", as_index=False)
        .tail(1)
        .set_index("slot")["CGM"]
        .sort_index()
    )
    basal_rows = df.loc[df["Basal"].notna(), ["slot", "timestamp", "Basal"]].copy()
    basal_decimal_fixed_rows = 0
    if basal_rows.empty:
        basal_series = pd.Series(dtype=float)
    else:
        basal_rows["Basal"], basal_decimal_fixed_rows = (
            convert_basal_to_interval_amounts(basal_rows["Basal"])
        )
        basal_series = (
            basal_rows.sort_values(["slot", "timestamp"])
            .groupby("slot", as_index=False)
            .tail(1)
            .set_index("slot")["Basal"]
            .sort_index()
        )
    bolus_series = (
        df.loc[df["Bolus"].notna()].groupby("slot")["Bolus"].sum().sort_index()
    )
    cho_series = df.loc[df["CHO"].notna()].groupby("slot")["CHO"].sum().sort_index()
    return (
        cgm_series,
        basal_series,
        bolus_series,
        cho_series,
        basal_decimal_fixed_rows,
    )


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


def align_basal_to_output(
    date_index: pd.DatetimeIndex, basal_series: pd.Series
) -> tuple[np.ndarray, int]:
    if basal_series.empty or len(date_index) == 0:
        return (np.zeros(len(date_index), dtype=np.float64), 0)
    target = pd.DataFrame({"date": pd.DatetimeIndex(date_index).sort_values()})
    source = (
        basal_series.reset_index()
        .rename(columns={"slot": "date", basal_series.name or 0: "Basal"})
        .sort_values("date")
    )
    if "date" not in source.columns:
        source = source.rename(columns={source.columns[0]: "date"})
    if "Basal" not in source.columns:
        source = source.rename(columns={source.columns[-1]: "Basal"})
    aligned = pd.merge_asof(
        target,
        source[["date", "Basal"]],
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(minutes=20),
    )
    exact_basal = basal_series.reindex(target["date"])
    carried_mask = aligned["Basal"].notna() & exact_basal.isna().to_numpy()
    return (
        aligned["Basal"].fillna(0.0).to_numpy(dtype=np.float64),
        int(carried_mask.sum()),
    )


def convert_file(input_path: Path, output_path: Path) -> Path:
    source_df = load_source(input_path)
    cgm_series, basal_series, bolus_series, cho_series, basal_decimal_fixed_rows = (
        aggregate_to_grid(source_df)
    )
    basal_carried_rows = 0
    if cgm_series.empty:
        output_df = pd.DataFrame(columns=TARGET_COLUMNS)
        dropped_days = 0
    else:
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
            output_df = pd.concat(output_parts, ignore_index=True)
        else:
            output_df = pd.DataFrame(columns=["date", "CGM", "is_interpolated"])
        date_index = pd.DatetimeIndex(output_df["date"])
        basal_values, basal_carried_rows = align_basal_to_output(
            date_index, basal_series
        )
        output_df["Basal"] = basal_values
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, float_format="%.10g")
    return output_path


def convert(input_dir: Path, output_dir: Path) -> list[Path]:
    """Write canonical CSV files to a new or empty directory."""
    from .common import require_empty_directory

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    require_empty_directory(output_dir)

    paths = discover_input_files(input_dir)
    return [
        convert_file(
            p,
            output_dir
            / f"New_Subject{subject_number_from_path(p) + SUBJECT_ID_OFFSET}.csv",
        )
        for p in paths
    ]
