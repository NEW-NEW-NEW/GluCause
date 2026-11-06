"""Convert OhioT1DM source records to the canonical schema."""

from __future__ import annotations
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd

TARGET_COLUMNS = ["date", "CGM", "is_interpolated", "Basal", "Bolus", "CHO", "INS"]
GRID_FREQ = "5min"
OHIO_TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S"
MAX_CONSECUTIVE_MISSING_CGM = 4
FIVE_MINUTES = pd.Timedelta(minutes=5)


def parse_ohio_timestamp(value: str | None) -> pd.Timestamp:
    if value is None or value == "":
        return pd.NaT
    return pd.to_datetime(value, format=OHIO_TIMESTAMP_FORMAT, errors="coerce")


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
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


def discover_xml_files(input_root: Path) -> list[Path]:
    raw_dirs = [
        input_root / "2018" / "train" / "0_RAW",
        input_root / "2018" / "test" / "0_RAW",
        input_root / "2020" / "train" / "0_RAW",
        input_root / "2020" / "test" / "0_RAW",
    ]
    missing_dirs = [path for path in raw_dirs if not path.is_dir()]
    if missing_dirs:
        missing = ", ".join((str(path) for path in missing_dirs))
        raise FileNotFoundError(f"Missing OhioT1DM RAW directories: {missing}")
    xml_paths: list[Path] = []
    for raw_dir in raw_dirs:
        xml_paths.extend(sorted(raw_dir.glob("*.xml")))
    if not xml_paths:
        raise FileNotFoundError(f"No OhioT1DM XML files found under {input_root}")
    return xml_paths


def group_xml_files_by_subject(xml_paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for xml_path in xml_paths:
        patient = ET.parse(xml_path).getroot()
        subject_id = str(patient.attrib.get("id", "")).strip()
        if not subject_id:
            raise ValueError(f"Missing patient id in {xml_path}")
        grouped.setdefault(subject_id, []).append(xml_path)
    for subject_id in grouped:
        grouped[subject_id] = sorted(grouped[subject_id], key=get_first_event_timestamp)
    return dict(sorted(grouped.items(), key=lambda item: int(item[0])))


def get_first_event_timestamp(xml_path: Path) -> tuple[pd.Timestamp, str]:
    patient = ET.parse(xml_path).getroot()
    timestamps: list[pd.Timestamp] = []
    for event_group in patient:
        for event in list(event_group):
            timestamp = parse_ohio_timestamp(
                event.attrib.get("ts")
                or event.attrib.get("ts_begin")
                or event.attrib.get("tbegin")
            )
            if not pd.isna(timestamp):
                timestamps.append(timestamp)
    first_timestamp = min(timestamps) if timestamps else pd.Timestamp.max
    return (first_timestamp, xml_path.as_posix())


def load_subject_events(xml_paths: list[Path]) -> dict[str, pd.DataFrame]:
    cgm_rows: list[dict[str, object]] = []
    basal_rows: list[dict[str, object]] = []
    temp_basal_rows: list[dict[str, object]] = []
    bolus_rows: list[dict[str, object]] = []
    meal_rows: list[dict[str, object]] = []
    source_order = 0
    for xml_path in xml_paths:
        patient = ET.parse(xml_path).getroot()
        glucose_level = patient.find("glucose_level")
        for event in list(glucose_level) if glucose_level is not None else []:
            timestamp = parse_ohio_timestamp(event.attrib.get("ts"))
            value = parse_float(event.attrib.get("value"))
            if pd.isna(timestamp) or pd.isna(value):
                continue
            cgm_rows.append(
                {
                    "timestamp": timestamp,
                    "slot": timestamp.round(GRID_FREQ),
                    "CGM": value,
                    "source_order": source_order,
                }
            )
            source_order += 1
        basal = patient.find("basal")
        for event in list(basal) if basal is not None else []:
            timestamp = parse_ohio_timestamp(event.attrib.get("ts"))
            rate = parse_float(event.attrib.get("value"))
            if pd.isna(timestamp) or pd.isna(rate):
                continue
            basal_rows.append(
                {"timestamp": timestamp, "rate": rate, "source_order": source_order}
            )
            source_order += 1
        temp_basal = patient.find("temp_basal")
        for event in list(temp_basal) if temp_basal is not None else []:
            start = parse_ohio_timestamp(event.attrib.get("ts_begin"))
            end = parse_ohio_timestamp(event.attrib.get("ts_end"))
            rate = parse_float(event.attrib.get("value"))
            if pd.isna(start) or pd.isna(end) or pd.isna(rate) or (end <= start):
                continue
            temp_basal_rows.append(
                {"start": start, "end": end, "rate": rate, "source_order": source_order}
            )
            source_order += 1
        bolus = patient.find("bolus")
        for event in list(bolus) if bolus is not None else []:
            start = parse_ohio_timestamp(event.attrib.get("ts_begin"))
            end = parse_ohio_timestamp(event.attrib.get("ts_end"))
            dose = parse_float(event.attrib.get("dose"))
            bolus_type = str(event.attrib.get("type", "")).strip()
            if pd.isna(start) or pd.isna(dose):
                continue
            if pd.isna(end):
                end = start
            bolus_rows.append(
                {
                    "start": start,
                    "end": end,
                    "dose": dose,
                    "type": bolus_type,
                    "source_order": source_order,
                }
            )
            source_order += 1
        meal = patient.find("meal")
        for event in list(meal) if meal is not None else []:
            timestamp = parse_ohio_timestamp(event.attrib.get("ts"))
            carbs = parse_float(event.attrib.get("carbs"))
            if pd.isna(timestamp) or pd.isna(carbs):
                continue
            meal_rows.append(
                {
                    "timestamp": timestamp,
                    "slot": timestamp.round(GRID_FREQ),
                    "CHO": carbs,
                    "source_order": source_order,
                }
            )
            source_order += 1
    return {
        "cgm": pd.DataFrame(cgm_rows),
        "basal": pd.DataFrame(basal_rows),
        "temp_basal": pd.DataFrame(temp_basal_rows),
        "bolus": pd.DataFrame(bolus_rows),
        "meal": pd.DataFrame(meal_rows),
    }


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
    date_index: pd.DatetimeIndex, basal_df: pd.DataFrame, temp_basal_df: pd.DataFrame
) -> np.ndarray:
    if len(date_index) == 0:
        return np.zeros(0, dtype=np.float64)
    date_frame = pd.DataFrame({"date": pd.DatetimeIndex(date_index).sort_values()})
    if basal_df.empty:
        basal_rate = pd.Series(
            np.zeros(len(date_frame), dtype=np.float64), index=date_frame.index
        )
    else:
        basal_source = (
            basal_df.sort_values(["timestamp", "source_order"])
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .rename(columns={"timestamp": "date"})
        )
        aligned = pd.merge_asof(
            date_frame, basal_source[["date", "rate"]], on="date", direction="backward"
        )
        basal_rate = aligned["rate"].fillna(0.0).astype(np.float64)
    if not temp_basal_df.empty:
        grid_dates = date_frame["date"]
        for row in temp_basal_df.sort_values(["start", "source_order"]).itertuples(
            index=False
        ):
            mask = (grid_dates >= row.start) & (grid_dates < row.end)
            if mask.any():
                basal_rate.loc[mask] = float(row.rate)
    return (basal_rate.to_numpy(dtype=np.float64) / 12.0).astype(np.float64)


def add_extended_bolus(
    slots_to_amount: dict[pd.Timestamp, float],
    start: pd.Timestamp,
    end: pd.Timestamp,
    dose: float,
) -> None:
    duration_seconds = (end - start).total_seconds()
    if duration_seconds <= 0:
        slots_to_amount[start.round(GRID_FREQ)] = (
            slots_to_amount.get(start.round(GRID_FREQ), 0.0) + dose
        )
        return
    first_slot = start.floor(GRID_FREQ)
    last_slot = (end - pd.Timedelta(nanoseconds=1)).floor(GRID_FREQ)
    for slot in pd.date_range(first_slot, last_slot, freq=GRID_FREQ):
        slot_end = slot + FIVE_MINUTES
        overlap_start = max(start, slot)
        overlap_end = min(end, slot_end)
        overlap_seconds = (overlap_end - overlap_start).total_seconds()
        if overlap_seconds <= 0:
            continue
        amount = dose * (overlap_seconds / duration_seconds)
        slots_to_amount[slot] = slots_to_amount.get(slot, 0.0) + amount


def build_bolus_series(bolus_df: pd.DataFrame) -> pd.Series:
    if bolus_df.empty:
        return pd.Series(dtype=np.float64)
    slots_to_amount: dict[pd.Timestamp, float] = {}
    for row in bolus_df.sort_values(["start", "source_order"]).itertuples(index=False):
        start = row.start
        end = row.end
        dose = float(row.dose)
        if pd.isna(end) or end <= start:
            slot = start.round(GRID_FREQ)
            slots_to_amount[slot] = slots_to_amount.get(slot, 0.0) + dose
        else:
            add_extended_bolus(slots_to_amount, start, end, dose)
    if not slots_to_amount:
        return pd.Series(dtype=np.float64)
    return (
        pd.Series(slots_to_amount, dtype=np.float64).groupby(level=0).sum().sort_index()
    )


def build_cho_series(meal_df: pd.DataFrame) -> pd.Series:
    if meal_df.empty:
        return pd.Series(dtype=np.float64)
    return meal_df.groupby("slot")["CHO"].sum().astype(np.float64).sort_index()


def convert_subject(subject_id: str, xml_paths: list[Path], output_dir: Path) -> Path:
    events = load_subject_events(xml_paths)
    cgm_series = build_cgm_series(events["cgm"])
    cgm_frame, candidate_days, dropped_days = build_cgm_frame(cgm_series)
    if cgm_frame.empty:
        output_df = pd.DataFrame(columns=TARGET_COLUMNS)
    else:
        date_index = pd.DatetimeIndex(cgm_frame["date"])
        basal_values = build_basal_values(
            date_index, events["basal"], events["temp_basal"]
        )
        bolus_series = build_bolus_series(events["bolus"])
        cho_series = build_cho_series(events["meal"])
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
    output_path = output_dir / f"New_Subject{subject_id}.csv"
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

    groups = group_xml_files_by_subject(discover_xml_files(input_dir))
    return [convert_subject(s, paths, output_dir) for s, paths in groups.items()]
