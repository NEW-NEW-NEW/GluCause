"""Table 1: leave-one-dataset-out evaluation."""

from .dynamics_evaluation import load_results, subject_metrics
from .metrics import read_summary, subject_summary, write_csv, require_metrics

MODELS = {"GluCause", "GluCause (Non-personalized)"}
DATASETS = {
    "AZT1D",
    "BrisT1D",
    "DiaTrend",
    "HUPA-UCM",
    "OhioT1DM",
    "T1D-UOM",
}


def run(args):
    if args.input:
        frame = load_results(args.input, paired=True)
        if not set(frame.model).issubset(MODELS):
            raise ValueError(
                "Table 1 includes only GluCause and GluCause (Non-personalized)."
            )
        subject = subject_metrics(frame, uncertainty=False)
        subject["heldout_dataset"] = subject.dataset
        summary = subject_summary(
            subject, ["heldout_dataset", "model", "horizon"], ["MAE", "RMSE"], ddof=1
        )
        write_csv(subject, args.output, "lodo_subject_metrics")
    else:
        summary = read_summary(
            args.summary, ["heldout_dataset", "model", "horizon"], {"MAE", "RMSE"}
        )
    if not set(summary.model).issubset(MODELS) or not set(
        summary.heldout_dataset
    ).issubset(DATASETS):
        raise ValueError("Unexpected model or held-out dataset.")
    if not summary.horizon.isin([30, 60, 90, 120]).all():
        raise ValueError("Table 1 horizons are 30, 60, 90, and 120 minutes.")
    require_metrics(summary, ["heldout_dataset", "model", "horizon"], {"MAE", "RMSE"})
    if (summary.groupby("heldout_dataset").n_subjects.nunique() != 1).any():
        raise ValueError(
            "Within each held-out dataset, all supplied models and horizons "
            "must have the same subject count."
        )
    write_csv(summary, args.output, "table1_lodo")
