"""Command-line entry points for the distribution package."""

import argparse
import importlib
from pathlib import Path

from .preprocessing.common import require_empty_directory

DATASETS = {
    "AZT1D": "azt1d",
    "BrisT1D": "brist1d",
    "DiaTrend": "diatrend",
    "HUPA-UCM": "hupa_ucm",
    "OhioT1DM": "ohiot1dm",
    "T1D-UOM": "t1d_uom",
}


def parser():
    root = argparse.ArgumentParser(description="Data preprocessing and analysis.")
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "preprocess",
        help="Convert source records, split canonical data, or construct windows.",
    )
    operations = prep.add_subparsers(dest="operation", required=True)
    for operation in ("convert", "split", "windows"):
        sub = operations.add_parser(operation)
        sub.add_argument("--input", type=Path, required=True)
        sub.add_argument(
            "--output", type=Path, required=True, help="New or empty output directory."
        )
        if operation == "convert":
            sub.add_argument("--dataset", choices=DATASETS, required=True)
        if operation == "windows":
            sub.add_argument(
                "--heldout-dataset",
                help="Exclude this dataset when estimating training set standardization statistics (LODO).",
            )
    descriptions = {
        "dynamics": "Fig. 4, Table 2, Supplementary Fig. 4.",
        "intervention": "Fig. 2–3, Supplementary Fig. 1–2 and Table 1–2.",
        "ablation": "Supplementary Fig. 3.",
        "latent": "Fig. 5a–g and Supplementary Note 2.",
        "lodo": "Table 1.",
        "virtual": "Table 3 and Supplementary Fig. 5.",
        "policy": "Fig. 6a–d: bolus insulin and rescue carbohydrate policy evaluation.",
    }
    for command, description in descriptions.items():
        sub = commands.add_parser(command, help=description, description=description)
        source = sub.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--input", type=Path, help="Detailed CSV/Parquet source table."
        )
        source.add_argument(
            "--summary",
            type=Path,
            help="Aggregated summary table. Latent abcd/fg use a directory.",
        )
        sub.add_argument(
            "--output", type=Path, required=True, help="New or empty output directory."
        )
        if command == "dynamics":
            sub.add_argument(
                "--clarke-summary",
                type=Path,
                help="Optional Table 2 Clarke Error Grid zone percentages in summary mode.",
            )
        elif command == "intervention":
            sub.add_argument(
                "--kind", choices=["present", "future", "replacement"], required=True
            )
        elif command == "latent":
            sub.add_argument("--panel", choices=["abcd", "e", "fg"], required=True)
            sub.add_argument(
                "--responses",
                type=Path,
                help="Matched baseline, carbohydrate, and bolus insulin trajectories for panels a–d.",
            )
            sub.add_argument(
                "--training-insulin",
                type=Path,
                help="Canonical training set insulin records for panels f–g.",
            )
        elif command == "virtual":
            sub.add_argument(
                "--curves",
                type=Path,
                help="Optional Supplementary Fig. 5 curve summary in summary mode.",
            )
        elif command == "policy":
            sub.add_argument(
                "--kind",
                choices=["bolus_insulin", "rescue_carbohydrate"],
                required=True,
            )
            sub.add_argument(
                "--detections",
                type=Path,
                help="Hypoglycemia labels and detections for rescue carbohydrate policy evaluation.",
            )
    return root


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        if args.command == "preprocess":
            if args.operation == "convert":
                module = importlib.import_module(
                    f".preprocessing.{DATASETS[args.dataset]}", "glucause_analysis"
                )
                paths = module.convert(args.input, args.output)
                print(f"Wrote {len(paths)} canonical subject files to {args.output}")
            elif args.operation == "split":
                from .preprocessing.common import split_directory

                count = split_directory(args.input, args.output)
                print(f"Wrote {count} split files to {args.output}")
            else:
                from .preprocessing.common import prepare_windows

                count = prepare_windows(args.input, args.output, args.heldout_dataset)
                print(f"Wrote {count} continuous windows to {args.output}")
            return
        if args.input and (
            getattr(args, "curves", None) or getattr(args, "clarke_summary", None)
        ):
            raise ValueError(
                "--curves/--clarke-summary are available only with --summary."
            )
        require_empty_directory(args.output)
        modules = {
            "dynamics": "dynamics_evaluation",
            "intervention": "intervention_analysis",
            "ablation": "ablation_analysis",
            "latent": "latent_analysis",
            "lodo": "lodo_evaluation",
            "virtual": "virtual_evaluation",
            "policy": "policy_evaluation",
        }
        module = importlib.import_module(
            f".{modules[args.command]}", "glucause_analysis"
        )
        module.run(args)
        print(f"Wrote analysis outputs to {args.output}")
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        command_parser.error(str(error))


if __name__ == "__main__":
    main()
