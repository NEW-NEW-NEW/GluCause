"""Ablation analyses for Supplementary Fig. 3."""

from .intervention_analysis import (
    load_trajectories,
    trajectory_summary,
    render_trajectories,
    validate_curve_summary,
)
from .metrics import read_summary, write_csv

MODELS = {
    "Without Causal Loss": 120,
    "Direct CGM output": 120,
    "Single Dynamics Head": 120,
}


def run(args):
    if args.input:
        summary = trajectory_summary(load_trajectories(args.input, "ablation", MODELS))
    else:
        summary = read_summary(
            args.summary, ["model", "scenario", "minute"], {"cgm_change_from_present"}
        )
        validate_curve_summary(summary, "ablation", MODELS)
    write_csv(summary, args.output, "supp_fig3_trajectory_summary")
    render_trajectories(summary, "ablation", args.output, "supp_fig3")
