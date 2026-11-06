# GluCause data preprocessing and analysis

Python code for data preprocessing and analysis accompanying the GluCause study.

The package includes dataset preprocessing and analysis of numerical results. GluCause and comparison-model implementations, training, weights, inference, representation extraction, simulator code and parameters, and bolus insulin and rescue carbohydrate policy implementations are excluded.

## Installation and quick start

Use Python 3.10–3.12. From this directory:

```bash
python -m pip install .
glucause-analysis --help
glucause-analysis dynamics --help
```

Dependencies are listed in [pyproject.toml](pyproject.toml). Each command requires a new or empty output directory and leaves its inputs unchanged.

For example, convert AZT1D records, split them chronologically, and construct windows:

```bash
glucause-analysis preprocess convert --dataset AZT1D --input /path/to/CGM_Records --output /path/to/canonical/AZT1D
glucause-analysis preprocess split --input /path/to/canonical --output /path/to/splits
glucause-analysis preprocess windows --input /path/to/splits --output /path/to/processed
```

For leave-one-dataset-out (LODO) preprocessing, add `--heldout-dataset AZT1D` to the window command to exclude that dataset when estimating training set standardization statistics. Run separately for each held-out dataset.

Analyze detailed records or an aggregated summary:

```bash
glucause-analysis dynamics --input modeling_results.parquet --output results/dynamics
glucause-analysis dynamics --summary metric_summary.csv --output results/dynamics_from_summary
```

Analysis commands are `dynamics`, `intervention`, `ablation`, `latent`, `lodo`, `virtual`, and `policy`. Use `glucause-analysis <command> --help` for command-specific options.

## Files and paper correspondence

Modules are in `src/glucause_analysis/`. The table links to their implementations.

| Module | Purpose | Paper correspondence |
| --- | --- | --- |
| [preprocessing/](src/glucause_analysis/preprocessing/) | Dataset converters | Methods: Data source and Data preprocessing |
| [preprocessing/common.py](src/glucause_analysis/preprocessing/common.py) | Record selection, features, splits, standardization, and windows | Methods: Data preprocessing, Eqs. (1)–(3) |
| [intervention_analysis.py](src/glucause_analysis/intervention_analysis.py) | Added interventions and replacement analysis | Fig. 2–3, Supplementary Fig. 1–2 and Table 1–2 |
| [ablation_analysis.py](src/glucause_analysis/ablation_analysis.py) | Ablation analysis | Supplementary Fig. 3 |
| [dynamics_evaluation.py](src/glucause_analysis/dynamics_evaluation.py) | Error metrics, uncertainty intervals, and Clarke Error Grid | Fig. 4, Table 2, Supplementary Fig. 4 |
| [latent_analysis.py](src/glucause_analysis/latent_analysis.py) | Latent representations and intervention responses | Fig. 5a–g, Supplementary Note 2 |
| [policy_evaluation.py](src/glucause_analysis/policy_evaluation.py) | Bolus insulin and rescue carbohydrate policy outcomes | Fig. 6a–d |
| [lodo_evaluation.py](src/glucause_analysis/lodo_evaluation.py) | Leave-one-dataset-out evaluation | Table 1 |
| [virtual_evaluation.py](src/glucause_analysis/virtual_evaluation.py) | Virtual-subject intervention evaluation | Table 3, Supplementary Fig. 5 |

`metrics.py` and `plot_style.py` provide shared calculations and plotting. `cli.py`, `__init__.py`, and `pyproject.toml` support execution and packaging.

## Data and inputs

Obtain source datasets under their providers' terms: [AZT1D](https://data.mendeley.com/datasets/gk9m674wcx/1), [BrisT1D](https://data.bris.ac.uk/data/dataset/33z5jc8fa6tob21ptrugzqog08), [DiaTrend](https://www.synapse.org/Synapse:syn38187184), [HUPA-UCM](https://data.mendeley.com/datasets/3hbcscwz44/1), [OhioT1DM](https://webpages.charlotte.edu/rbunescu/ohiot1dm.html), and [T1D-UOM](https://zenodo.org/records/15806142). Use these exact dataset names for converter options and canonical directories.

- Preprocessing reads provider records and produces canonical CSV files, training/validation/test sets (`Training/`, `Validation/`, `Test/`), standardization statistics (`standardization_statistics.csv`), and subject windows. Canonical split input follows `<dataset>/New_Subject*.csv`.
- Analysis inputs are CSV or Parquet. `--input` computes statistics from detailed records. `--summary` renders aggregated results. Latent `abcd` and `fg` summaries use directories containing the corresponding plot-data files.
- Preserve `(dataset, subject_id)` identifiers across related inputs. Trajectories use complete 5-minute grids. CGM and interval bounds are in mg/dL, carbohydrate amounts in g, and insulin amounts in U. Canonical insulin columns represent amounts per 5-minute row, with `INS = Basal + Bolus`. INS interventions use bolus insulin.
- Outcome evaluation uses the test set. Latent analyses also use the training and validation sets for fitting and selection, and the training set for TDD. Subject-level summaries use equal weights and sample SD (`ddof=1`).

Required columns, model/scenario labels, and validation rules are defined in the input-loading functions of the linked modules. Prepare files in those formats before running the examples. Analysis outputs are CSV tables and, where applicable, PDF and PNG figures named by paper panel.

## Citation and rights

Please cite the accompanying paper:

Data-Driven Personalized Blood Glucose Dynamics Modeling with Causally Informed Carbohydrate and Insulin Interventions. Wonjun Hwang, Sooyeon Jung, Junyoung Yoo, Yein Lee, Gwanwoo Kim, Sunguk Hong, and Sung-Min Park.

All rights reserved. See [LICENSE](LICENSE).
