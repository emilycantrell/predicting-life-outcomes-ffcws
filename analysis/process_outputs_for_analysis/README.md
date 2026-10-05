# Process outputs for analysis

This directory contains code for processing job outputs into files used for analysis. The scripts parse raw JSON output files, run quality-control checks, select winning pipelines, calculate predictive performance and standard errors, extract results for the simple benchmark, summarize computation time, and produce counts used in the STROBE variable flowcharts in the appendix.

## Files

### `01a_process_json_to_rds_for_jobs_with_automated_feature_selection.R`

Converts the raw JSON output into several RDS files used in subsequent analysis, including parsed results, predictions, hyperparameter-tuning results, and feature-selection information. This is for jobs that use automated feature selection, such as the main job in our paper.

### `01b_process_json_to_rds_for_jobs_with_manual_feature_selection.R`

This script is the manual-feature-selection counterpart to `01a_process_json_to_rds_for_jobs_with_automated_feature_selection.R`. Only one of the two `01` scripts should be used for a given job, depending on how feature selection was performed. In our paper, we use manual feature selection for our exploration of the predictability of two height outcomes in the section on "skill."

### `02_test_parsed_rds_files.R`

Runs quality-control checks on the parsed RDS files created by the `01` scripts. These tests check conditions that should never fail for valid outputs.

### `03_select_winning_pipelines_and_calculate_se.R`

Selects the winning pipeline and hyperparameter draw for each outcome and outer fold using performance on the inner cross-validation folds. It then combines predictions across outer folds to calculate an overall cross-validated R-squared and its standard error for each outcome.

### `04_extract_simple_benchmark_and_calculate_se.R`

Extracts results for the simple benchmark: standard scaling, feature selection via k-best-1 with f_regression, and linear or logistic regression as appropriate. It calculates overall cross-validated R-squared and its standard error for each outcome's benchmark.

### `05_parse_timing_files.R`

Parses `timing.json` files from the worker machines and summarizes computation time overall, by machine, and by processing step. It writes CSV summaries to `parsed/timing_parsed/`.

### `06_strobe_flowchart.Rmd`

Uses preprocessing metadata and parsed results to calculate and check the variable counts reported in the STROBE flowcharts in the appendix, which map the number of variables in the raw data to the final numbers of outcomes and features.

## Workflow

1. Run a job as instructed in `docs/1-Local-Setup.md` or `docs/2-Azure-Setup.md`. This will produce the json files that will be processed by the scripts in this directory.

2. Install the `groundhog` package in R if you do not already have it.

3. Edit the configuration values near the beginning of the `01` to `06` files, including `rootdir` and `slug`, to reflect the location and name of the job outputs. In the `06` file you also need to edit `n_machines` to reflect the number of virtual machines used in the run.

4. Run the appropriate `01` script depending on whether the job used automated or manual feature selection.

5. Run the remaining scripts in this directory in numerical order.
