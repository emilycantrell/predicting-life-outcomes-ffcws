# Job files

## Descriptions of job files

- `jobfile_monkeys_master.toml` is an example job file that can be copied and edited to your desired settings.
- `jobfile_test_local_setup.toml` and `jobfile_test_azure_setup.toml` are very small jobs designed to test whether the code is working on your computer or your Azure VMs, following the setup instructions in the `docs/` directory.
- `jobfile_monkeys_full_run_2026-01-26.toml` was used to generate the main results in the manuscript we submitted for journal review and posted to arXiv.
- `jobfile_monkeys_height_with_2_manually_selected_features_2026-02-20_percentile_feature.toml` was used to generate additional results for two height outcomes in the "skill" section of our paper.

## Notes

(1) There is one known discrepancy between the organization of the job description file and the order of operations in the code: predictor imputation. In the job description file, predictor imputation is specified within `[pipeline.imputation]`, which may suggest that it occurs within each pipeline. In the code, however, predictor imputation occurs before the pipelines are run.

(2) If add_scales_to_outcomes = true, and if regression_outcomes is a list of strings, then the job will predict the outcomes in the list of strings plus all scale outcomes, even if the scale outcomes are not included in the list of strings. 

## Updating file hashes

Input files in the jobfiles are specified using a file path and an MD5 hash. The hash is used to verify that the file has not changed. File hashes must therefore be updated whenever the corresponding files change.

On a Mac, you can obtain the MD5 hash for a file by typing `md5` followed by the file path in the terminal:

    md5 <path/to/file>
