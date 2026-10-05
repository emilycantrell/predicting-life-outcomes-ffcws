# This code calculates global R^2 and confidence intervals for pipelines with the "simple benchmark": 
  # Standard scaler, 
  # feature selection via k-best-1 with f_regression, 
  # logistic regression with L2 penalty (scikit default) for binary outcomes; 
  # linear regression for continuous and ordered categorical outcomes.
# It is an adjusted version of the code from 03_select_winning_pipelines_and_calculate_se.R. 

# NOTE ON TERMINOLOGY:
# In this code, "baseline" refers to using the mean of the training data as the prediction for all test cases (i.e., a null model),
# while "simple benchmark" or "benchmark" refers to the pipelines with the criteria described above.

library(groundhog)
pkgs <- c("tidyverse")
groundhog.day <- "2023-06-01"
groundhog.library(pkgs, groundhog.day)

#### READ FILES #### 

# Directory containing the job directory or directories you generated
# (e.g., if you ran multiple jobs and stored their outputs in the same parent directory).
# This should be the parent directory of the directory named by `slug`.
rootdir <- "/path/to/parent/directory/"

# Name of the specific job directory to process.
slug <- "my_job_name"

# Read files
results_cv_merged <- readRDS(file = file.path(rootdir, slug, "parsed", paste0(slug, "_results_cv_merged.rds"))) 
outcome_pipeline_outerfold_draw <- readRDS(file = file.path(rootdir, slug, "parsed", paste0(slug, "_outcome_pipeline_outerfold_draw.rds")))
predictions <- readRDS(file = file.path(rootdir, slug, "parsed", paste0(slug, "_predictions.rds")))
test_outcomes <- readRDS(file = file.path(rootdir, slug, "parsed", paste0(slug, "_cv_test_outcomes.rds")))

#### IDENTIFY PIPELINES THAT ARE THE "SIMPLE BENCHMARK" ####

# Identify "simple benchmark" pipelines for binary outcomes
# First filter to the correct scaler, feature selection options, and model
# The below are pipelines that "MIGHT" be benchmarks because I have not yet filtered to the hyperparameter draw with defaults.
logistic_regression_pipelines_that_might_be_benchmarks <- results_cv_merged %>%
  filter(scaler == "standard scaler", 
         fs1 == "f_regression", 
         fs3 == "1", 
         model == "logistic regressor") %>%
  select(task_id, scaler, model, fs1, fs2, fs3, mean_train, outcome_outerfold_id)
# Merge with outcome_pipeline_outerfold_draw 
logistic_regression_pipelines_that_might_be_benchmarks <- left_join(logistic_regression_pipelines_that_might_be_benchmarks, outcome_pipeline_outerfold_draw, by = "task_id")
# Filter to the pipelines that use default hyperparameters
logistic_regression_benchmarks <- logistic_regression_pipelines_that_might_be_benchmarks %>%
  filter(draw == "rv_0")
# Identify binary outcomes 
binary_outcomes <- unique(logistic_regression_benchmarks$outcome)

# Identify "simple benchmark" pipelines for continuous and ordered categorical outcomes
# Filter to the correct scaler, feature selection options, and model, and to only continuous and ordered categorical outcomes
# There is no hyperparameter tuning on linear regression pipelines, so I don't need to filter to default hyperparameter values
linear_regression_benchmarks <- results_cv_merged %>%
  filter(scaler == "standard scaler",
         fs1 == "f_regression", 
         fs3 == "1", 
         model == "linear regressor") %>%
  filter(!(outcome %in% binary_outcomes)) %>%
  select(task_id, scaler, model, fs1, fs2, fs3, mean_train, outcome_outerfold_id) 
# Merge with outcome_pipeline_outerfold_draw 
linear_regression_benchmarks <- left_join(linear_regression_benchmarks, outcome_pipeline_outerfold_draw, by = "task_id")

# Combine all benchmarks
benchmarks <- bind_rows(logistic_regression_benchmarks, linear_regression_benchmarks)

# Summaries for tests
n_outcomes <- n_distinct(results_cv_merged$outcome)
n_obs_per_outcome <- results_cv_merged %>%
  select(outcome, nobs_train, nobs_test) %>%
  mutate(nobs_total = nobs_train + nobs_test) %>%
  distinct(outcome, nobs_total)
if (nrow(n_obs_per_outcome) != n_outcomes) stop("ERROR: wrong number of rows")  
n_obs_total <- sum(n_obs_per_outcome$nobs_total)

#### CALCULATE GLOBAL OUTERFOLD R^2 ####

# merge on predictions
benchmarks <- left_join(x = benchmarks, y = predictions, by = "task_id")
rm(predictions) # cleanup

# merge on test outcomes
benchmarks <- left_join(x = benchmarks, y = test_outcomes, by = "outcome_outerfold_id")
rm(test_outcomes) # cleanup

# check that predictions and test_outcomes have the same number of cases
if (!all(map_int(benchmarks$predictions, nrow) == map_int(benchmarks$test_outcomes, nrow))) stop("ERROR: number of cases don't match")

# collect benchmark predictions and baseline predictions
outcome_benchmark_predictions <- benchmarks %>% 
  select(outcome, predictions, mean_train) %>% 
  unnest(cols = c(predictions)) %>%
  dplyr::rename(prediction_benchmark = prediction, prediction_baseline = mean_train) %>%
  group_by(outcome) %>%
  nest(benchmark_predictions_all_folds = c(ffcws_id, prediction_benchmark, prediction_baseline)) %>%
  ungroup()
if (nrow(outcome_benchmark_predictions) != n_outcomes) stop("ERROR: wrong number of rows")  

# collect outcome truth
outcome_truth <- benchmarks %>% 
  select(outcome, test_outcomes) %>% 
  unnest(cols = c(test_outcomes)) %>% 
  group_by(outcome) %>%
  nest(truth_all_folds = c(ffcws_id, outcome_value)) %>%
  ungroup()
if (nrow(outcome_truth) != n_outcomes) stop("ERROR: wrong number of rows")  

# combine predictions and truth
outcome_benchmark <- full_join(x = outcome_benchmark_predictions, y = outcome_truth, by = c("outcome"))
if (nrow(outcome_benchmark) != n_outcomes) stop("ERROR: wrong number of rows")  

merge_tibbles <- function(tib_predictions, tib_truth, RUN.CHECKS = TRUE) {
  if (RUN.CHECKS) {
    if (!is_tibble(tib_predictions)) stop("ERROR: tib_predictions is not tibble")
    if (!is_tibble(tib_truth)) stop("ERROR: tib_truth is not tibble")
    if (!(all(tib_predictions$ffcws_id == tib_truth$ffcws_id))) stop("ERROR: ffcws_ids don't match")
  }
  
  left_join(tib_predictions, tib_truth, by = "ffcws_id") %>%
    return()
}  

outcome_benchmark <- outcome_benchmark %>%
  mutate(predictions_truth = map2(benchmark_predictions_all_folds, truth_all_folds, merge_tibbles)) %>%
  select(outcome, predictions_truth)
if (nrow(outcome_benchmark) != n_outcomes) stop("ERROR: wrong number of rows")  
rm(merge_tibbles) # cleanup

# make long
outcome_benchmark_long <- outcome_benchmark %>%
  unnest(cols = c(predictions_truth))
if (nrow(outcome_benchmark_long) != n_obs_total) stop("ERROR: wrong number of rows")

# do calculations
outcome_benchmark_long <- outcome_benchmark_long %>%
  mutate(err_baseline = prediction_baseline - outcome_value) %>%
  mutate(err_benchmark = prediction_benchmark - outcome_value) %>%
  mutate(sq_err_baseline = err_baseline^2) %>%
  mutate(sq_err_benchmark = err_benchmark^2) 

# renest
outcome_benchmark <- outcome_benchmark_long %>%
  group_by(outcome) %>%
  nest(data = c(ffcws_id, prediction_benchmark, prediction_baseline, outcome_value, err_baseline, err_benchmark, sq_err_baseline, sq_err_benchmark)) %>%
  ungroup()

# add n
outcome_benchmark <- outcome_benchmark %>%
  mutate(n = map_int(data, nrow))

# collect benchmark task_id, outerfold, outcome_outerfold_id, and pipeline_id to be merged on
outcome_benchmark_ids <- benchmarks %>% 
  select(outcome, outerfold, task_id, outcome_outerfold_id, pipeline_id) %>% 
  group_by(outcome) %>%
  nest(benchmark_ids = c(outerfold, task_id, outcome_outerfold_id, pipeline_id)) %>%
  ungroup()
if (nrow(outcome_benchmark_ids) != n_outcomes) stop("ERROR: wrong number of rows")  

outcome_benchmark <- outcome_benchmark %>%
  left_join(x = ., y = outcome_benchmark_ids, by = c("outcome"))

# calculate r2
outcome_benchmark <- outcome_benchmark %>% 
  mutate(mean_sq_err_benchmark = map_dbl(data, ~ mean(.x$sq_err_benchmark))) %>%
  mutate(mean_sq_err_baseline = map_dbl(data, ~ mean(.x$sq_err_baseline))) %>%
  mutate(r2_benchmark = 1 - (mean_sq_err_benchmark / mean_sq_err_baseline))

#### CALCULATE STANDARD ERROR ####
# inspired by https://stackoverflow.com/a/71037030

outcome_benchmark <- outcome_benchmark %>% 
  mutate(sd_sq_err_benchmark = map_dbl(data, ~ sd(.x$sq_err_benchmark))) %>%
  mutate(sd_sq_err_baseline = map_dbl(data, ~ sd(.x$sq_err_baseline))) %>%
  mutate(cov_sq_err = map_dbl(data, ~ cov(.x$sq_err_benchmark, .x$sq_err_baseline)))

outcome_benchmark <- outcome_benchmark %>%
  mutate(hat_se_benchmark = 1/sqrt(n) * sd_sq_err_benchmark) %>%
  mutate(hat_se_baseline = 1/sqrt(n) * sd_sq_err_baseline)

# qnorm(0.975) is about 1.96, qnorm(0.025) is about -1.96
outcome_benchmark <- outcome_benchmark %>%
  mutate(lower_benchmark = mean_sq_err_benchmark + qnorm(0.025) * hat_se_benchmark) %>%
  mutate(upper_benchmark = mean_sq_err_benchmark + qnorm(0.975) * hat_se_benchmark) %>%
  mutate(lower_baseline = mean_sq_err_baseline + qnorm(0.025) * hat_se_baseline) %>%
  mutate(upper_baseline = mean_sq_err_baseline + qnorm(0.975) * hat_se_baseline) 

# create ratio and calculate se of ratio (formula from Sarndall et al 2003)
outcome_benchmark <- outcome_benchmark %>%
  mutate(ratio = (mean_sq_err_benchmark / mean_sq_err_baseline)) %>%
  mutate(term1 = 1 / mean_sq_err_baseline^2) %>%
  mutate(term2 = 1 / n) %>%
  mutate(term3 = sd_sq_err_benchmark^2) %>%
  mutate(term4 = ratio^2 * sd_sq_err_baseline^2) %>%
  mutate(term5 = 2 * ratio * cov_sq_err) %>%
  mutate(hat_var_ratio = term1 * term2 * (term3 + term4 - term5))

if (any(outcome_benchmark$hat_var_ratio < 0)) {
  print("These hat_var_ratio are negative, but variance can't be negative")
  outcome_benchmark %>%
    filter(hat_var_ratio < 0) %>%
    print(n = Inf)
  print("If these are just barely negative that could be because of numerical precision issues")
  print("If so, you should probably set these very, very small negative values to 0")
  stop("ERROR: problem with estimated variance of ratio")
}

outcome_benchmark <- outcome_benchmark %>%  
  mutate(hat_se_ratio = sqrt(hat_var_ratio)) %>%
  mutate(lower_ratio = r2_benchmark + qnorm(0.025) * hat_se_ratio) %>%
  mutate(upper_ratio = r2_benchmark + qnorm(0.975) * hat_se_ratio)

if (any(outcome_benchmark$upper_ratio > 1)) stop("ERROR: Some benchmark pipelines have CI for R^2 that include values bigger than 1")

#### SAVE OUTPUT ####
location_for_output <- file.path(rootdir, slug, "parsed")
saveRDS(outcome_benchmark, file = file.path(location_for_output, paste0(slug, "_simple_benchmark.rds")))

print("Good news! Code completed successfully and is exiting cleanly.")