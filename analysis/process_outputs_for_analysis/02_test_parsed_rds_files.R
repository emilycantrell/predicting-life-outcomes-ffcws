# This code is run after parsing the JSON files to rds files using either 
# 01a_process_json_to_rds_for_jobs_with_automated_feature_selection.R or 
# 01b_process_json_to_rds_for_jobs_with_manual_feature_selection.R.
# It only includes tests that should never fail.

rm(list=ls())
library(groundhog)
pkgs <- c("tidyverse")
groundhog.day <- "2023-06-01"
groundhog.library(pkgs, groundhog.day)
set.seed(08544)


########

# Directory containing the job directory or directories you generated
# (e.g., if you ran multiple jobs and stored their outputs in the same parent directory).
# This should be the parent directory of the directory named by `slug`.
rootdir <- "/path/to/parent/directory/"

# Name of the specific job directory to process.
slug <- "my_job_name"

STOP_ON_ERROR <- TRUE # TRUE means stop on error, FALSE means warn
print_width <- 80
# Disable ANSI escape codes
options(crayon.enabled = FALSE) # makes it easier capture printed output to files

VERBOSE <- TRUE
print(paste("Processing slug:", slug, "STOP_ON_ERROR:", STOP_ON_ERROR))
#########

# read files
if (VERBOSE) print("Beginning never fail tests of results_cv_merged . . . ")
results_cv_merged <- readRDS(file = file.path(rootdir, slug, "parsed", paste0(slug, "_results_cv_merged.rds")))

# Some failures are expected for these models
models_allowed_to_fail <- c("epsilon support vector regressor", "nu support vector regressor")

# Exclude only failed tasks from models that are allowed to fail
results_cv_merged_for_never_fail_tests <- results_cv_merged %>%
  filter(!(failed == TRUE & model %in% models_allowed_to_fail))

### tests of results_cv_merged
# no tasks should have failed
if (any(results_cv_merged_for_never_fail_tests$failed)) {
  info_str <- "Some tasks failed"
  print(info_str)
  results_cv_merged_for_never_fail_tests %>%
    filter(failed == TRUE) %>%
    select(task_id, failed) %>%
    print(width = print_width)
  
  results_cv_merged_for_never_fail_tests %>%
    filter(failed == TRUE) %>%
    select(task_id, outcome, outerfold, scaler, model) %>%
    print(width = print_width)
  
  ifelse(STOP_ON_ERROR, stop(info_str), warning(info_str)) 
}

# is_tuned should never be NA
if (any(is.na(results_cv_merged_for_never_fail_tests$is_tuned))) {
  info_str <- "Some tasks is_tuned is NA"
  print(info_str)
  results_cv_merged_for_never_fail_tests %>%
    filter(is.na(is_tuned)) %>%
    select(task_id, is_tuned) %>%
    print(width = print_width)
  
  results_cv_merged_for_never_fail_tests %>%
    filter(is.na(is_tuned)) %>%
    select(task_id, outcome, outerfold, scaler, model) %>%
    print(width = print_width)
  
  ifelse(STOP_ON_ERROR, stop(info_str), warning(info_str)) 
}

# is_tuned should never be false
if (any(results_cv_merged_for_never_fail_tests$is_tuned == FALSE, na.rm=TRUE)) {
  info_str <- "Some tasks is_tuned is FALSE"
  print(info_str)
  
  results_cv_merged_for_never_fail_tests %>%
    filter(is_tuned==FALSE) %>%
    select(task_id, is_tuned) %>%
    print(width = print_width)
  
  results_cv_merged_for_never_fail_tests %>%
    filter(is_tuned==FALSE) %>%
    select(task_id, outcome, outerfold, scaler, model) %>%
    print(width = print_width)
  
  ifelse(STOP_ON_ERROR, stop(info_str), warning(info_str)) 
}

# there should never be an error
if ("error" %in% colnames(results_cv_merged_for_never_fail_tests) &&
    any(!is.na(results_cv_merged_for_never_fail_tests$error))) {
  info_str <- "Some tasks there is an error"
  print(info_str)
  cols_to_print <- c("task_id", "error", "stack_trace")
  results_cv_merged_for_never_fail_tests %>%
    filter(!is.na(error)) %>%
    select(any_of(cols_to_print)) %>%
    print(width = print_width)
  ifelse(STOP_ON_ERROR, stop(info_str), warning(info_str))
}

# there should never be a stack_trace
if ("stack_trace" %in% colnames(results_cv_merged_for_never_fail_tests) &&
    any(!is.na(results_cv_merged_for_never_fail_tests$stack_trace))) {
  info_str <- "Some tasks there is a stack trace"
  print("Here are a few examples:")
  cols_to_print <- c("task_id", "error", "stack_trace")
  results_cv_merged_for_never_fail_tests %>%
    filter(!is.na(stack_trace)) %>%
    select(any_of(cols_to_print)) %>%
    print(width = print_width)
  ifelse(STOP_ON_ERROR, stop(info_str), warning(info_str))
}

# we should not use huber regression
if (any(results_cv_merged$model == "huber regressor")) stop("ERROR: some models use huber regressor")

# no duplicate task_id
if (any(duplicated(results_cv_merged$task_id))) stop("ERROR: task_id not unique")

# there should be no task_ids with 0 variance inner folds
if (any(results_cv_merged$zero_var_innerfold == TRUE)) stop("ERROR: there are some task_ids with zero variance innerfolds")

# there should be no task_ids with 0 variance outerfold train or test
if (any(results_cv_merged$variance_train == 0) | any(results_cv_merged$variance_test == 0)) stop("ERROR: there are some task_ids with zero variance outerfolds")

# some variables should be > 0 (list not exhaustive)
vars_should_be_greater_than_zero <- c("num_samples", "num_unique", "nobs_train", "nobs_test")
invalid_tasks <- results_cv_merged_for_never_fail_tests %>%
  filter(if_any(all_of(vars_should_be_greater_than_zero), ~ . <= 0))
if (nrow(invalid_tasks) > 0) {
  info_str <- "Some values should be > 0 and are not"
  print("Here are a few examples:")
  print(invalid_tasks)
  ifelse(STOP_ON_ERROR, stop(info_str), warning(info_str)) 
}
rm(invalid_tasks)

# some variables should be not NA (list not exhaustive)
vars_should_not_be_NA <- c( "task_id", "outcome", "outerfold", "pipeline_id", "scaler", "model", "failed",  vars_should_be_greater_than_zero, "r2_score")
invalid_tasks <- results_cv_merged_for_never_fail_tests %>%
  filter(if_any(all_of(vars_should_not_be_NA), ~ is.na(.)))
if (nrow(invalid_tasks) > 0) {
  info_str <- "Some values should not be NA but are"
  print(info_str)
  print("Here are a few examples:")
  invalid_tasks %>% 
    select(task_id) %>%
    print(width = print_width)
  
  invalid_tasks %>% 
    select(all_of(vars_should_not_be_NA)) %>%
    print(width = print_width)
  ifelse(STOP_ON_ERROR, stop(info_str), warning(info_str)) 
}
rm(invalid_tasks)

rm(results_cv_merged_for_never_fail_tests)
if (VERBOSE) print("Ending never fail tests of results_cv_merged . . . ")


if (VERBOSE) print("Beginning never fail tests of predictions . . . ")
predictions <- readRDS(file = file.path(rootdir, slug, "parsed", paste0(slug, "_predictions.rds")))

# cleanup names
predictions$predictions_tib <- predictions$predictions
predictions$predictions <- NULL

# check if all predictions for a task are 0/1
check_predictions_all01 <- function(predictions_tib) {
  return(all(predictions_tib$prediction %in% c(0, 1)))
}

# check if any predictions for a task are NA
check_predictions_anyNA <- function(predictions_tib) {
  return(any(is.na(predictions_tib$prediction)))
}

predictions <- predictions %>%
  rowwise() %>%
  mutate(predictions_all01 = check_predictions_all01(predictions_tib)) %>%
  mutate(predictions_anyNA = check_predictions_anyNA(predictions_tib)) %>%
  ungroup()

results_cv_merged_predictions <- left_join(results_cv_merged,
                                           predictions,
                                           by = "task_id")

all_01_test <- results_cv_merged_predictions %>%
  group_by(pipeline_id) %>%
  summarize(prop_outcomes_all_01 = mean(predictions_all01))

if (any(all_01_test$prop_outcomes_all_01 == 1)) {
  all_01_test %>%
    filter(all_01_test$prop_outcomes_all_01 == 1) %>%
    print()
  stop("ERROR: some pipelines produce all 0/1 for all outcomes")
}
rm(predictions, results_cv_merged_predictions, all_01_test, results_cv_merged) # clean-up
if (VERBOSE) print("Ending never fail tests of predictions . . . ")


if (VERBOSE) print("Beginning never fail tests of hyperparameter_tuning_df . . . ")
hyperparameter_tuning_df <- readRDS(file = file.path(rootdir, slug, "parsed", paste0(slug, "_hyperparameter_tuning.rds")))

## Begin: test of relationship between TSE and R^2 for inner fold
# TSE was added to JSON late in the process so we are testing more carefully
print(paste("Beginning test of TSE and R^2:", Sys.time()))
# make long and extract needed info
name_filter <- "hyperparameter_tuning_fold_[:digit:]_(tse_predictions|tse_baseline|r2_score)"
# Check up to 1000 tasks
n_tasks_to_check <- min(1000, nrow(hyperparameter_tuning_df)) 

temp_long <- hyperparameter_tuning_df %>%
  sample_n(n_tasks_to_check) %>%
  unnest(hyperparameter_tuning) %>%
  filter(str_detect(name, name_filter)) %>%
  unnest_longer("value") # Note this was optimized on June 27, 2023 (this is just a big tibble and unnest_longer is slow)

# parse name to extract needed info
temp_long$innerfold <- str_extract(temp_long$name, "(?<=tuning_fold_)\\d+")
temp_long$measure <- str_extract(temp_long$name, "(tse_predictions|tse_baseline|r2_score)$")

# reshape
tib_task_value_innerfold <- pivot_wider(temp_long, id_cols = c(task_id, value_id, innerfold), 
                                        names_from = measure, values_from = value)
# calculate r2 and compare to original
tib_task_value_innerfold <- tib_task_value_innerfold %>%
  mutate(r2_score_from_tse = 1 - (tse_predictions/tse_baseline)) %>%
  mutate(r2_scores_match = all.equal(r2_score_from_tse, r2_score,)) 

if (any(!tib_task_value_innerfold$r2_scores_match)) {
  tib_task_value_innerfold %>%
    filter(r2_scores_match == FALSE) %>%
    print()
  stop("ERROR: r2_score precalculated doesn't match the one calculated from TSEs")
}
rm(name_filter, temp_long, tib_task_value_innerfold) # clean-up
print(paste("Ending test of TSE and R^2:", Sys.time()))
if (VERBOSE) print("Ending never fail tests of hyperparameter_tuning_df . . . ")
## End: test of relationship between TSE and R^2 for inner fold

print("Good news! Code completed successfully and is exiting cleanly.")

