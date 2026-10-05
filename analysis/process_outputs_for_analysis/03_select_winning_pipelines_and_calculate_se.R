# This file selects the winning pipeline for each outcome-outerfold and calculates standard errors.
# The "monkeys" terminology in object names refers to an internal nickname for the project, 
# referencing the "million monkeys" theorem since we have many pipelines attempting to 
# predict each outcome: https://en.wikipedia.org/wiki/Infinite_monkey_theorem

rm(list = ls())
library(groundhog)
pkgs <- c("tidyverse")
groundhog.day <- "2023-06-01"
groundhog.library(pkgs, groundhog.day)

#######################################################################
## Background information on file locations and run
#######################################################################

# Directory containing the job directory or directories you generated
# (e.g., if you ran multiple jobs and stored their outputs in the same parent directory).
# This should be the parent directory of the directory named by `slug`.
rootdir <- "/path/to/parent/directory/"

# Name of the specific job directory to process.
slug <- "my_job_name"

# Where should the output file be saved? 
location_for_output <- file.path(rootdir, slug, "parsed")

###
# Begin: Select winning pipelines
###
# Read data
results_cv_merged <- readRDS(file = file.path(rootdir, slug, "parsed", paste0(slug, "_results_cv_merged.rds")))
hyperparameter_tuning <- readRDS(file = file.path(rootdir, slug, "parsed", paste0(slug, "_hyperparameter_tuning.rds")))

# Summaries for tests
n_outcomes <- length(unique(results_cv_merged$outcome))
n_outerfolds <- length(unique(results_cv_merged$outerfold))

n_obs_per_outcome <- results_cv_merged %>%
  select(outcome, nobs_train, nobs_test) %>%
  mutate(nobs_total = nobs_train + nobs_test) %>%
  distinct(outcome, nobs_total)
if (nrow(n_obs_per_outcome) != n_outcomes) stop("ERROR: wrong number of rows")  

n_obs_total <- sum(n_obs_per_outcome$nobs_total)

# process data
tuning_info <- results_cv_merged %>% 
  select(task_id, outcome_outerfold_id, pipeline_id, outcome, outerfold, model, is_tuned) %>%
  left_join(hyperparameter_tuning, ., by = "task_id") 
rm(hyperparameter_tuning) # cleanup

tuning_info <- tuning_info %>%
  filter(is_tuned == TRUE) %>%
  select(-is_tuned)

### Prep results for selection
start <- Sys.time()
outcome_pipeline_outerfold_innerfold_metric <- tuning_info %>%
  unnest(cols = c(hyperparameter_tuning)) %>% 
  filter(str_detect(name, "_fold_[:number:]_r2_score") | str_detect(name, "_fold_[:number:]_tse_predictions") | str_detect(name, "_fold_[:number:]_tse_baseline")) %>%
  unnest_longer(col = value, values_to = "value", indices_to = "draw") %>% # this is the main slow step
  mutate(innerfold = str_extract(name, "_[:number:]_")) %>% # keep _ in fold number to avoid false positives
  mutate(innerfold = str_extract(name, "[:number:]")) %>% # remove _ from fold number
  select(outcome, outerfold, pipeline_id, draw, innerfold, name, value, task_id, outcome_outerfold_id)
stop <- Sys.time()
print("This data wrangle took:")
print(difftime(stop, start))
rm(start, stop) # cleanup

# prep to turn from long to wider
outcome_pipeline_outerfold_innerfold_metric$name <- str_remove(outcome_pipeline_outerfold_innerfold_metric$name, 
                                                               "hyperparameter_tuning_fold_[:number:]_")

outcome_pipeline_outerfold_innerfold_draw <- outcome_pipeline_outerfold_innerfold_metric %>% 
  pivot_wider(names_from = name, values_from = value) 

n_innerfolds <- outcome_pipeline_outerfold_innerfold_draw %>%
  pull(innerfold) %>%
  unique() %>%
  length()

start <- Sys.time()
outcome_pipeline_outerfold_draw <- outcome_pipeline_outerfold_innerfold_draw %>%
  group_by(task_id, outcome, pipeline_id, outerfold, draw) %>%
  summarize(mean_r2_across_innerfolds = mean(r2_score),
            median_r2_across_innerfolds = median(r2_score),
            sd_r2_across_innerfolds = sd(r2_score),
            total_tse_predictions = sum(tse_predictions),
            total_tse_baseline = sum(tse_baseline),
            .groups = "drop")
if ((nrow(outcome_pipeline_outerfold_draw) * n_innerfolds) != nrow(outcome_pipeline_outerfold_innerfold_draw)) stop("ERROR: unexpected sizes")
stop <- Sys.time()
print("Calculating summaries took:")
print(difftime(stop, start))
rm(start, stop) # cleanup

outcome_pipeline_outerfold_draw <- outcome_pipeline_outerfold_draw %>%
  mutate(global_r2_innerfold = 1 - (total_tse_predictions/total_tse_baseline)) %>%
  select(-total_tse_predictions, -total_tse_baseline)

# checks
r2s <- outcome_pipeline_outerfold_draw %>%
  select("mean_r2_across_innerfolds", "median_r2_across_innerfolds", "global_r2_innerfold")
if (any(r2s > 1)) stop("ERROR: some r2s > 1")
rm(r2s)
if (any(outcome_pipeline_outerfold_draw$sd_r2_across_innerfolds < 0)) stop("ERROR: sd_r2_across_innerfolds < 0")

###
saveRDS(outcome_pipeline_outerfold_draw, file = file.path(location_for_output, paste0(slug, "_outcome_pipeline_outerfold_draw.rds")))
gc()


# for each outcome-outerfold pick the winning pipeline-draw 
set.seed(08544) # ties broken randomly
outcome_outerfold_selected_global_r2 <- outcome_pipeline_outerfold_draw %>%
  group_by(outcome, outerfold) %>%
  slice_max(global_r2_innerfold, with_ties = FALSE) %>%
  ungroup()
outcome_outerfold_selected <- outcome_outerfold_selected_global_r2
if (nrow(outcome_outerfold_selected) != (n_outcomes * n_outerfolds)) stop("ERROR: outcome_outerfold_selected unexpected number of rows")

# append mean of training data
training_means <- results_cv_merged %>% 
  select(outcome, outerfold, outcome_outerfold_id, mean_train) %>%
  distinct(outcome, outerfold, outcome_outerfold_id, mean_train)

outcome_outerfold_selected <- left_join(outcome_outerfold_selected,
                                        training_means, 
                                        by = c("outcome", "outerfold"))

##
# End: Select winning pipelines
###

### Prep for standard error calculations
# read data
predictions <- readRDS(file = file.path(rootdir, slug, "parsed", paste0(slug, "_predictions.rds")))
test_outcomes <- readRDS(file = file.path(rootdir, slug, "parsed", paste0(slug, "_cv_test_outcomes.rds")))

# merge on predictions
outcome_outerfold_selected <- left_join(x = outcome_outerfold_selected, y = predictions, by = "task_id")
rm(predictions) # cleanup

# merge on test outcomes
outcome_outerfold_selected <- left_join(x = outcome_outerfold_selected, y = test_outcomes, by = "outcome_outerfold_id")
rm(test_outcomes) # cleanup

# check that predictions and test_outcomes have the same number of cases
if (!all(map_int(outcome_outerfold_selected$predictions, nrow) == map_int(outcome_outerfold_selected$test_outcomes, nrow))) stop("ERROR: number of cases don't match")

# collect winning-pipeline predictions and baseline predictions
outcome_monkeys_predictions <- outcome_outerfold_selected %>% 
  select(outcome, predictions, mean_train) %>% 
  unnest(cols = c(predictions)) %>% 
  rename(prediction_monkeys = prediction, prediction_baseline = mean_train) %>%
  group_by(outcome) %>%
  nest(monkeys_predictions_all_folds = c(ffcws_id, prediction_monkeys, prediction_baseline)) %>%
  ungroup()
if (nrow(outcome_monkeys_predictions) != n_outcomes) stop("ERROR: wrong number of rows")  

# collect outcome truth
outcome_truth <- outcome_outerfold_selected %>% 
  select(outcome, test_outcomes) %>% 
  unnest(cols = c(test_outcomes)) %>% 
  group_by(outcome) %>%
  nest(truth_all_folds = c(ffcws_id, outcome_value)) %>%
  ungroup()
if (nrow(outcome_truth) != n_outcomes) stop("ERROR: wrong number of rows")  

# combine predictions and truth
outcome_monkeys <- full_join(x = outcome_monkeys_predictions, y = outcome_truth, by = "outcome")
if (nrow(outcome_monkeys) != n_outcomes) stop("ERROR: wrong number of rows")  

merge_tibbles <- function(tib_predictions, tib_truth, RUN.CHECKS = TRUE) {
  if (RUN.CHECKS) {
    if (!is_tibble(tib_predictions)) stop("ERROR: tib_predictions is not tibble")
    if (!is_tibble(tib_truth)) stop("ERROR: tib_truth is not tibble")
    if (!(all(tib_predictions$ffcws_id == tib_truth$ffcws_id))) stop("ERROR: ffcws_ids don't match")
  }
  
  left_join(tib_predictions, tib_truth, by = "ffcws_id") %>%
    return()
}  

outcome_monkeys <- outcome_monkeys %>%
  mutate(predictions_truth = map2(monkeys_predictions_all_folds, truth_all_folds, merge_tibbles)) %>%
  select(outcome, predictions_truth)
if (nrow(outcome_monkeys) != n_outcomes) stop("ERROR: wrong number of rows")  
rm(merge_tibbles) # cleanup

# make long
outcome_monkeys_long <- outcome_monkeys %>%
  unnest(cols = c(predictions_truth))
if (nrow(outcome_monkeys_long) != n_obs_total) stop("ERROR: wrong number of rows")

# do calculations
outcome_monkeys_long <- outcome_monkeys_long %>%
  mutate(err_baseline = prediction_baseline - outcome_value) %>%
  mutate(err_monkeys = prediction_monkeys - outcome_value) %>%
  mutate(sq_err_baseline = err_baseline^2) %>%
  mutate(sq_err_monkeys = err_monkeys^2) 

# renest
outcome_monkeys <- outcome_monkeys_long %>%
  group_by(outcome) %>%
  nest(data = c(ffcws_id, prediction_monkeys, prediction_baseline, outcome_value, err_baseline, err_monkeys, sq_err_baseline, sq_err_monkeys)) %>%
  ungroup()
  
# add n
outcome_monkeys <- outcome_monkeys %>%
  mutate(n = map_int(data, nrow))

# collect winning-pipeline task id, outerfold id, pipeline_id to be merged on
outcome_monkeys_ids <- outcome_outerfold_selected %>% 
  select(outcome, outerfold, task_id, outcome_outerfold_id, pipeline_id) %>% 
  group_by(outcome) %>%
  nest(monkeys_ids = c(outerfold, task_id, outcome_outerfold_id, pipeline_id)) %>%
  ungroup()
if (nrow(outcome_monkeys_ids) != n_outcomes) stop("ERROR: wrong number of rows")  

outcome_monkeys <- outcome_monkeys %>%
  left_join(x = ., y = outcome_monkeys_ids, by = "outcome")

# calculate r2
outcome_monkeys <- outcome_monkeys %>% 
  mutate(mean_sq_err_monkeys = map_dbl(data, ~ mean(.x$sq_err_monkeys))) %>%
  mutate(mean_sq_err_baseline = map_dbl(data, ~ mean(.x$sq_err_baseline))) %>%
  mutate(r2_monkeys = 1 - (mean_sq_err_monkeys / mean_sq_err_baseline))

### begin standard error calculation
# inspired by https://stackoverflow.com/a/71037030
outcome_monkeys <- outcome_monkeys %>% 
  mutate(sd_sq_err_monkeys = map_dbl(data, ~ sd(.x$sq_err_monkeys))) %>%
  mutate(sd_sq_err_baseline = map_dbl(data, ~ sd(.x$sq_err_baseline))) %>%
  mutate(cov_sq_err = map_dbl(data, ~ cov(.x$sq_err_monkeys, .x$sq_err_baseline)))

outcome_monkeys <- outcome_monkeys %>%
  mutate(hat_se_monkeys = 1/sqrt(n) * sd_sq_err_monkeys) %>%
  mutate(hat_se_baseline = 1/sqrt(n) * sd_sq_err_baseline)

# qnorm(0.975) is about 1.96, qnorm(0.025) is about -1.96
outcome_monkeys <- outcome_monkeys %>%
  mutate(lower_monkeys = mean_sq_err_monkeys + qnorm(0.025) * hat_se_monkeys) %>%
  mutate(upper_monkeys = mean_sq_err_monkeys + qnorm(0.975) * hat_se_monkeys) %>%
  mutate(lower_baseline = mean_sq_err_baseline + qnorm(0.025) * hat_se_baseline) %>%
  mutate(upper_baseline = mean_sq_err_baseline + qnorm(0.975) * hat_se_baseline) 

# create ratio and calculate se of ratio (formula from Sarndall et al 2003)
outcome_monkeys <- outcome_monkeys %>%
  mutate(ratio = (mean_sq_err_monkeys / mean_sq_err_baseline)) %>%
  mutate(term1 = 1 / mean_sq_err_baseline^2) %>%
  mutate(term2 = 1 / n) %>%
  mutate(term3 = sd_sq_err_monkeys^2) %>%
  mutate(term4 = ratio^2 * sd_sq_err_baseline^2) %>%
  mutate(term5 = 2 * ratio * cov_sq_err) %>%
  mutate(hat_var_ratio = term1 * term2 * (term3 + term4 - term5))

if (any(outcome_monkeys$hat_var_ratio < 0)) {
  print("These hat_var_ratio are negative, but variance can't be negative")
  outcome_monkeys %>%
    filter(hat_var_ratio < 0) %>%
    print(n = Inf)
  print("If these are just barely negative that could be because of numerical precision issues")
  print("If so, you should probably set these very, very small negative values to 0")
  stop("ERROR: problem with estimated variance of ratio")
}

outcome_monkeys <- outcome_monkeys %>%  
  mutate(hat_se_ratio = sqrt(hat_var_ratio)) %>%
  mutate(lower_ratio = r2_monkeys + qnorm(0.025) * hat_se_ratio) %>%
  mutate(upper_ratio = r2_monkeys + qnorm(0.975) * hat_se_ratio)

if (any(outcome_monkeys$upper_ratio > 1)) stop("ERROR: Some winning pipelines have CI for R^2 that include values bigger than 1")
###
# End: Calculate se for winning pipelines
###

saveRDS(outcome_monkeys, file = file.path(location_for_output, paste0(slug, "_outcome_winners.rds")))

print("Good news! Code completed successfully and is exiting cleanly.")
