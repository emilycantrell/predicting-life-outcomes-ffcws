# This file processes jobs that use automated feature selection.
# Manual feature selection jobs are processed by 01b_process_json_to_rds_for_jobs_with_manual_feature_selection.R.

rm(list = ls())
library(groundhog)
pkgs <- c("tidyverse", "jsonlite", "RcppSimdJson", "furrr")
# jsonlite: read ndjson files 
# RcppSimdJson: parse json strings
# furrr: parallel purrr
groundhog.day <- "2023-06-01"
groundhog.library(pkgs, groundhog.day)

STOP_ON_ERROR <- FALSE # TRUE means stop on error, FALSE means warn
print_width <- 80
options(crayon.enabled = FALSE) # makes it easier capture printed output to files

start <- Sys.time()

#######################################################################
## Background information on file locations and run
#######################################################################

# Directory containing the job directory or directories you generated
# (e.g., if you ran multiple jobs and stored their outputs in the same parent directory).
# This should be the parent directory of the directory named by `slug`.
rootdir <- "/path/to/parent/directory/"

# Name of the specific job directory to process.
slug <- "my_job_name"

# Option to read in subset of machines, done for speed of development
n_machines_to_read <- Inf # set to Inf to read all 
if (n_machines_to_read < Inf) warning(paste("You are only reading results from this many machines:", n_machines_to_read))

print(paste("Processing slug:", slug, ", with option STOP_ON_ERROR:", STOP_ON_ERROR))


# Where should the output file be saved? 
location_for_output <- file.path(rootdir, slug, "parsed")

# Check for output directory, and create if does not exist, more info: https://stackoverflow.com/a/46008094
if (!dir.exists(location_for_output)) {
  dir.create(location_for_output)
} else {
  # directory already exists, so do nothing
  print("location_for_output ready ... ")
}
##############################
# Functions for both results and cv
# get file names to be read
get_filenames <- function(rootdir, exp_prefix) {
  files <- list.files(rootdir, recursive = TRUE)
  cvdata <- paste0(rootdir, files[grepl(sprintf("*%s.*cv\\.json", exp_prefix), files)])
  resdata <- paste0(rootdir, files[grepl(sprintf("*%s.*results\\.json", exp_prefix), files)])
  jobdata <- paste0(rootdir, files[grepl(sprintf("*%s.*jobdesc\\.json", exp_prefix), files)])
  matshdata <- paste0(rootdir, files[grepl(sprintf("*%s.*matrix_shapes\\.json", exp_prefix), files)])
  errdata <- paste0(rootdir, files[grepl(sprintf("*%s.*errors\\.txt", exp_prefix), files)])
  list("result_files"= resdata, "cv_files"= cvdata, "job_files"= jobdata, "error_files" = errdata, "matrix_shapes" = matshdata)
}

# function to read ndjson
json_stream_reader <- function(filename, VERBOSE=FALSE) {
  if (VERBOSE) print(filename)
  con <- file(filename, open="r")
  df <- jsonlite::stream_in(con, verbose = FALSE) # turn off progress report
  on.exit(close(con)) # https://stackoverflow.com/a/6304163
  as_tibble(df) %>%
    return()
}

append_machine_number <- function(tib, .y) {
  # machine num passed in as .y in the imap() call below
  tib %>%
    mutate(machine = .y) %>%
    return()
}
#######################################################################
## Read the results and cv files, and rename variables
######################################################################
# Get the file names
files <- get_filenames(rootdir, slug)

# sample first n machines
files <- map(files, ~ head(., n = n_machines_to_read))

### Check errors.txt files to ensure they are all empty.
ensure_error_txt_empty <- function(filename) {
  if (file.size(filename) != 0) {
    print(paste("Problem file:", filename))
    stop(paste("ERROR: file errors.txt not empty"), filename)
  } else {
    return(TRUE)
  }
}
pass_errors_check <- map(files$error_files, ensure_error_txt_empty)
if (all(pass_errors_check != TRUE)) stop("ERROR: some errors.txt not empty")

#### Results file
# Make a list with the results data (each element of list is the results file from 1 machine)

print(paste("Beginning reading results json files:", Sys.time()))
plan(multisession)
results_list <- future_map(pluck(files, "result_files"), json_stream_reader, 
                           .options = furrr_options(seed = 08540))
plan(sequential)
print(paste("Ending reading results json files:", Sys.time()))

results_list <- imap(results_list, append_machine_number)

# convert list of tibbles to single tibble
results <- map_dfr(results_list, ~ .)
if (nrow(results) != map_dbl(results_list, nrow)  %>% sum()) stop("ERROR: something went wrong going from results_list to results")
rm(results_list) # clean-up

# rename results
results$task_id <- results$`_id`$`$oid`
results$`_id` <- NULL
results <- results %>%
  rename("outerfold" = "index",
         "outcome_outerfold_id" = "dataid",
         "task_model_training_time" = "pipeline_model_time",
         "metafeature_id" = "metafeatureid") %>%
  relocate("task_id") # move to first column

# save original variables names to be used later for testing
initial_variables_results <- names(results)
processed_variables_results <- c()

if ("error" %in% names(results)) {
  info_str <- "ISSUE: some tasks have error"
  print(info_str)
  results %>% 
    filter(!is.na(error)) %>%
    select(task_id, error) %>%
    print(width = print_width)

  results %>%
    filter(!is.na(error)) %>%
    select(task_id, outcome, scaler, model, outerfold, error) %>%
    print(width = 80)
  ifelse(STOP_ON_ERROR, stop(info_str), warning(info_str)) 
}

###### CV file
# Make a list with the cv data (each element of list is the cv file from 1 machine)
cv_list <- map(pluck(files, "cv_files"), json_stream_reader)

# append machine number
cv_list <- imap(cv_list, append_machine_number)

# check for NaN in skewness/kurtosis
explore_NaNs <- function(tib) {
  summary_stats <- NULL
  summary_stats["machine"] <- unique(tib$machine)
  summary_stats["skewness_train_nan"] <- sum(str_detect(unlist(tib$ystats$skewness$train), "NaN"))
  summary_stats["skewness_test_nan"] <- sum(str_detect(unlist(tib$ystats$skewness$test), "NaN"))
  summary_stats["kurtosis_train_nan"] <- sum(str_detect(unlist(tib$ystats$kurtosis$train), "NaN"))
  summary_stats["kurtosis_test_nan"] <- sum(str_detect(unlist(tib$ystats$kurtosis$test), "NaN"))
  return(summary_stats)
}
skew_kurt_check <- map_dfr(cv_list, explore_NaNs)

if (sum(c(skew_kurt_check$skewness_train_nan, skew_kurt_check$skewness_test_nan,
          skew_kurt_check$kurtosis_train_nan, skew_kurt_check$kurtosis_test_nan)) != 0) {
  stop("ERROR: there are NaN in skewness or kurtosis")
}
rm(skew_kurt_check) # cleanup

# convert list of tibbles to single tibble
cv <- map_dfr(cv_list, ~ .)

if (nrow(cv) != map_dbl(cv_list, nrow)  %>% sum()) stop("ERROR: something went wrong going from cv_list to cv")
rm(cv_list)

# rename cv
cv$outcome_outerfold_id <- cv$`_id`$`$oid`
cv$`_id` <- NULL
cv$jobfile_id <- cv$confid$`$oid`
cv$confid <- NULL
cv <- cv %>% 
    rename("outerfold_id" = "cvid") %>%
    relocate("outcome_outerfold_id") # move to first column

# save original variables names to be used later for testing
initial_variables_cv <- names(cv)
processed_variables_cv <- c()

##################
## drop any outcome that has 1 or more, 0 variance outer or inner folds
## these outcomes are below the "resolution limit"
##################
# hoist the variance of outerfold train and test
cv$outerfold_train_var <- cv$ystats$variance$train # hoisting this variable out
cv$outerfold_test_var <- cv$ystats$variance$test # hoisting this variable out

# parse the innerfold train and test mean and variance
parse_hyperparameter_tuning_cv_folds <- function(single_innerfold_json) {
  if (!(is.character(single_innerfold_json))) stop("ERROR: single_innerfold_json not as expected")
  if (length(single_innerfold_json) != 1) stop("ERROR: single_innerfold_json not as expected")

  # Note this function has 2 return statements
  entry_names <- c(
    "innerfold_0_train_mean", "innerfold_0_train_var",
    "innerfold_0_test_mean", "innerfold_0_test_var",
    "innerfold_1_train_mean", "innerfold_1_train_var",
    "innerfold_1_test_mean", "innerfold_1_test_var",
    "innerfold_2_train_mean", "innerfold_2_train_var",
    "innerfold_2_test_mean", "innerfold_2_test_var",
    "innerfold_3_train_mean","innerfold_3_train_var",
    "innerfold_3_test_mean", "innerfold_3_test_var",
    "innerfold_4_train_mean","innerfold_4_train_var",
    "innerfold_4_test_mean", "innerfold_4_test_var")
  if (is.na(single_innerfold_json)) {
    NA_vector <- rep(NA, length(entry_names))
    names(NA_vector) <- entry_names
    return(NA_vector)
  } else {
    result <- single_innerfold_json %>%
      RcppSimdJson::fparse() %>%
      unlist() 
    names(result) <- entry_names
    return(result)
  }
}

results <- map_dfr(results$hyperparameter_tuning_cvfolds, parse_hyperparameter_tuning_cv_folds) %>%
  bind_cols(results, .)
results$hyperparameter_tuning_cvfolds <- NULL #cleanup
processed_variables_results <- c(processed_variables_results, "hyperparameter_tuning_cvfolds") 

# innerfold train 0 variance
results <- results %>%
  mutate(zero_var_innerfold_train = if_else(
    ((innerfold_0_train_var == 0) | (innerfold_1_train_var == 0) |
       (innerfold_2_train_var == 0) | (innerfold_3_train_var == 0) | 
       (innerfold_4_train_var == 0)),
    TRUE, FALSE, missing = FALSE))

# innerfold test 0 variance
results <- results %>%
  mutate(zero_var_innerfold_test = if_else(
    ((innerfold_0_test_var == 0) | (innerfold_1_test_var == 0) | 
       (innerfold_2_test_var == 0) | (innerfold_3_test_var == 0) |
       (innerfold_4_test_var == 0)),
    TRUE, FALSE, missing = FALSE))

results <- results %>%
  mutate(zero_var_innerfold = zero_var_innerfold_train | zero_var_innerfold_test) 

outcomes_zero_var_innerfold <- results %>%
  filter(zero_var_innerfold == TRUE) %>% 
  select(outcome) %>% 
  unique() %>%
  arrange(outcome) 

outcomes_zero_var_outerfold <- cv %>%
  filter(outerfold_train_var == 0 | outerfold_test_var == 0) %>%
  select(outcome) %>%
  unique() %>%
  arrange(outcome) 

# compare and combine outcomes with zero variance folds
# test
if (nrow(anti_join(outcomes_zero_var_outerfold, outcomes_zero_var_outerfold, by = "outcome")) != 0) {
  info_str <- "Some outcomes with zero var outerfold don't have zero var innerfold"
  print(info_str)
  print(paste("outcomes_zero_var_outerfold:", outcomes_zero_var_outerfold))
  print(paste("outcomes_zero_var_innerfold:", outcomes_zero_var_innerfold))
  ifelse(STOP_ON_ERROR, stop(info_str), warning(info_str)) 
}

# combine 
outcomes_below_resolution_limit <- union(outcomes_zero_var_innerfold, outcomes_zero_var_outerfold) %>%
  mutate(row_id = row_number()) %>%
  relocate(row_id)
if (nrow(outcomes_below_resolution_limit) == 0) {
  # no variables below resolution limit, create a placeholder file
  outcomes_below_resolution_limit <- tibble(id_num = NA, 
                                            outcome = "No outcomes below resolution limit")
}

# write
write_csv(x = outcomes_below_resolution_limit, 
          file = file.path(location_for_output, paste0(slug, "_outcomes_below_resolution_limit.csv")),
          append = FALSE)

# drop outcomes below the resolution limit
remove_outcomes_below_resolution_limit <- function(tib, outcomes_to_drop, VERBOSE = FALSE) {
  if (VERBOSE) print(paste("Dropping outcomes below resolution limit from", deparse(substitute(tib))))
  nrow_before <- nrow(tib)
  tib <- tib %>%
    filter(!(outcome %in% outcomes_to_drop))
  nrow_after <- nrow(tib)  
  if (VERBOSE) print(paste("Rows dropped:", nrow_before - nrow_after)) 
  if (nrow_before < nrow_after) stop("ERROR: nrow_after bigger than nrow_before")
  return(tib)
}

if (any(outcomes_below_resolution_limit$outcome == "No outcomes below resolution limit")) {
  # nothing to drop 
} else {
  cv <- remove_outcomes_below_resolution_limit(cv, outcomes_below_resolution_limit$outcome, VERBOSE = TRUE)
  results <- remove_outcomes_below_resolution_limit(results, outcomes_below_resolution_limit$outcome, VERBOSE = TRUE)
}

# test: after dropping outcome below resolution limit, number of rows in results should be multiple of number of outer folds
n_outerfolds <- results %>% 
  pull(outerfold) %>% 
  n_distinct()
if (nrow(results) %% n_outerfolds != 0) {
  info_str <- "Unexpected number of rows in results (relative to number of outerfolds) after dropping outcomes below resolution threshold"
  ifelse(STOP_ON_ERROR, stop(info_str), warning(info_str)) 
}

invisible(gc)
cv$outerfold_train_var  <- NULL # cleanup
cv$outerfold_test_var <- NULL # cleanup
#################

########################################
print("processing results starting. . . .")
print(Sys.time())

#####
# predictions and predictions_raw 
# we are not parsing predictions_raw because we don't need them now
results$predictions_raw <- NULL # don't need these now
processed_variables_results <- c(processed_variables_results, "predictions_raw")

print("beginning processing predictions . . . ")
if (any(is.na(results$predictions))) {
  info_str <- "ISSUE: predictions is NA"
  print(info_str)
  results %>% 
    filter(is.na(predictions)) %>%
    select(task_id, predictions) %>%
    print(width = print_width)
  
  results %>% 
    filter(is.na(predictions)) %>% 
    select(task_id, outcome, outerfold, scaler, model) %>%
    print(width = print_width)
  ifelse(STOP_ON_ERROR, stop(info_str), warning(info_str)) 
}

tidy_predictions <- function(predictions_json, RUN.CHECKS = FALSE) {
  # this function was optimized on 2023-01-19
  output <- predictions_json %>%
    RcppSimdJson::fparse() %>% 
    unlist() %>%
    enframe(name = "ffcws_id", value = "prediction")

  if (RUN.CHECKS) {
    if (!is_tibble(output)) stop("ERROR: output is not tibble")
    if (ncol(output) != 2) stop("ERROR: output does not have two columns")
  } 
  return(output)
}

predictions_processed <- map(results$predictions, tidy_predictions)

# make sure that we don't have the same number of predictions for every task
num_predictions_per_task <- map_int(predictions_processed, nrow)
if (length(unique(num_predictions_per_task)) == 1) {
  stop("ERROR: all tasks have the same number of predictions")
}
rm(num_predictions_per_task)
predictions_processed_tib <- tibble(task_id = results$task_id, 
                                    predictions = predictions_processed)
rm(predictions_processed) # cleanup
saveRDS(predictions_processed_tib, file = file.path(location_for_output, paste0(slug, "_predictions.rds")))
rm(predictions_processed_tib) # cleanup
results$predictions <- NULL # cleanup
processed_variables_results <- c(processed_variables_results, "predictions") 
print("finished processing predictions . . . ")

# hyperparameter_tuning 
print(paste("Beginning simplify hyperparameters:", Sys.time()))
parse_simplify_hyperparameter_tuning <- function(single_object) {
  # note there are two returns
  if (is.na(single_object)) {
    return(NA)
  } else {
    single_object %>%
      RcppSimdJson::fparse() %>%
      # _r2_score gets hyperparameter_tuning_fold_[innerfold]_r2_score (note no trailing _)
      # _tse_ gets hyperparameter_tuning_fold_[innerfold]_tse_baseline and hyperparameter_tuning_fold_[innerfold]_tse_predictions
      keep(str_detect(names(.), "_r2_score|_tse_|_param_")) %>%
      enframe() %>%
      return()
     
      # There is more we could parse, and here are some things we used to parse but stopped:
        # _best_ gets the information need to make something like Fig 2 in Bergstra and Bengio (2012), but I don't think (not sure) we are producing a weighted average like Eq 5 & 6
        # _holdout_ gets the performance information for the selected hyperparameters on this holdout fold
  }
}

plan(sequential) 
# Note: for smaller jobs, you may be able to use plan(multisession, workers = 2) instead of plan(sequential) here; 
# if you make that switch, you should still keep the plan(sequential) after the future_map() call
hyperparameter_tuning_simplified <- future_map(results$hyperparameter_tuning, parse_simplify_hyperparameter_tuning, 
                                               .options = furrr_options(seed = 08540))
plan(sequential)
print(paste("Ending simplify hyperparameters:", Sys.time()))

hyperparameter_tuning_simplified_tib <- tibble(task_id = results$task_id, hyperparameter_tuning = hyperparameter_tuning_simplified)
rm(hyperparameter_tuning_simplified) # cleanup
saveRDS(hyperparameter_tuning_simplified_tib, file = file.path(location_for_output, paste0(slug, "_hyperparameter_tuning.rds")))
rm(hyperparameter_tuning_simplified_tib) # cleanup 
results$hyperparameter_tuning <- NULL # cleanup
processed_variables_results <- c(processed_variables_results, "hyperparameter_tuning") 
print("finished processing hyperparameters . . .")
#####

# feature_selection 
fs_columns <- map_dfr(results$feature_selection, 
                      as_tibble_row, 
                      .name_repair = ~paste0("fs", seq(.x)))

if (ncol(fs_columns) != 3) stop("ERROR: fs_columns is different than expected. This could be OK but code is not robust enough yet.")

results <- bind_cols(results, fs_columns) %>%
  select(-feature_selection)
rm(fs_columns) # cleanup
processed_variables_results <- c(processed_variables_results, "feature_selection") 
#####

# feature_selection_stats contains information about selected, remaining, and dropped features
process_task_features <- function(task_features) {
  if (!(is.matrix(task_features))) stop("ERROR: task_features not as expected")
  if ((ncol(task_features)) != 2) stop("ERROR: task_features not as expected")
  
  colnames(task_features) <- c("predictor", "feature_selection_metric")
  as_tibble(task_features) %>%
    return()
}

selected_features <- pluck(results, "feature_selection_stats", "selected_features") %>%
  map(., process_task_features)
selected_features_tib <- tibble(task_id = results$task_id, selected_features = selected_features)
rm(selected_features) # cleanup
saveRDS(selected_features_tib, file = file.path(location_for_output, paste0(slug, "_selected_features.rds")))
rm(selected_features_tib) 

remaining_features <- pluck(results, "feature_selection_stats", "remaining_features") %>%
  map(., process_task_features)
remaining_features_tib <- tibble(task_id = results$task_id, remaining_features = remaining_features)
rm(remaining_features) # cleanup
saveRDS(remaining_features_tib, file = file.path(location_for_output, paste0(slug, "_remaining_features.rds")))
rm(remaining_features_tib) # cleanup

process_dropped_features <- function(task_dropped_features) {
  # Note two return statements
  if (length(task_dropped_features) == 0) {
    # empty list; there were no dropped features because of collinearity
    return(task_dropped_features)
  } else {
    colnames(task_dropped_features) <- c("predictor", "value")
    as_tibble(task_dropped_features) %>%
    return()
  }
}

dropped_features <- pluck(results, "feature_selection_stats", "dropped_features") %>%
    map(., process_dropped_features)

dropped_features_tib <- tibble(task_id = results$task_id, dropped_features = dropped_features)
rm(dropped_features) # cleanup
saveRDS(dropped_features_tib, file = file.path(location_for_output, paste0(slug, "_dropped_features.rds")))
rm(dropped_features_tib) # cleanup

results$feature_selection_stats <- NULL # cleanup 
processed_variables_results <- c(processed_variables_results, "feature_selection_stats") 
#####
processed_variables_results <- c(processed_variables_results, "task_id") # no processing needed
processed_variables_results <- c(processed_variables_results, "scaler") # no processing needed
processed_variables_results <- c(processed_variables_results, "outcome") # no processing needed
processed_variables_results <- c(processed_variables_results, "outerfold") #  no processing needed
processed_variables_results <- c(processed_variables_results, "outcome_outerfold_id") # no processing needed
processed_variables_results <- c(processed_variables_results, "features") # no processing needed yet, might in the future
processed_variables_results <- c(processed_variables_results, "failed") # no processing needed
processed_variables_results <- c(processed_variables_results, "model") # no processing needed
processed_variables_results <- c(processed_variables_results, "is_tuned") # no processing needed
#####
# model_train_stats
results <- bind_cols(results, (results$model_train_stats)) %>%
  select(-model_train_stats)

processed_variables_results <- c(processed_variables_results, "model_train_stats")
#####
# scores
results <- bind_cols(results, (results$scores)) %>%
  select(-scores)
processed_variables_results <- c(processed_variables_results, "scores") 
#####
processed_variables_results <- c(processed_variables_results, "task_model_training_time") # no processing needed 
processed_variables_results <- c(processed_variables_results, "metafeature_id") # no processing needed 
processed_variables_results <- c(processed_variables_results, "machine") # no parsing needed
#####
# the next two variables (error and stack_trace) only appear as needed
# so some .json files have them and some don't
# neither needs to be parsed, but we do need to track them in the processed_variables_results vector
if ("error" %in% names(results)) processed_variables_results <- c(processed_variables_results, "error") # no parsing needed
if ("stack_trace" %in% names(results)) processed_variables_results <- c(processed_variables_results, "stack_trace") # no parsing needed
####

# check to make sure that we parsed everything
if (!(isTRUE(all.equal(sort(initial_variables_results), sort(processed_variables_results))))) {
  print(initial_variables_results)
  print(processed_variables_results)
  base::setdiff(initial_variables_results, processed_variables_results)
  base::setdiff(processed_variables_results, initial_variables_results)
  warning("Warning: not all original variables have been processed. This is expected behavior because we do not process hyperparameter_tuning_fs_info. If you want hyperparameter_tuning_fs_info, please edit the parsing code.")
}
rm(processed_variables_results)

# add pipeline_id
results <- results %>%
  group_by(scaler, fs1, fs2, fs3, model) %>%
  mutate(pipeline_id = cur_group_id()) %>%
  relocate(pipeline_id, .before = scaler) %>%
  ungroup()

invisible(gc()) # clean-up
print(Sys.time())
print("processing results complete. . . .")
########################
# CV file
print(paste("Begin processing cv:", Sys.time()))
processed_variables_cv <- c(processed_variables_cv, "outcome_outerfold_id") # no processing needed
processed_variables_cv <- c(processed_variables_cv, "outerfold_id") # no processing needed
processed_variables_cv <- c(processed_variables_cv, "jobfile_id") # no processing needed
processed_variables_cv <- c(processed_variables_cv, "outcome") # no processing needed

# ystats
cv_ystats_tib <- cv$ystats %>%
  unnest_wider(col = everything(), names_sep = "_")
cv <- bind_cols(cv, cv_ystats_tib)
rm(cv_ystats_tib) # cleanup
cv$ystats <- NULL # cleanup
processed_variables_cv <- c(processed_variables_cv, "ystats")

# train_outcomes and test_outcomes
parse_outcomes <- function(outcomes_single_JSON) {
  if (!is.character(outcomes_single_JSON)) stop("ERROR: outcome_single_JSON not as expected")
  
  output <- outcomes_single_JSON %>%
    RcppSimdJson::fparse() %>%
    enframe(. , name = "ffcws_id", value = "outcome_value") %>%
    unnest(., outcome_value) 
  
  if (!is_tibble(output)) stop("ERROR: output not as expected")
  if (ncol(output) != 2) stop("ERROR: output not as expected")
  return(output)
}

train_outcomes <- map(cv$train_outcomes, parse_outcomes)
train_outcomes_tib <- tibble(outcome_outerfold_id = cv$outcome_outerfold_id, train_outcomes) 
rm(train_outcomes) # cleanup
saveRDS(train_outcomes_tib, file = file.path(location_for_output, paste0(slug, "_cv_train_outcomes.rds")))
rm(train_outcomes_tib) # cleanup
cv$train_outcomes <- NULL # cleanup
processed_variables_cv <- c(processed_variables_cv, "train_outcomes") 

# test_outcomes
test_outcomes <- map(cv$test_outcomes, parse_outcomes)
test_outcomes_tib <- tibble(outcome_outerfold_id = cv$outcome_outerfold_id, test_outcomes) 
rm(test_outcomes)
saveRDS(test_outcomes_tib, file = file.path(location_for_output, paste0(slug, "_cv_test_outcomes.rds")))
rm(test_outcomes_tib)
cv$test_outcomes <- NULL # cleanup
processed_variables_cv <- c(processed_variables_cv, "test_outcomes") 
print(paste("End processing cv:", Sys.time()))
###########

### Merge
results_merged <- left_join(results, cv, 
                             by=c("outcome_outerfold_id" = "outcome_outerfold_id", 
                                  "machine" = "machine", 
                                  "outcome" = "outcome"), 
                             suffix = c("_results", "_cv"))

# Tests of results_merged
if (nrow(results_merged) != nrow(results)) stop("ERROR: problem in left_join")

saveRDS(results_merged, file = file.path(location_for_output, paste0(slug, "_results_cv_merged.rds")))
rm(results_merged)

saveRDS(results, file = file.path(location_for_output, paste0(slug, "_results.rds")))
rm(results)

saveRDS(cv, file = file.path(location_for_output, paste0(slug, "_cv.rds")))
rm(cv)

##############################
# Process matrix_shapes.json
##############################
print(paste("Begin processing matrix shapes:", Sys.time()))

# Read and prep shape files
read_prep_json <- function(shapes_filename) {
  js <- fromJSON(shapes_filename)
  
  id_oid <- js[['_id']]
  js[['_id']] <- NULL
  
  df <- as.data.frame(matrix(unlist(js, recursive = T), ncol = 8, byrow = F))
  colnames(df) <- names(js)
  
  df <- df %>%
    as_tibble %>%
    mutate(global_stepid = as.numeric(global_stepid), 
           group_stepid = as.numeric(group_stepid),
           ncols = as.numeric(ncols),
           nrows = as.numeric(nrows),
           time = as_datetime(time))
  return(df)
}

machines_steps_list <- map(files$matrix_shapes, read_prep_json)
machines_steps_list <- imap(machines_steps_list, append_machine_number)
# convert list of tibbles to single tibble
machines_steps <- map_dfr(machines_steps_list, ~ .)
rm(machines_steps_list)

saveRDS(machines_steps, file = file.path(location_for_output, paste0(slug, "_matrix_shapes.rds")))
rm(machines_steps)
print(paste("End processing matrix shapes:", Sys.time()))

stop <- Sys.time()
print(paste("Estimate of overall runtime:", stop - start, 2))
print("Finished processing json files. Code exiting normally.")

