# This file produces CSV summaries of timing.json files found under a specified slug.

# Parse timing.json files across mm-worker-* directories
# This version uses groundhog for package management and writes outputs
# into a parsed/timing_parsed directory inside the selected slug.

rm(list = ls())

library(groundhog)
pkgs <- c("jsonlite", "tidyverse")
groundhog.day <- "2023-06-01"
groundhog.library(pkgs, groundhog.day)

# Directory containing the job directory or directories you generated
# (e.g., if you ran multiple jobs and stored their outputs in the same parent directory).
# This should be the parent directory of the directory named by `slug`.
rootdir <- "/path/to/parent/directory/"

# Name of the specific job directory to process.
slug <- "my_job_name"

job_dir <- file.path(rootdir, slug)

# Find timing.json files in directories named mm-worker-<number>
timing_files <- list.files(
  path = job_dir,
  pattern = "^timing\\.json$",
  recursive = TRUE,
  full.names = TRUE
)

timing_files <- timing_files[
  grepl(sprintf("%smm-worker-[0-9]+%stiming\\.json$", .Platform$file.sep, .Platform$file.sep), timing_files)
]

if (length(timing_files) == 0) {
  stop("No timing.json files found under: ", job_dir)
}

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

extract_machine_number <- function(path) {
  out <- stringr::str_match(path, "mm-worker-([0-9]+)")[, 2]
  as.integer(out)
}

parse_one_timing_file <- function(path) {
  x <- jsonlite::fromJSON(path, simplifyVector = FALSE)

  machine <- extract_machine_number(path)

  # Drop non-timing metadata fields
  step_names <- setdiff(names(x), c("_id", "conf_id"))

  if (length(step_names) == 0) {
    return(tibble::tibble(
      machine = machine,
      step = character(),
      count = numeric(),
      min_seconds = numeric(),
      max_seconds = numeric(),
      avg_seconds = numeric(),
      total_seconds = numeric(),
      file = character()
    ))
  }

  purrr::map_dfr(step_names, function(step_name) {
    item <- x[[step_name]]

    tibble::tibble(
      machine = machine,
      step = step_name,
      count = as.numeric(item$count %||% NA),
      min_seconds = as.numeric(item$min %||% NA),
      max_seconds = as.numeric(item$max %||% NA),
      avg_seconds = as.numeric(item$avg %||% NA),
      total_seconds = as.numeric((item$count %||% NA) * (item$avg %||% NA)),
      file = path
    )
  })
}

timing_long <- purrr::map_dfr(timing_files, parse_one_timing_file) %>%
  dplyr::arrange(machine, step)

machine_summary <- timing_long %>%
  dplyr::group_by(machine) %>%
  dplyr::summarise(
    n_steps = dplyr::n(),
    total_runtime_seconds = sum(total_seconds, na.rm = TRUE),
    total_runtime_minutes = total_runtime_seconds / 60,
    total_runtime_hours = total_runtime_seconds / 3600,
    longest_single_step_seconds = max(max_seconds, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(machine)

step_summary <- timing_long %>%
  dplyr::group_by(step) %>%
  dplyr::summarise(
    machines_present = dplyr::n_distinct(machine),
    total_count = sum(count, na.rm = TRUE),
    total_runtime_seconds = sum(total_seconds, na.rm = TRUE),
    total_runtime_minutes = total_runtime_seconds / 60,
    total_runtime_hours = total_runtime_seconds / 3600,
    mean_runtime_per_machine_seconds = mean(total_seconds, na.rm = TRUE),
    median_runtime_per_machine_seconds = stats::median(total_seconds, na.rm = TRUE),
    max_single_observed_call_seconds = max(max_seconds, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(dplyr::desc(total_runtime_seconds))

overall_summary <- tibble::tibble(
  slug = slug,
  machines_found = dplyr::n_distinct(timing_long$machine),
  timing_files_found = length(timing_files),
  summed_runtime_seconds = sum(machine_summary$total_runtime_seconds, na.rm = TRUE),
  summed_runtime_minutes = summed_runtime_seconds / 60,
  summed_runtime_hours = summed_runtime_seconds / 3600,
  wall_clock_proxy_seconds = max(machine_summary$total_runtime_seconds, na.rm = TRUE),
  wall_clock_proxy_minutes = wall_clock_proxy_seconds / 60,
  wall_clock_proxy_hours = wall_clock_proxy_seconds / 3600
)

machine_by_step <- timing_long %>%
  dplyr::select(machine, step, total_seconds) %>%
  dplyr::mutate(step = make.names(step)) %>%
  tidyr::pivot_wider(
    names_from = step,
    values_from = total_seconds
  ) %>%
  dplyr::arrange(machine)

cat("\n====================\n")
cat("OVERALL SUMMARY\n")
cat("====================\n")
print(overall_summary, width = Inf)

cat("\n====================\n")
cat("RUNTIME PER MACHINE\n")
cat("====================\n")
print(machine_summary, n = Inf, width = Inf)

cat("\n====================\n")
cat("RUNTIME PER STEP ACROSS MACHINES\n")
cat("====================\n")
print(step_summary, n = Inf, width = Inf)

# Write outputs to: {slug}/parsed/timing_parsed/
output_dir <- file.path(job_dir, "parsed", "timing_parsed")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

readr::write_csv(timing_long, file.path(output_dir, paste0(slug, "_timing_long.csv")))
readr::write_csv(machine_summary, file.path(output_dir, paste0(slug, "_timing_machine_summary.csv")))
readr::write_csv(step_summary, file.path(output_dir, paste0(slug, "_timing_step_summary.csv")))
readr::write_csv(machine_by_step, file.path(output_dir, paste0(slug, "_timing_machine_by_step.csv")))
readr::write_csv(overall_summary, file.path(output_dir, paste0(slug, "_timing_overall_summary.csv")))

cat("\nWrote output files to:\n", output_dir, "\n")
