# This file generates plots for the paper and appendices, plus a CSV file with results and metadata for all outcomes.
# Launch this file from the GitHub repository to ensure that "here()" points to the correct root directory.

################################################################################################ 
############### USER SHOULD EDIT THESE ENTRIES TO YOUR DESIRED INPUTS ##########################  

# Optional version-control label for plots.
# This can be any string and does not need to match the job directory name.
slug <- "v0.2.8 Jan. 2026"

# Name of the directory containing the main job for the paper (winning-pipeline and benchmark results).
name_of_main_job <- "v0.2.8_2026-01-26_full_job_with_baseline"

# Name of the directory containing the job with the two-feature models for the height outcomes.
name_of_two_feature_model_job <- "v0.2.8_2026-02-20_height_outcomes_2_manually_selected_features_percentile_feature"

# Parent directory containing the job directory or directories above.
# For example, if multiple jobs are stored in the same location, specify that shared parent directory here.
root_dir_for_results <- "/path/to/parent/directory/"

# Label for output files.
file_path_label <- format(Sys.Date(), "%Y-%m-%d") 
################################################################################################  
################################################################################################ 


#############################################################################
############################ DATA PREPARATION ###############################
#############################################################################

# Load packages
library(groundhog)
pkgs <- c("tidyverse", "ggmosaic", "readxl", "knitr", "kableExtra", "stringr", "here", "patchwork")
groundhog.day <- "2024-04-23"
groundhog.library(pkgs, groundhog.day)

# Create directory where output will be saved
fig_dir <- here::here("analysis/figures_and_analyses_for_paper/figures_and_tables", file_path_label)
if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

# Load results for winning pipelines
winners_path <- file.path(
  root_dir_for_results,
  name_of_main_job,
  "parsed",
  paste0(name_of_main_job, "_outcome_winners.rds")
)
winners <- readRDS(winners_path)

# Load results for simple benchmark
benchmark_path <- file.path(
  root_dir_for_results,
  name_of_main_job,
  "parsed",
  paste0(name_of_main_job, "_simple_benchmark.rds")
)
benchmarks <- readRDS(benchmark_path)

# Load results for the two-feature models for height outcomes
two_feature_model_path <- file.path(
  root_dir_for_results,
  name_of_two_feature_model_job,
  "parsed",
  paste0(name_of_two_feature_model_job, "_outcome_winners.rds")
)
two_feature_models <- readRDS(two_feature_model_path)

# Load metadata
metadata_dir <- here::here("analysis", "metadata_for_analysis")
meta <- read_csv(file.path(metadata_dir, "monkeys_metadata_v09.csv"),
                 guess_max = Inf, show_col_types = FALSE)
measured_earlier <- read_csv(file.path(metadata_dir, "same_construct_previously_measured_FINAL.csv"),
                             guess_max = Inf, show_col_types = FALSE)

# Remove unnecessary columns
winners <- winners %>%
  select(-c("data", "monkeys_ids"))
gc()

# Merge metadata with results
winners <- left_join(winners, meta,
                     by = c("outcome" = "new_name"))
measured_earlier <- measured_earlier %>%
  rename(measured_at_earlier_wave = `Was the same construct measured at an earlier wave (prior to dropping features based on non-modal value threshold)?`) %>%
  select(outcome_name, measured_at_earlier_wave)
winners <- left_join(winners, measured_earlier,
                     by = c("outcome" = "outcome_name"))

# Drop outcomes that meet our exclusion criteria
winners <- winners %>%
  # Drop outcomes that have imputation from earlier waves
  filter(!(outcome %in% c("cp6hhinc", "cp6povco", "cp6povca", "cp6edu"))) %>%
  # Drop z-scores
  filter(!(outcome %in% c("ch6bmiz", "ch6haz", "ch6waz", "ck6bmiz", "ck6haz", "ck6waz")))

# Create function for saving plots
save_plot <- function(name, width, height) {
  ggsave(
    filename = file.path(fig_dir, paste0(name, "_", file_path_label, ".pdf")),
    width = width,
    height = height,
    device = cairo_pdf
  )
}

# Clarify labels for anthropometric outcomes that are available in
# both an in-home measured version and a youth self-report version.
#
# "h" outcomes are based on measurements taken during the in-home study.
# "k" outcomes are based on the youth's self-reported height and weight.
clarify_anthropometric_varlab <- function(outcome, varlab) {
  case_when(
    outcome %in% c(
      "ch6cbmi",
      "ch6bmip",
      "ch6hap",
      "ch6wap"
    ) ~ paste0(varlab, " (from in-home measurement)"),
    
    outcome %in% c(
      "ck6cbmi",
      "ck6bmip",
      "ck6hap",
      "ck6wap"
    ) ~ paste0(varlab, " (from youth's self-report)"),
    
    TRUE ~ varlab
  )
}


# Helper function for converting straight double quotes to proper
# opening and closing quotation marks.
#
# For example:
#   Theme - "Adolescent Social Skills" and "Adaptive Social Behavior Inventory"
#
# becomes:
#   Theme - “Adolescent Social Skills” and “Adaptive Social Behavior Inventory”
clean_quotes <- function(x) {
  str_replace_all(
    x,
    '"([^"]*)"',
    '“\\1”'
  )
}


# Apply substantive cleanup to variable labels used in reader-facing outputs.
#
# These cleanups are applied to both the results-and-metadata CSV and
# paper tables. Typography or LaTeX-specific changes (such as smart quotes,
# em dashes, and comparison-symbol rendering) are handled separately so
# that they do not alter the CSV text.
clean_varlab_for_output <- function(outcome, varlab) {
  
  # Correct a small number of truncated or misspelled variable labels.
  cleaned_varlab <- case_when(
    outcome == "p6j46" ~
      "Someone in household did not see doctor/go to hospital because of cost",
    
    outcome == "k6d61i" ~
      "Used or threatened to use a weapon to get something",
    
    outcome == "o6c11" ~
      "Interior - Youth's artwork or photographs displayed",
    
    outcome == "p6b52" ~
      "Youth is nervous, high-strung, or tense",
    
    TRUE ~ varlab
  )
  
  # Clarify whether anthropometric outcomes are based on in-home
  # measurements or youth self-report.
  cleaned_varlab <- clarify_anthropometric_varlab(
    outcome,
    cleaned_varlab
  )
  
  cleaned_varlab %>%
    
    # Spell out PCG in reader-facing labels.
    str_replace_all("\\bPCG\\b", "primary caregiver") %>%
    
    # Capitalize "Primary caregiver" at the beginning of a label
    # or immediately after a spaced dash.
    str_replace_all("^primary caregiver", "Primary caregiver") %>%
    str_replace_all("(?<= - )primary caregiver", "Primary caregiver") %>%
    
    # Standardize capitalization of scale titles.
    str_replace_all(
      regex("primary caregiver perception", ignore_case = TRUE),
      "Primary Caregiver Perception"
    ) %>%
    str_replace_all(
      c(
        "Child-perceived parental monitoring" = "Child-Perceived Parental Monitoring",
        "Primary caregiver-perceived parental monitoring" = "Primary Caregiver-Perceived Parental Monitoring"
      )
    ) %>%
    
    # Standardize capitalization in reader-facing labels.
    str_replace_all("\\bFocal Child\\b", "focal child") %>%
    str_replace_all("\\bAutism\\b", "autism") %>%
    
    # Spell out abbreviations.
    str_replace_all("\\bint\\b", "interview") %>%
    str_replace_all("\\bNum\\.\\s*", "Number ") %>%
    
    # Correct possessive in youth skin-color label.
    str_replace_all("Youth skin", "Youth's skin") %>%
    
    # Correct hyphenation in reader-facing label.
    str_replace_all(
      "Youth takes doctor prescribed medication\\?",
      "Youth takes doctor-prescribed medication?"
    ) %>%
    
    # BMI is defined in the abbreviation list accompanying the table,
    # so use the acronym alone in the variable label.
    str_replace_all(
      fixed("Body Mass Index (BMI)"),
      "BMI"
    ) %>%
    
    # Remove survey-question prefixes such as
    # "B31a. ", "I30. ", and "E17_1. ".
    str_remove(
      "^[A-Z]+[0-9]+(?:_[0-9]+|[A-Za-z]*)?\\.\\s*"
    ) %>%
    
    str_trim() %>%
    
    # Capitalize the first letter after cleaning.
    {
      paste0(
        str_to_upper(str_sub(., 1, 1)),
        str_sub(., 2)
      )
    }
}


# Apply typography changes that should appear only in paper tables,
# not in the results-and-metadata CSV.
format_varlab_for_table <- function(x) {
  x %>%
    # Convert straight quotation marks to proper opening/closing quotes.
    clean_quotes() %>%
    
    # Convert spaced punctuation hyphens to unspaced em dashes.
    # Hyphens within compound words are left unchanged.
    str_replace_all(" - ", "—")
}

# Standardize respondent labels for reader-facing tables.
format_respondent_for_table <- function(x) {
  x %>%
    str_replace("^Primary Caregiver$", "Primary caregiver") %>%
    str_replace("^Child$", "Youth")
}

# Standardize respondent labels for figures.
format_respondent_for_figure <- function(x) {
  x %>%
    str_replace("^Child$", "Youth")
}

#############################################################################
######################## RESULTS FOR MAIN PAPER #############################
#############################################################################

#### OVERALL DISTRIBUTION ####

# What percent of outcomes have R2 <= 0? 
winners %>%
  group_by(r2_monkeys <= 0) %>%
  count() %>% 
  mutate(prop = n / nrow(winners))

# Median
overall_median <- median(winners$r2_monkeys)
overall_median

# Plot
winners %>% 
  ggplot(mapping = aes(x = r2_monkeys)) +
  theme_bw(base_size = 20) + 
  geom_histogram(binwidth = .01) +
  geom_vline(aes(xintercept = overall_median, color = "Median"), linetype = "longdash", 
             show.legend = TRUE) + 
  theme(legend.title = element_blank()) + 
  #ggtitle(expression(paste("Distribution of ", italic(R)[CV]^2, " across outcomes"))) + 
  ylab("Number of Outcomes") + 
  xlab(expression(paste(italic(R)[CV]^2))) + 
  theme(plot.caption = element_text(size=10)) 

save_plot("overall_distribution", width = 12, height = 8)

#### TOP 10 TABLE ####

# Top 10 outcomes
top_10_latex <- winners %>%
  filter(!is.na(r2_monkeys)) %>%
  arrange(desc(r2_monkeys)) %>%
  slice_head(n = 10) %>%
  
  # Apply the same variable-label cleanups used in the public
  # results-and-metadata CSV and appendix table.
  mutate(
    varlab = clean_varlab_for_output(outcome, varlab)
  ) %>%
  
  transmute(
    `$\\widehat{R^2}_{\\mathrm{CV}}$ (95\\% CI)` = sprintf(
      "%.2f (%.2f, %.2f)",
      r2_monkeys,
      lower_ratio,
      upper_ratio
    ),
    Outcome = format_varlab_for_table(varlab),
    Respondent = format_respondent_for_table(respondent)
  )

# Function for making LaTeX tables
make_latex_table <- function(df, caption_text, file_name) {
  
  latex_table <- df %>%
    kable(
      format = "latex",
      booktabs = TRUE,
      escape = FALSE,
      align = c("l", "l", "l"),
      caption = caption_text
    ) %>%
    kable_styling(
      latex_options = "hold_position",
      font_size = 10
    ) %>%
    as.character()
  
  # Remove \addlinespace
  latex_table <- gsub(
    "\\\\addlinespace\\s*",
    "",
    latex_table
  )
  
  # Replace tabular with tabularx and set width
  latex_table <- gsub(
    "\\\\begin\\{tabular\\}\\[t\\]\\{lll\\}",
    "\\\\begin{tabularx}{\\\\linewidth}{lXl}",
    latex_table
  )
  
  latex_table <- gsub(
    "\\\\end\\{tabular\\}",
    "\\\\end{tabularx}",
    latex_table
  )
  
  writeLines(
    latex_table,
    file.path(fig_dir, file_name)
  )
}

# Save table
make_latex_table(
  top_10_latex,
  "Top ten most predictable outcomes.",
  "top_ten_most_predictable_outcomes_table.tex"
)

#### SIMPLE BENCHMARK & SKILL ####

# Join benchmark and winning pipeline results
comparison_df <- benchmarks %>%
  select(outcome, r2_benchmark) %>%
  inner_join(
    winners %>% select(outcome, r2_monkeys),
    by = "outcome"
  )

# Compute common limits for x and y so the plot will be square
axis_limits <- range(
  c(comparison_df$r2_benchmark,
    comparison_df$r2_monkeys),
  na.rm = TRUE
)

# Identify the two points to circle
circled_points <- comparison_df %>%
  filter(outcome %in% c("ch6chtcm", "ck6chtcm"))

# Compute the center of the circle as the midpoint of the two points
x_center <- mean(circled_points$r2_benchmark)
y_center <- mean(circled_points$r2_monkeys)

# Compute a radius that encloses both points, plus a little padding
radius <- max(
  sqrt(
    (circled_points$r2_benchmark - x_center)^2 +
      (circled_points$r2_monkeys - y_center)^2
  )
) + 0.03

# Create coordinates for a circle
theta <- seq(0, 2 * pi, length.out = 300)
circle_df <- tibble(
  x = x_center + radius * cos(theta),
  y = y_center + radius * sin(theta)
)

# Plot benchmark vs. winning pipeline
comparison_df %>%
  ggplot(aes(x = r2_benchmark, y = r2_monkeys)) +
  geom_point(alpha = 0.25) +
  geom_abline(
    slope = 1,
    intercept = 0,
    color = "blue",
    linetype = "dashed"
  ) +
  geom_path(
    data = circle_df,
    aes(x = x, y = y),
    inherit.aes = FALSE,
    color = "darkgreen",
    linewidth = .5
  ) +
  coord_equal(xlim = axis_limits, ylim = axis_limits) +
  theme_bw(base_size = 20) +
  labs(
    x = expression(paste(italic(R)[CV]^2, " for Simple One-Feature Benchmark")),
    y = expression(paste(italic(R)[CV]^2, " for Winning Pipeline"))
  )

save_plot("benchmark_vs_winners", width = 8, height = 8)

# Calculate skill for each outcome
comparison_df <- comparison_df %>%
  mutate(skill = r2_monkeys - r2_benchmark)

# Median skill across all outcomes
median_skill_all <- median(comparison_df$skill)
median_skill_all

# Find the R2 cutoff for the top decile of winning-model performance
top_decile_cutoff <- quantile(
  comparison_df$r2_monkeys,
  probs = 0.90
)

# Median skill among outcomes in the top decile of winning-model performance
median_skill_top_decile <- comparison_df %>%
  filter(r2_monkeys >= top_decile_cutoff) %>%
  summarise(
    median_skill = median(skill)
  )
median_skill_top_decile

# Median accuracy among outcomes in the top decile of winning-model performance
# (we mention median accuracy for the top decile in the section on skill)
median_accuracy_top_decile <- comparison_df %>%
  filter(r2_monkeys >= top_decile_cutoff) %>%
  summarise(
    median_accuracy = median(r2_monkeys)
  )
median_accuracy_top_decile

# What percent of outcomes have skill < 0? 
comparison_df %>%
  group_by(skill < 0) %>%
  count() %>%
  mutate(prop = n / nrow(comparison_df))

# For two outcomes with high accuracy and high skill (both are height outcomes), retrieve results for 
# (1) the simple benchmark, (2) the model with two handpicked features, and (3) the winning pipeline.

two_height_outcomes_for_special_exploration <- c("ch6chtcm", "ck6chtcm")

height_outcome_results <- benchmarks %>%
  filter(outcome %in% two_height_outcomes_for_special_exploration) %>%
  select(
    outcome,
    benchmark_r2 = r2_benchmark,
    benchmark_lower = lower_ratio,
    benchmark_upper = upper_ratio
  ) %>%
  
  left_join(
    two_feature_models %>%
      filter(outcome %in% two_height_outcomes_for_special_exploration) %>%
      select(
        outcome,
        two_feature_r2 = r2_monkeys,
        two_feature_lower = lower_ratio,
        two_feature_upper = upper_ratio
      ),
    by = "outcome"
  ) %>%
  
  left_join(
    winners %>%
      filter(outcome %in% two_height_outcomes_for_special_exploration) %>%
      select(
        outcome,
        winning_pipeline_r2 = r2_monkeys,
        winning_pipeline_lower = lower_ratio,
        winning_pipeline_upper = upper_ratio
      ),
    by = "outcome"
  ) %>%
  
  arrange(outcome)

# Print results in a text file
height_outcome_results_for_text <- height_outcome_results %>%
  transmute(
    outcome,
    
    `One-feature benchmark` = sprintf(
      "%.2f (95%% CI %.2f, %.2f)",
      benchmark_r2,
      benchmark_lower,
      benchmark_upper
    ),
    
    `Two-feature model` = sprintf(
      "%.2f (95%% CI %.2f, %.2f)",
      two_feature_r2,
      two_feature_lower,
      two_feature_upper
    ),
    
    `Winning pipeline` = sprintf(
      "%.2f (95%% CI %.2f, %.2f)",
      winning_pipeline_r2,
      winning_pipeline_lower,
      winning_pipeline_upper
    )
  )

# Format as readable plain text
height_outcome_results_text <- capture.output(
  print(
    height_outcome_results_for_text,
    n = Inf,
    width = Inf
  )
)

# Save to the same output directory as the figures and tables
writeLines(
  height_outcome_results_text,
  file.path(fig_dir, "height_outcomes_special_exploration.txt")
)

#### PLOT BY SUBTOPIC ####
# Note: We refer to subtopics as "topics" in our paper

# Make df with a row for every unique subtopic-outcome combination 
### Step 1: split the subtopics column into two columns, each with a single subtopic name
### Note: the "expected 2 pieces" warning is expected because some outcomes have only one subtopic
winners_by_subtopic_split <- winners %>%
  separate(subtopics, sep = ";", into = c("subtopic_1", "subtopic_2")) %>%
  mutate(subtopic_1 = trimws(subtopic_1), 
         subtopic_2 = trimws(subtopic_2))
### Step 2: Put data in long format 
winners_by_subtopic <- winners_by_subtopic_split %>%
  pivot_longer(cols = c(subtopic_1, subtopic_2), 
               names_to = NULL, 
               values_to = "individual_subtopic") %>%
  filter(!is.na(individual_subtopic))

# Clean up subtopic names for plotting
winners_by_subtopic <- winners_by_subtopic %>%
  mutate(
    # Start with title case
    individual_subtopic = str_to_title(individual_subtopic),
    
    # Lowercase short function words
    individual_subtopic = str_replace_all(
      individual_subtopic,
      "\\b(And|Of|The|In|On|For|To)\\b",
      ~ str_to_lower(.x)
    ),
    
    # Lowercase the first letter after a hyphen in compound words
    # e.g., "Non-Traditional" -> "Non-traditional"
    individual_subtopic = str_replace_all(
      individual_subtopic,
      "(?<=-)[A-Z]",
      ~ str_to_lower(.x)
    ),
    
    # Replace spaced hyphen with em dash
    individual_subtopic = str_replace_all(
      individual_subtopic,
      " - ",
      "\u2014"  # em dash "
    )
  )

# Plot
winners_by_subtopic %>%
  ggplot(aes(x = r2_monkeys, y = reorder(individual_subtopic, r2_monkeys, 
                                         FUN = median))) + 
  theme_bw(base_size = 20) + 
  geom_jitter(width = 0, height = 0.3, alpha = .5, size = 1) +
  geom_boxplot(alpha = 0) + 
  guides(color = "none") + 
  xlab(expression(paste(italic(R)[CV]^2))) + 
  ylab("Topic") #+ 
#ggtitle(expression(paste(italic(R)[CV]^2, " by subtopic"))) + 
#labs(caption = paste("Each outcome has up to two subtopics, so an outcome can appear twice on the plot.
#Points are vertically jittered. |", slug, sep = " "))

save_plot("all_subtopics", width = 12, height = 12)

# Calculate median R2 for each subtopic
winners_by_subtopic %>%
  group_by(individual_subtopic) %>%
  summarise(median(r2_monkeys)) %>%
  arrange(desc(`median(r2_monkeys)`))

#### PLOT BY FOCAL PERSON ####

# Create focal person variable based on coding of whether the outcome is about the child, parent, or both
winners <- winners %>%
  mutate(focal_person = case_when(
    fp_fchild==1 & fp_PCG==0 & fp_mother==0 & fp_father==0 ~ "Youth Only",
    fp_fchild==0 & (fp_PCG==1 | fp_mother==1 | fp_father==1) ~ "Parent Only",
    fp_fchild==1 & (fp_PCG==1 | fp_mother==1 | fp_father==1) ~ "Youth + Parent", 
    fp_fchild==0 & fp_PCG==0 & fp_mother==0 & fp_father==0 ~ "Other",
    TRUE ~ "Category missed in coding" # No rows should receive this assignment 
  ))

# Plot
winners %>%
  ggplot(aes(x = r2_monkeys, y = reorder(focal_person, r2_monkeys, 
                                         FUN = median))) + 
  theme_bw(base_size = 20) + 
  geom_jitter(width = 0, height = 0.3, alpha = .5, size = 2) +
  geom_boxplot(alpha = 0) + 
  guides(color = "none") + 
  xlab(expression(paste(italic(R)[CV]^2))) + 
  ylab("Focal Person") + 
  #ggtitle(expression(paste(italic(R)[CV]^2, " by focal person"))) + 
  #labs(caption = paste("Points are vertically jittered. |", slug, sep = " ")) + 
  theme(plot.caption = element_text(size=10)) 

save_plot("focal_person", width = 12, height = 8)

# Calculate median R2 for each focal person category
winners %>% 
  group_by(focal_person) %>%
  summarise(median(r2_monkeys))

#### IF WE RANDOMLY CHOOSE AN OUTCOME FOR PARENT OR CHILD FOCAL PERSON, WHAT PERCENT OF THE TIME IS PARENT R2 HIGHER? ####

# Split into separate data frames
child_only <- winners %>%
  filter(focal_person == "Youth Only") %>%
  select(child_outcome = outcome,
         child_r2 = r2_monkeys)

parent_only <- winners %>%
  filter(focal_person == "Parent Only") %>%
  select(parent_outcome = outcome,
         parent_r2 = r2_monkeys)

# Create all possible child-parent pairs and compare R2
focal_person_comparison_results <- tidyr::crossing(child_only, parent_only) %>%
  mutate(
    parent_higher = parent_r2 > child_r2,
    tie = parent_r2 == child_r2
  )

# Percent of the time the parent outcome has a higher R2
focal_person_comparison_results %>%
  summarise(
    n_pairs = n(),
    prop_parent_higher = mean(parent_higher),
    percent_parent_higher = 100 * mean(parent_higher),
    prop_ties = mean(tie)
  )

#### PLOT BY RESPONDENT ####

# Plot 
winners %>%
  mutate(respondent_plot = format_respondent_for_figure(respondent)) %>%
  ggplot(aes(x = r2_monkeys, y = reorder(respondent_plot, r2_monkeys,
                                         FUN = median))) + 
  theme_bw(base_size = 20) + 
  geom_jitter(width = 0, height = 0.3, alpha = .5, size = 2) +
  geom_boxplot(alpha = 0) + 
  guides(color = "none") + 
  xlab(expression(paste(italic(R)[CV]^2))) + 
  ylab("Respondent") + 
  #ggtitle(expression(paste(italic(R)[CV]^2, " by respondent"))) + 
  #labs(caption = paste("Points are vertically jittered. |", slug, sep = " ")) + 
  theme(plot.caption = element_text(size=10)) 

save_plot("respondent", width = 12, height = 8)

# Calculate median R2 for each respondent category
winners %>% 
  group_by(respondent) %>%
  summarise(median(r2_monkeys)) 

#### IF WE RANDOMLY CHOOSE AN OUTCOME FOR PARENT OR CHILD RESPONDENT, WHAT PERCENT OF THE TIME IS PARENT R2 HIGHER? ####

# Split into separate data frames
child_respondent <- winners %>%
  filter(respondent == "Child") %>%
  select(child_outcome = outcome,
         child_r2 = r2_monkeys)

pcg_respondent <- winners %>%
  filter(respondent == "Primary Caregiver") %>%
  select(pcg_outcome = outcome,
         pcg_r2 = r2_monkeys)

# Create all possible Child–PCG pairs and compare R2
respondent_comparison_results <- tidyr::crossing(child_respondent, pcg_respondent) %>%
  mutate(
    pcg_higher = pcg_r2 > child_r2,
    tie = pcg_r2 == child_r2
  )

# Percent of the time the parent outcome has a higher R2
respondent_comparison_results %>%
  summarise(
    n_pairs = n(),
    prop_pcg_higher = mean(pcg_higher),
    percent_pcg_higher = 100 * mean(pcg_higher),
    prop_ties = mean(tie)
  )

#### PLOT BY OUTCOME SOURCE (SCALES VS. INDIVIDUAL SURVEY ITEMS) ####

# Create variable indicating whether the outcome is a scale
winners <- winners %>% 
  mutate(is_scale = ifelse(source == "scale", "Scale", "Non-scale"))

# Plot
winners %>%
  ggplot(aes(x = r2_monkeys, y = reorder(is_scale, r2_monkeys,
                                         FUN = median))) + 
  theme_bw(base_size = 20) + 
  geom_jitter(width = 0, height = 0.3, alpha = .5, size = 1.5) +
  geom_boxplot(alpha = 0) + 
  guides(color = "none") + 
  xlab(expression(paste(italic(R)[CV]^2))) + 
  ylab("Type") + 
  #ggtitle(expression(paste(italic(R)[CV]^2, " for scales vs. non-scales"))) +
  #labs(caption = paste("Points are vertically jittered. |", slug, sep = " ")) + 
  theme(plot.caption = element_text(size=10)) 

save_plot("scale", width = 12, height = 8)

# Calculate median R2 for scales vs. non-scales
winners %>%
  group_by(is_scale) %>% 
  summarise(median(r2_monkeys))

#### IF WE RANDOMLY CHOOSE AN OUTCOME FOR SCALE OR NON-SCALE, WHAT PERCENT OF THE TIME IS SCALE R2 HIGHER? ####

# Split into separate data frames
scales_df <- winners %>%
  filter(is_scale == "Scale") %>%
  select(scale_outcome = outcome,
         scale_r2 = r2_monkeys)

non_scales_df <- winners %>%
  filter(is_scale == "Non-scale") %>%
  select(non_scale_outcome = outcome,
         non_scale_r2 = r2_monkeys)

# Create all possible scale vs. non-scale pairs and compare R2
scale_comparison_results <- tidyr::crossing(scales_df, non_scales_df) %>%
  mutate(
    scale_higher = scale_r2 > non_scale_r2,
    tie = scale_r2 == non_scale_r2
  )

# Percent of the time the scale outcome has a higher R2
scale_comparison_results %>%
  summarise(
    n_pairs = n(),
    prop_scale_higher = mean(scale_higher),
    percent_scale_higher = 100 * mean(scale_higher),
    prop_ties = mean(tie)
  )


############################################################################
################################ APPENDICES ################################
############################################################################

#### COMPARISON TO FFC OUTCOMES ####
# Note: I confirmed the outcome values matched between our version and FFC.
# The details of the FFC measures are reported in the Salganik et al. (2020) appendix.
# We do not predict the "layoff" outcome from FFC (p6k13) because that question 
# could be skipped in the survey logic, and therefore does not meet our inclusion criteria.

# Filter to outcomes that were in FFC and clean their names
outcomes_that_were_in_FFC <- c("k6_scale_GPA", 
                               "k6_scale_EPOCH_perseverance", # "grit"
                               "p6_scale_materialHardship_pastYear", 
                               "p6_scale_eviction", # eviction since past wave
                               "p6k4" # job training
)
winners_for_ffc_comparison <- winners %>%
  filter(outcome %in% outcomes_that_were_in_FFC) %>%
  select(outcome, r2_monkeys, lower_ratio, upper_ratio) %>%
  # Rename outcome  
  mutate(outcome = case_when(
    outcome == "k6_scale_GPA" ~ "GPA", 
    outcome == "k6_scale_EPOCH_perseverance" ~ "Grit", 
    outcome == "p6_scale_materialHardship_pastYear" ~ "Material Hardship", 
    outcome == "p6_scale_eviction" ~ "Eviction", 
    outcome == "p6k4" ~ "Job Training", 
    TRUE ~ outcome
  ))

# Create df with results for the FFC outcomes (from Salganik et al. 2020)
ffc_results <- tibble(
  outcome = c("Material Hardship", "GPA", "Grit", "Eviction", "Job Training"),
  r2_ffc = c(0.23, 0.19, 0.06, 0.05, 0.05)
)

# Merge results from the two studies & create ordered factor based on winning pipeline performance
ffc_comparison_df <- left_join(winners_for_ffc_comparison, ffc_results, by = "outcome") %>%
  arrange(desc(r2_monkeys)) %>%
  mutate(outcome = factor(outcome, levels = outcome))

# Plot 
dodge_w <- 0.35
ggplot(ffc_comparison_df, aes(x = outcome)) +
  theme_bw(base_size = 20) +
  
  geom_errorbar(
    aes(ymin = lower_ratio, ymax = upper_ratio,
        color = "pipeline"),
    width = 0.18,
    position = position_nudge(x = -dodge_w/2)
  ) +
  
  geom_point(
    aes(y = r2_monkeys, color = "pipeline", shape = "pipeline"),
    size = 3,
    position = position_nudge(x = -dodge_w/2)
  ) +
  
  geom_point(
    aes(y = r2_ffc, color = "ffc", shape = "ffc"),
    size = 3,
    position = position_nudge(x = +dodge_w/2)
  ) +
  
  expand_limits(y = 0) + 
  
  xlab("Outcome") +
  ylab(expression(italic(R)^2)) +
  #ggtitle("Winning pipeline vs. winning Challenge submission") +
  
  scale_color_manual(
    name = NULL,
    values = c(
      "pipeline" = "#0072B2",  # blue (Okabe–Ito)
      "ffc" = "#D55E00"        # vermillion (Okabe–Ito)
    ),
    breaks = c("pipeline", "ffc"),
    labels = c(
      expression("Winning Pipeline" ~ italic(R)[CV]^2),
      expression("Winning Challenge" ~ italic(R)[Holdout]^2)
    )
  ) +
  
  scale_shape_manual(
    name = NULL,
    values = c(
      "pipeline" = 16,  # solid circle
      "ffc" = 17        # solid triangle
    ),
    breaks = c("pipeline", "ffc"),
    labels = c(
      expression("Winning Pipeline" ~ italic(R)[CV]^2),
      expression("Winning Challenge" ~ italic(R)[Holdout]^2)
    )
  ) +
  
  guides(
    color = guide_legend(
      override.aes = list(linetype = c(1, 0))
    )
  ) +
  
  theme(
    plot.caption = element_text(size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.key.width = unit(1.1, "lines"),
    legend.text = element_text(hjust = 0),
    legend.position = c(0.98, 0.98),
    legend.justification = c(1, 1),
    legend.background = element_rect(fill = "white", color = "black", linewidth = 0.3)
  )

save_plot("winning_pipeline_vs_ffc_results", width = 8, height = 6)

#### CROSSWALK OF FOCAL PERSON AND MEASURED EARLIER ####

# Plot with focal person on y-axis and separate facet for whether same construct was measured at an earlier wave
winners %>%
  filter(focal_person %in% c("Youth Only", "Parent Only")) %>%
  mutate(
    measured_at_earlier_wave = factor(
      measured_at_earlier_wave,
      levels = c("Yes", "No"),
      labels = c("Previously Measured", "Not Previously Measured")
    )
  ) %>%
  ggplot(aes(x = r2_monkeys, y = reorder(focal_person, r2_monkeys,
                                         FUN = median))) +
  facet_wrap(~measured_at_earlier_wave, ncol = 1) + 
  theme_bw(base_size = 18) + 
  geom_jitter(width = 0, height = 0.3, alpha = .5, size = 1.5) +
  geom_boxplot(alpha = 0) + 
  guides(color = "none") + 
  xlab(expression(paste(italic(R)[CV]^2))) + 
  ylab("Focal Person") #+ 
#ggtitle(expression(paste(italic(R)[CV]^2, " by focal person and whether same construct was measured earlier"))) + 
#labs(caption = paste("Points are vertically jittered. |", slug, sep = " "))

save_plot("measured_earlier_by_focal_person", width = 12, height = 8)

#### CROSSWALK OF RESPONDENT AND MEASURED EARLIER ####

# Plot with respondent on y-axis and separate facet for whether same construct was measured at an earlier wave
winners %>%
  filter(respondent %in% c("Child", "Primary Caregiver")) %>%
  mutate(
    measured_at_earlier_wave = factor(
      measured_at_earlier_wave,
      levels = c("Yes", "No"),
      labels = c("Previously Measured", "Not Previously Measured")
    ),
    respondent_plot = format_respondent_for_figure(respondent)
  ) %>%
  ggplot(aes(x = r2_monkeys,
             y = reorder(respondent_plot, r2_monkeys, FUN = median))) +
  facet_wrap(~measured_at_earlier_wave, ncol = 1) +
  theme_bw(base_size = 18) +
  geom_jitter(width = 0, height = 0.3, alpha = .5, size = 1.5) +
  geom_boxplot(alpha = 0) +
  guides(color = "none") +
  xlab(expression(paste(italic(R)[CV]^2))) +
  ylab("Respondent") #+
#ggtitle(expression(paste(italic(R)[CV]^2, " by respondent and whether same construct was measured earlier"))) +
#labs(caption = paste("Points are vertically jittered. |", slug, sep = " "))

save_plot("measured_earlier_by_respondent", width = 12, height = 8)


#######################################################################
######### CSV FILE WITH RESULTS AND METADATA FOR ALL OUTCOMES #########
#######################################################################

cleaned_version_of_winners <- winners %>% 
  rename(r2_cv_for_winning_pipelines = r2_monkeys, 
         lower_ci_for_winning_pipelines = lower_ratio, 
         upper_ci_for_winning_pipelines = upper_ratio,
         sample_size_before_cross_validation_splits = n) %>%
  mutate(
    outcome_was_in_FFC = ifelse(outcome %in% outcomes_that_were_in_FFC, "Yes", "No"),
    
    # Apply substantive reader-facing variable-label cleanups to the CSV.
    # Table-only typography/LaTeX changes (such as em dashes and LaTeX
    # rendering of < and >) are applied later only when tables are created.
    varlab = clean_varlab_for_output(outcome, varlab)
  ) %>%
  select(outcome, 
         # Results
         r2_cv_for_winning_pipelines, lower_ci_for_winning_pipelines, upper_ci_for_winning_pipelines,
         # Metadata
         # Notes: 
         # Most of these metadata columns are from FFCWS; see https://metadata.ffcws.princeton.edu/
         # Some columns are created for our analyses in this project.
         # For scales that we calculated ourselves, we generated the metadata following the FFCWS metadata format
         sample_size_before_cross_validation_splits,
         outcome_was_in_FFC,
         varlab:is_scale,
         -in_FFC_file) # in_FFC_file is replaced by "outcome_was_in_FFC" for clarity (the FFCWS "in_FFC_file" column refers to features)

cleaned_version_of_benchmarks <- benchmarks %>%
  rename(r2_cv_for_benchmark_pipeline = r2_benchmark, 
         lower_ci_for_benchmark_pipeline = lower_ratio, 
         upper_ci_for_benchmark_pipeline = upper_ratio) %>%
  select(outcome, r2_cv_for_benchmark_pipeline, lower_ci_for_benchmark_pipeline, upper_ci_for_benchmark_pipeline)   

cleaned_version_of_winners_and_benchmarks <- left_join(
  cleaned_version_of_winners,
  cleaned_version_of_benchmarks,
  by = "outcome"
) %>%
  relocate(
    outcome,
    varlab,
    r2_cv_for_winning_pipelines,
    lower_ci_for_winning_pipelines,
    upper_ci_for_winning_pipelines,
    r2_cv_for_benchmark_pipeline,
    lower_ci_for_benchmark_pipeline,
    upper_ci_for_benchmark_pipeline
  )       

readr::write_csv(
  cleaned_version_of_winners_and_benchmarks,
  file.path(
    fig_dir,
    paste0("results_and_metadata_for_all_outcomes_", file_path_label, ".csv")
  )
)


###################################################
#### LATEX TABLE WITH RESULTS FOR ALL OUTCOMES ####
###################################################

# Helper function for formatting R2 estimates and confidence intervals.
# Values that would round to "-0.00" are printed as "0.00".
format_r2_ci <- function(r2, lower, upper) {
  
  # Convert values that round to negative zero to ordinary zero
  r2[round(r2, 2) == 0] <- 0
  lower[round(lower, 2) == 0] <- 0
  upper[round(upper, 2) == 0] <- 0
  
  ifelse(
    is.na(r2),
    "",
    sprintf("%.2f (%.2f, %.2f)", r2, lower, upper)
  )
}


# Helper function for making < and > render correctly in LaTeX.
#
# Temporary placeholders are used here because kable(escape = TRUE)
# will otherwise escape the LaTeX commands themselves. The placeholders
# are converted to LaTeX after the table has been generated.
protect_comparison_symbols <- function(x) {
  x %>%
    # Replace multi-character operators first so >= is not split apart.
    str_replace_all(">=", "GREATERTHANOREQUALPLACEHOLDER") %>%
    str_replace_all("<", "LESSTHANPLACEHOLDER") %>%
    str_replace_all(">", "GREATERTHANPLACEHOLDER")
}


# Helper function for wrapping long outcome names.
#
# A temporary placeholder is inserted after every 25 characters.
# Underscores count as characters. The placeholder is converted to
# a LaTeX line break after kable() has escaped the table contents.
wrap_outcome <- function(x, width = 25) {
  vapply(
    x,
    function(s) {
      
      # Split the outcome name into individual characters
      chars <- strsplit(s, "", fixed = TRUE)[[1]]
      
      # Do nothing if the outcome name already fits on one line
      if (length(chars) <= width) {
        return(s)
      }
      
      # Divide the characters into groups of at most 'width' characters
      pieces <- split(
        chars,
        ceiling(seq_along(chars) / width)
      )
      
      # Reassemble each group and insert a temporary marker between lines
      paste(
        vapply(
          pieces,
          paste0,
          collapse = "",
          FUN.VALUE = character(1)
        ),
        collapse = "OUTCOMELINEBREAKPLACEHOLDER"
      )
    },
    FUN.VALUE = character(1)
  )
}


# Create a simplified data frame for the appendix table
all_outcomes_results_latex <- cleaned_version_of_winners_and_benchmarks %>%
  
  # Order outcomes from highest to lowest winning-pipeline R2
  arrange(desc(r2_cv_for_winning_pipelines)) %>%
  
  transmute(
    
    # Outcome variable name.
    #
    # Long outcome names are divided after every 25 characters.
    # The temporary markers are converted to LaTeX line breaks below.
    Outcome = wrap_outcome(
      outcome,
      width = 25
    ),
    
    # Variable labels have already received substantive cleanup before the
    # results-and-metadata CSV is written. Apply table-only typography here,
    # then protect < and > because this table is rendered in LaTeX.
    `Variable label` = varlab %>%
      format_varlab_for_table() %>%
      protect_comparison_symbols(),
    
    # Clean respondent labels
    Respondent = format_respondent_for_table(respondent),
    
    # Winning-pipeline R2 and 95% CI
    `Winning pipeline` = format_r2_ci(
      r2_cv_for_winning_pipelines,
      lower_ci_for_winning_pipelines,
      upper_ci_for_winning_pipelines
    ),
    
    # Benchmark-pipeline R2 and 95% CI
    `Benchmark pipeline` = format_r2_ci(
      r2_cv_for_benchmark_pipeline,
      lower_ci_for_benchmark_pipeline,
      upper_ci_for_benchmark_pipeline
    )
  )

# Create a longtable that can span multiple appendix pages.
#
# A 6pt gap is added after each data row. This creates visible
# separation between different outcomes, while wrapped lines within
# the same row remain directly adjacent to one another.
all_outcomes_results_table <- all_outcomes_results_latex %>%
  kable(
    format = "latex",
    booktabs = TRUE,
    longtable = TRUE,
    escape = TRUE,
    align = c("l", "l", "l", "l", "l"),
    caption = "Predictive performance for all outcomes.",
    col.names = c(
      "Outcome",
      "Variable label",
      "Respondent",
      "Winning pipeline R2CV (95% CI)",
      "Benchmark pipeline R2CV (95% CI)"
    ),
    
    # Add vertical space between separate rows.
    # This does not add space between wrapped lines within a cell.
    linesep = "\\addlinespace[6pt]"
  ) %>%
  
  kable_styling(
    latex_options = "repeat_header"
  ) %>%
  
  # Set column widths for a landscape page.
  column_spec(1, width = "3.8cm") %>%
  column_spec(2, width = "9.1cm") %>%
  column_spec(3, width = "2.6cm") %>%
  column_spec(4, width = "3.0cm") %>%
  column_spec(5, width = "3.0cm") %>%
  
  as.character()


# Convert temporary markers in outcome names to LaTeX line breaks.
all_outcomes_results_table <- str_replace_all(
  all_outcomes_results_table,
  fixed("OUTCOMELINEBREAKPLACEHOLDER"),
  function(x) "\\newline{}"
)


# Convert temporary comparison-symbol markers to LaTeX symbols.
#
# >= is rendered as the mathematical greater-than-or-equal-to symbol.
all_outcomes_results_table <- str_replace_all(
  all_outcomes_results_table,
  fixed("GREATERTHANOREQUALPLACEHOLDER"),
  function(x) "$\\geq$"
)

# Using \textless{} and \textgreater{} makes individual < and > symbols
# render as ordinary text within variable labels.
all_outcomes_results_table <- str_replace_all(
  all_outcomes_results_table,
  fixed("LESSTHANPLACEHOLDER"),
  function(x) "\\textless{}"
)

all_outcomes_results_table <- str_replace_all(
  all_outcomes_results_table,
  fixed("GREATERTHANPLACEHOLDER"),
  function(x) "\\textgreater{}"
)


# Replace the plain-text R2CV column headings generated by kable()
# with properly formatted LaTeX math.
all_outcomes_results_table <- str_replace_all(
  all_outcomes_results_table,
  fixed("Winning pipeline R2CV (95\\% CI)"),
  function(x) "Winning pipeline $\\widehat{R^2}_{\\mathrm{CV}}$ (95\\% CI)"
)

all_outcomes_results_table <- str_replace_all(
  all_outcomes_results_table,
  fixed("Benchmark pipeline R2CV (95\\% CI)"),
  function(x) "Benchmark pipeline $\\widehat{R^2}_{\\mathrm{CV}}$ (95\\% CI)"
)


# Wrap the table in a landscape environment.
#
# The table is explicitly set to 8-point font with 9-point line spacing.
# This keeps wrapped lines compact while retaining a 6-point gap between
# separate rows. Setting the font here also prevents the document-wide
# double spacing from affecting the table.
#
# Reducing \tabcolsep slightly decreases horizontal whitespace between
# columns and helps the five-column table fit comfortably on the page.
all_outcomes_results_table <- paste0(
  "\\begin{landscape}\n",
  "\\begingroup\n",
  "\\singlespacing\n",
  "\\fontsize{8}{9}\\selectfont\n",
  "\\setlength{\\tabcolsep}{3pt}\n",
  all_outcomes_results_table,
  "\n\\endgroup\n",
  "\\end{landscape}"
)


# Save LaTeX table
writeLines(
  all_outcomes_results_table,
  file.path(
    fig_dir,
    "all_outcomes_results_table_for_appendix.tex"
  )
)