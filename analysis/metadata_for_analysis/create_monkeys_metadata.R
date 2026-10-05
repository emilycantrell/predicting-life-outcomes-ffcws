# This code creates a metadata csv that contains both the
# original FFCWS metadata and the metadata created for scales 
# and units cases from the parsable file 

library(tidyverse)

# Load files (use repo root as the working directory)
ffcws_meta <- read.csv("data/dataset/fragile-families/FFMetadata_v09.csv")
parsable <- read.csv("data/dataset/fragile-families/parsable_scales_instructions.csv")

# Select just the metadata columns and remove "metadata_" prefix
parsable_meta <- parsable %>%
  select(starts_with("metadata")) %>%
  rename_with(~str_replace(., "metadata_", ""))

# Remove "intermediate" rows
parsable_meta <- parsable_meta %>%
  filter(!grepl("intermediate", new_name))

# Change fp columns to type "integer" as needed.
# This is necessary for columns where an "intermediate" row 
# had a value of "0 ; 1"; we did not edit these rows since the metadata
# for intermediate rows will not be used
parsable_meta <- parsable_meta %>%
  mutate(fp_fchild = as.integer(fp_fchild), 
         fp_PCG = as.integer(fp_PCG), 
         fp_other = as.integer(fp_other))

# Merge the FFCWS metadata with the scales & units metadata
monkeys_metadata <- full_join(ffcws_meta, parsable_meta)

# Save the file, using the version number of the FFCWS metadata in the file name
# (Set your desired working directory before running the next line)
write_csv(monkeys_metadata, "monkeys_metadata_v09.csv")
