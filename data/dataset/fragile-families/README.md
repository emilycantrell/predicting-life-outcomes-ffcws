# Fragile Families 

This directory contains a variety of data files used as inputs to our code. These files are specified in the `[data.files]` section of the jobfiles.

## Public files

### `FFCWS_idnums_cases_from_FFC_training.csv`

Contains a list of the case IDs from the Fragile Families Challenge training set. All analyses prior to journal submission were conducted within this sample. As explained in the manuscript, we will use the other half of the observations from the Challenge (the leaderboard set and holdout set from the Challenge) after receiving reviewer feedback.

### `FFMetadata_v09.csv`

This is the version of the FFCWS metadata we used.

### `Metadata_Edit_Tracker.csv`

Tracks small changes we made to v09 of the FFCWS metadata. The FFCWS team integrated some of these changes into later versions of the official metadata.

### `parsable_scales_instructions.csv`

Parsable instructions for computing scales, standardizing units, and combining units. See our appendix for additional information.

### `VariablesToRemove.json`

A list of features and outcomes that are removed before running the modeling pipelines. The reasons for removing these variables are explained within the file. 

## FFCWS data

To reproduce our results, obtain the FFCWS data files `FF_allwaves_2020v2.dta` and `FF_wave6_2020v2.dta` and place them in this directory as instructed in [docs/1-Local-Setup.md](../../../docs/1-Local-Setup.md).