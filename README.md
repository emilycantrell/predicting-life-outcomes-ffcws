[![Million monkeys](https://github.com/emilycantrell/predicting-life-outcomes-ffcws/actions/workflows/million-monkeys.yml/badge.svg)](https://github.com/emilycantrell/predicting-life-outcomes-ffcws/actions/workflows/million-monkeys.yml)

# Overview

This repository contains code to reproduce the results in [*Machine Learning Models Trained on a Longitudinal Survey Dataset Poorly Predict Hundreds of Life Outcomes*](https://arxiv.org/abs/2610.17115). For setup and usage instructions, see the `docs/` directory.

**Table of Contents**

* [Terminology](#terminology)
* [Job description files](#job-description-files)
* [Data overview](#data-overview)
* [Data access](#data-access)
* [Architecture](#architecture)
* [Running on single machine](#running-on-single-machine)
* [Running on a cluster of machines](#running-on-a-cluster-of-machines)
* [Funding and acknowledgments](#funding-and-acknowledgments)

# Terminology

* **Million Monkeys:** Our internal nickname for this project was "Million Monkeys," referring to the "[infinite monkey theorem](https://en.wikipedia.org/wiki/Infinite_monkey_theorem)," since we have many pipelines working to predict each outcome. You will see the terms "million-monkeys" and "monkeys" in various places throughout the codebase.
* **Fragile Families:** The Future of Families and Child Wellbeing Study was originally named the "Fragile Families and Child Wellbeing Study." In some places in the codebase, you will see the data referred to as "Fragile Families."

# Job description files

Pipelines are specified using job description files. These files are written in [TOML](https://github.com/toml-lang/toml) format. For examples, see the [Jobfiles](./jobfiles) directory.

# Data overview

This code is designed to use data from the [Future of Families and Child Wellbeing Study](https://ffcws.princeton.edu/). We used the original FFCWS data rather than the version created for the Fragile Families Challenge, which incorporated additional privacy protections such as the addition of noise to some variables. However, we restricted our analysis to the 4,242 families included in the Challenge sample, rather than using all 4,898 families in the full FFCWS sample. Differences between the full FFCWS dataset and the version created for the Challenge are explained in [Lundberg et al. (2019)](https://doi.org/10.1177/2378023118813023), and additional information about the Fragile Families Challenge data files is provided in [Salganik et al. (2020)](https://doi.org/10.1073/pnas.1915006117).

# Data access

The data we used are not available in this repository due to privacy protections. Researchers interested in getting access to the Future of Families and Child Wellbeing Study data should apply through the [Future of Families and Child Wellbeing data access process](https://ffcws.princeton.edu/documentation). We used file versions FF_allwaves_2020v2.dta and FF_wave6_2020v2.dta. These file versions are no longer available online because they have been superseded by newer file versions. You can email [ffdata@princeton.edu](mailto:ffdata@princeton.edu) for access to the file versions we used.

We use the metadata file [FFMetadata_v09.csv](./data/dataset/fragile-families/FFMetadata_v09.csv), which we downloaded from the [FFCWS metadata site](https://metadata.ffcws.princeton.edu/search). This file can no longer be downloaded from the metadata site because it has been superseded by a newer version.

# Architecture

![Architecture](/images/monkeys_simplified_architecture.png)

# Running on single machine

See `docs/1-Local-Setup.md`.

# Running on a cluster of machines

See `docs/2-Azure-Setup.md`. We used Azure to run the jobs for the paper. Additional information on computing infrastructure can be found in [`azure/worker.bicep`](./azure/worker.bicep).

# Funding and acknowledgments

This project received funding from the Princeton DataX Initiative, Princeton Catalysis Initiative, and the Center for Statistics and Machine Learning (in the form of Azure computing credits).

The data we used in this research are from the Future of Families and Child Wellbeing Study (FFCWS). FFCWS was supported by the Eunice Kennedy Shriver National Institute of Child Health and Human Development (NICHD), the Office of the Director, National Institutes of Health (OD), and the National Institute on Minority Health and Health Disparities (NIMHD) of the National Institutes of Health under award numbers R01HD036916, R01HD039135, and R01HD040421, as well as a consortium of private foundations. Our research is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.
