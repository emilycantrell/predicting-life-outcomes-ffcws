# Local Setup

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Notes on running commands in this guide](#notes-on-running-commands-in-this-guide)
4. [Clone repository](#clone-repository)
5. [Set up the local environment](#set-up-the-local-environment)
   1. [Miniconda](#miniconda)
   2. [Mongo](#mongo)
      1. [Mongo service commands](#mongo-service-commands)
   3. [Create a virtual environment and install dependencies](#create-a-virtual-environment-and-install-dependencies)
6. [Run tests](#run-tests)
7. [Running jobs](#running-jobs)
   1. [Get FFCWS data](#get-ffcws-data)
   2. [Start job](#start-job)
      1. [Run the desired job](#run-the-desired-job)
   3. [Export results](#export-results)
   4. [Verify results](#verify-results)

## Overview

This document includes instructions for

- Setting up the local environment
- Installing local MongoDB server
- Running tests
- Running jobs on local machine

## Prerequisites

- Git (version >= 2.3) on the command line


## Notes on running commands in this guide

- All commands except for cloning the repository must be run in the root directory of the repository.
- The comments (starting with #) above commands explain their purpose.
- Arguments enclosed in angular brackets should be replaced with appropriate values.

## Clone repository

```bash
# Clone codebase.
git clone https://github.com/emilycantrell/predicting-life-outcomes-ffcws.git
```

## Set up the local environment

### Miniconda

- Install miniconda v4.10 or higher: [Miniconda installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

**⚠ Source your shell profile (or restart the terminal) to make sure conda executable is in the path. You can verify this by entering `conda --version` in the terminal.**

### Mongo

MongoDB is used to store the pipeline results.

- Install MongoDB v4 or higher: [MongoDB installation](https://www.mongodb.com/docs/manual/installation/)

On Ubuntu, the following command installs MongoDB:

```bash
# Install MongoDB using Ubuntu's package manager
wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | sudo apt-key add -;
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.2.list;
sudo apt-get update;
sudo apt-get install -y mongodb-org;
```

#### Mongo service commands

The following commands are used to start and check the status of mongo service on a linux machine. Use similar commands for other operating systems. Here are [commands](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/#run-mongodb-community-edition) that might work on Mac OSX.

```bash
# Start mongo service
sudo service mongod start

# Check the status of mongo service
sudo systemctl status mongod
```

If mongo is running then you would see `active(running)`.

**⚠ WARNING: Ensure that the mongo service is up and running. Pipeline code doesn’t check the status of mongo service.**

### Create a virtual environment and install dependencies

Replace * in the code below with "linux", "macos_intel", or "macos_arm64" depending on your operating system. Please note that our macos_arm64 environment specification is a beta version; it has worked for quick test runs but has not been thoroughly tested.

```bash
# Checkout branch (use the main branch unless you have created your own branch)
git fetch
git checkout <branch name>

# Creates million-monkeys environment and installs all the required Python packages
conda env create -f environment_pinned_*.yml
```

## Run tests

Run the following tests to ensure that the local environment is set up correctly.

```bash
# Checkout branch and latest changes
git checkout <branch name>
git pull origin <branch name>

# activate million-monkeys environment
conda activate million-monkeys

# Run e2e tests
python -m pytest -v -p no:warnings --regression ./test/desc_files/regression --incorrect ./test/desc_files/incorrect --regression ./test/desc_files/regression_missing --imperfect_regression ./test/desc_files/imperfect_regression --reproducibility ./test/desc_files/reproducibility --metadata_warning ./test/desc_files/metadata_warning --metadata_concepts ./test/desc_files/mm_concepts --metalearning ./test/desc_files/metalearning

# Run model related unittests :
python -m pytest models/

# Run preprocessing related unittests :
python -m pytest data/test_preprocessing.py
```

The following warning can be ignored because it is only a problem if we update to a new version of Pandas or XGboost: *FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.*

## Running jobs

### Get FFCWS data

There are two required files that are not tracked in the repository: FF_wave6_2020v2.dta and FF_allwaves_2020v2.dta. These are older versions of the FFCWS data that are no longer available for download online. The FFCWS team maintains archived copies of these files. To access them, please email ffdata@princeton.edu.

The following commands copy the required files from your downloads directory to their destination.

```bash
# Copies csv and data files to appropriate location. Replace the path/to/downloads with the path to directory containing the downloaded files
cp <path/to/downloads/FF_wave6_2020v2.dta> data/dataset/fragile-families
cp <path/to/downloads/FF_allwaves_2020v2.dta> data/dataset/fragile-families
```

### Start job

`jobfiles/jobfile_test_local_setup.toml` provides a very small job that you can run to test your local setup. Once you have confirmed that your setup is working, you can create your own jobfile. We recommend running only small jobs locally. For computationally intensive jobs, such as the one used to generate the results in our paper, use Azure.

#### Run the desired job

```bash
# activate million-monkeys environment
conda activate million-monkeys

# Starts a run with the given jobfile
python monkeys.py run jobfiles/jobfile_test_local_setup.toml
```

### Export results

```bash
# Exports results from MongoDB to json files.
# Replace /path/to/results/dir with the path where you want to save the results.
# The second argument should be the value of the field results.database_name in the jobfile.
python monkeys.py export-to-json </path/to/results/dir> <results.database_name:test_local_setup>
```

### Verify results

```bash
# Verify the exported json files
# Replace /path/to/results/dir with the path where you saved the results.
python monkeys.py verify-json-files </path/to/results/dir>
```

The above command records any inconsistencies in the json files in `/path/to/results/dir/errors.txt`.