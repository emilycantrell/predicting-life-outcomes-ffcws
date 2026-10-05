#!/usr/bin/env bash
set -euo pipefail

# start vms
az vm start --ids $(az vm list --query "[].id" -o tsv | grep "mm-worker")

# stop vms
az vm stop --ids $(az vm list --query "[].id" -o tsv | grep "mm-worker")

# deallocate vms
az vm deallocate --ids $(az vm list --query "[].id" -o tsv | grep "mm-worker")
