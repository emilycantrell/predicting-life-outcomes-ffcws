# Azure Setup

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Preliminaries](#preliminaries)
   1. [Local commands vs remote commands](#local-commands-vs-remote-commands)
   2. [Running commands in this guide](#running-commands-in-this-guide)
   3. [Azure login](#azure-login)
4. [Summary of the process](#summary-of-the-process)
5. [Creating Azure resources](#creating-azure-resources)
   1. [Naming VMs](#naming-vms)
   2. [Accessing Azure VMs from local machine](#accessing-azure-vms-from-local-machine)
      1. [Getting ip addresses of all VMs](#getting-ip-addresses-of-all-vms)
6. [Copy and execute init script](#copy-and-execute-init-script)
7. [Copy data files from local to Azure machines](#copy-data-files-from-local-to-azure-machines)
8. [Deploying jobs from local machine](#deploying-jobs-from-local-machine)
   1. [Update codebase on remote machines](#update-codebase-on-remote-machines)
   2. [Check mongodb is running](#check-mongodb-is-running)
   3. [Deploy job](#deploy-job)
   4. [Monitor job [Optional]](#monitor-job-optional)
   5. [Export partial results while the job is still running [optional]](#export-partial-results-while-the-job-is-still-running-optional)
9. [Copying results to local machine](#copying-results-to-local-machine)
10. [Delete VM and associated resources](#delete-vm-and-associated-resources)

## Overview

This project uses a [Horizontal scaling](https://docs.microsoft.com/en-us/azure/azure-monitor/autoscale/autoscale-overview#horizontal-vs-vertical-scaling) approach to speed up the execution of big jobs.
It splits a big jobfile into *k* smaller jobfiles and schedules them to run on a pool of *k* Microsoft Azure based virtual machines (VMs). This document describes the steps required to create a pool of worker VMs.
The current description is based on Azure services, but it can be replicated on
any other cloud provider.

## Prerequisites

1. Basic commandline fluency
2. Complete the following parts of `1-Local-Setup.md`: "Prerequisites," "Clone repository," "Miniconda," "Create a virtual environment and install dependencies," and "Get FFCWS data." You do not need to install MongoDB locally or run a local job before using Azure.
3. Basic understanding of Azure terminology
4. Azure account with permissions to create resources
5. ssh keys [Create and manage ssh keys](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/create-ssh-keys-detailed)

## Preliminaries

### Local commands vs remote commands

Most of the commands in this guide are meant to be executed on the local machine where the project is installed.
Commands that need to be run on remote VMs are indicated via a comment. To run remote commands on multiple machines, a cluster administration tool like *clusterssh* is highly recommended.
[ClusterSSH for cluster administration](https://github.com/duncs/clusterssh)
On OSX, *csshx* serves the same purpose. It can be installed via

```bash
brew install csshx
```

clusterssh commands in this guide can be used on OSX by replacing *clusterssh* with *csshx*

➡️ If you are having problems using csshx on a Mac (e.g., an error that starts with "Unimplemented: POSIX::tmpnam()") check out the csshx section in `3-Troubleshooting.md`.

### Running commands in this guide

- The commands must be entered in the million-monkeys conda environment, created during the local setup. So, you need to run `conda activate million-monkeys` or whatever you named the environment to get that started.
- Unless specified otherwise, all commands in this guide are to be executed at the root directory of the project.
- Command arguments within <,> must be replaced with appropriate values.

### Azure login

Our conda environment includes [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/). Use the following command to login to your Azure account via commandline

```bash
az login
```

## Summary of the process

1. Create blank VMs
2. Download and install the project on VMs
3. Copy FFCWS data
4. Deploy jobs from local machine
5. Download results from VMs to local machine
6. Delete machines

## Creating Azure resources

### Naming VMs

Even though Azure accepts any string value for VM names, the following format is required for automations to work.
VM name format : *prefixsuffix*. Example *mm-worker-1*. *mm-worker-* is the prefix and *1* is the integer suffix.

The resources required for running jobs for this project are defined in a [Bicep](https://docs.microsoft.com/en-us/azure/azure-resource-manager/bicep/overview?tabs=bicep) file.

The Bicep file is at: [Bicep file for creating an instance of worker VM](../azure/worker.bicep)

Resources defined in this file can be created by executing the following command:

```bash
cd azure
./deploy.sh -u <username> -p <prefix to use for VM name> -k <path to a file containing ssh public key> -s <suffix number for the first vm in the pool> -n <number of VMs in the pool> -g <name of Azure resource group>
```

Some of the above parameters have default values

```text
prefix="mm-worker-"
ssh_pub_key_file="~/.ssh/id_rsa.pub"
starting_suffix=1
user="millionmonkeys"
resource_group="CSML-Credits-Budget-RG"
```

Note : On most machines, *deploy.sh* script doesn’t have permissions to read ‘~/.ssh/id_rsa.pub’. Therefore, make a copy of that file to local directory.

The following command copies the public key and creates 8 VMs (*mm-worker-1, mm-worker-2,…,mm-worker-8*). Replace `<your-resource-group>` with the name of the Azure resource group containing your VMs.

```bash
cd azure
cp ~/.ssh/id_rsa.pub ./pubkey.txt
./deploy.sh -n 8 -k ./pubkey.txt -s 1 -g <your-resource-group>
```

It takes a couple of minutes to create the resources defined in Bicep file. After successful termination of the above command, search in Azure portal for the newly created VMs.

Note: If you get an error starting with "'code': 'AuthorizationFailed'", see the "problems accessing Azure" section of `docs/3-Troubleshooting.md`.

### Accessing Azure VMs from local machine

#### Getting ip addresses of all VMs

The following command retrieves ip addresses of VMs in a resource group based on their names

```text
Usage: remote_file_management.py get-worker-ips [OPTIONS] START_WORKER_ID
                                                END_WORKER_ID

Options:
  --user TEXT       [default: millionmonkeys]
  --rg TEXT         [default: CSML-Credits-Budget-RG]
  --vm_prefix TEXT  [default: mm-worker-]
  --help            Show this message and exit.  [default: False]
```

The following command gets the IP addresses of the VMs. Replace `<your-resource-group>` with the name of the Azure resource group containing your VMs.

```bash
cd scripts
python remote_file_management.py get-worker-ips --rg <your-resource-group> 1 8
```

In the rest of the guide, we assume that the *start_worker_id = 1* and *end_worker_id = 8*.

**⚠ Check vCPU quota for the subscription to find out the maximum number of VMs. We tested the infra with up to 12 VMs.**

This command also writes the ip addresses to *servers.toml*, *hosts.ansi* , and writes a *clusterssh* command for logging into VMs. These files/commands are used for job deployment, infrastructure upgrade, and remote monitoring respectively.

## Copy and execute init script

The *copy-init* command copies initialization and configuration scripts to Azure VMs.

```text
Usage: remote_file_management.py copy-init [OPTIONS] START_WORKER_ID
                                        END_WORKER_ID

 Options:
 --user TEXT       [default: millionmonkeys]
 --rg TEXT         [default: CSML-Credits-Budget-RG]
 --vm_prefix TEXT  [default: mm-worker-]
 --help            Show this message and exit.  [default: False]
```

The following command copies the init scripts to VMs

```bash
cd scripts
python remote_file_management.py copy-init --rg <your-resource-group> 1 8
```

After copying the init script,

Get ip addresses and the *clusterssh* command via

```bash
cd scripts
python remote_file_management.py get-worker-ips --rg <your-resource-group> 1 8
```

ssh to Azure VMs using the command printed in the above output. Remember to use csshx instead of clusterssh if you are on a Mac. (If you are not using clusterssh or csshx then ssh to each VM separately and repeat the following steps).

On the VMs, make init.sh an executable via

```bash
# run this on VM
chmod +x init.sh
```

Run *init.sh* and follow the instructions to install the code

```bash
# run this on VM
./init.sh -b <branch name> -u <user name when creating VM> -c ~/miniconda3
```

We use *fdisk* and *mkfs* linux utilities to create a filesystem for the datadisk created. When prompted

- type *n* for new partition

  ```text
  Command (m for help): n(ew)
  ```

- Press Enter for all steps with defaults

  ```text
  Select (default p):
  Partition number (1-4, default 1):
  First sector (2048-2145386495, default 2048):
  Last sector, +sectors or +size{K,M,G,T,P} (2048-2145386495, default 2145386495):
  ```

- type *w(rite)* to update partition tables

- *init.sh* continues

## Copy data files from local to Azure machines

We are not allowed to ship FFCWS data through the Github repo. Therefore, we need to transfer
the required files separately to the VMs.

```text
Usage: remote_file_management.py copy-data [OPTIONS] [SRC]... START_WORKER_ID
                                           END_WORKER_ID

Options:
  --user TEXT       [default: millionmonkeys]
  --rg TEXT         [default: CSML-Credits-Budget-RG]
  --vm_prefix TEXT  [default: mm-worker-]
  --help            Show this message and exit.  [default: False]
```

```bash
cd scripts
python remote_file_management.py copy-data --rg <your-resource-group> FF_allwaves_2020v2.dta FF_wave6_2020v2.dta 1 8
```

The above command copies the *dta* and *csv* files from ../data/dataset/fragile-families directory to the corresponding location on VMs with suffix in the range [1, 8].

## Deploying jobs from local machine

We now have a VM that can run jobs for this project. This section describes how we deploy a
big job to a pool of worker VMs. Given a jobfile `j` with `n` outcomes and `k` VMs, we create `k` copies of
`j` each with at most `n/k` outcomes. These sub jobfiles are distributed among the VMs. Before deploying the job, ensure that the codebase is updated and that the database service is running.

### Update codebase on remote machines

If required, update the code version on all worker machines via ansible.

```bash
cd scripts
ansible-playbook -i hosts.ansi --user <user name on VMs> --tags "update" --extra-vars="branchname=<branch name>" playbook.yml
```

### Check mongodb is running

**A common error for failed deployments is that the mongo server is not running.
Ensure that mongo is running by restarting the mongo service.**

Start mongo service

```bash
# run this on vm
sudo service mongod start
```

### Deploy job

Use "git fetch" and "git checkout" as shown in the "Local setup guide" document to select the appropriate branch on your local computer.

To deploy a job on VMs

```bash
python monkeys.py remote jobfiles/jobfile_monkeys.toml scripts/servers.toml
```

If the command was successful, you should get a message in your local terminal that includes "Command exited with status 0." The VMs will not appear to do anything unless you follow the "monitor" instructions below.

### Monitor job [Optional]

- Use the following command to simultaneously login to all VMs.

  ```bash
  cd scripts
  python remote_file_management.py get-worker-ips --rg <your-resource-group> 1 8
  # run the clusterssh command
  ```

- Now, use the following tmux command to see the status of job

  ```bash
  # run this on vm
  tmux at -t million-monkeys
  ```

### Export partial results while the job is still running [optional]

Use the following commands to export partial results while machine(s) are still running, if database type is set to "mongo":

```bash
# run this on vm
mkdir my_results
source ~/miniconda3/bin/activate ~/miniconda3/envs/million-monkeys
cd predicting-life-outcomes-ffcws
python monkeys.py export-to-json ~/my_results/ <database_name>
```

Replace <database_name> with the value from [results.database_name] in the jobfile.

Now the results are in a directory called my_results on the virtual machine(s). To download to your local computer, find the address of a single virtual machine using the get-worker-ips command above. Then use that address in the following command.

```bash
# run this on local terminal at any root directory
scp millionmonkeys@xx.xx.xxx.xx:~/my_results/results.json <file path on local machine where you want to save results>
```

If you want to download files other than results.json, the other available files in my_results will be: cv.json, drops.json, jobdesc.json, matrix_shapes.json, metalearning.json, timing.json.

Note: Downloading the partial results from a single VM to a local machine is slow, so we have not tried downloading the partial results from multiple VMs at the same time, but you may be able to do so.

## Copying results to local machine

### Path to results directory on VMs

The following command can be used to retrieve the path to results directory

```bash
# run this on vm
(ls -dt /datadrive/experiments/*/ | head -1)
```

Copy results from VMs to a local directory via

```bash
cd scripts
python remote_file_management.py download-run-results --rg <your-resource-group> <path to results directory on VMs i.e., output of above command> <../local/path> 1 8
```

## Delete VM and associated resources

### ⚠ This step is important. Leaving the machines on rather than turning them off or deleting them may cost several hundred dollars per day.

_(This cost estimate for leaving machines on is based on 8 machines as of December 2022.)_

Once the results are copied to local machine, the following command can be used to delete the VMs and associated resources (data disk, os disk, ip address) we created for each VM.

```bash
az vm delete --ids $(az vm list --query "[].id" -o tsv | grep "mm-worker")
```

Alternatively, these VMs can be deleted from Azure portal.
[Delete vm and associated resources from Azure portal](https://docs.microsoft.com/en-us/azure/virtual-machines/delete?tabs=portal2%2Cportal3%2Cportal4)
