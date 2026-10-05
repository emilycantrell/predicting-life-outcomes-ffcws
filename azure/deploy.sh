#!/usr/bin/env bash
set -euo pipefail

usage()
{
    echo "Usage : ./deploy.sh -u user -p prefix -k ssh_pub_key_file -s starting_suffix -n num_workers -g resource_group"
    exit 0
}

prefix="mm-worker-"
ssh_pub_key_file="~/.ssh/id_rsa.pub"
starting_suffix=1
resource_group="CSML-Credits-Budget-RG"
user="millionmonkeys"

while getopts hp:u:k:s:n:g: flag
do
    case "${flag}" in
        p) prefix=${OPTARG:-mm-worker-};;
        u) user=${OPTARG:-millionmonkeys};;
        k) ssh_pub_key_file=${OPTARG:=~/.ssh/id_rsa.pub};;
        s) starting_suffix=${OPTARG:=1};;
        n) num_workers=${OPTARG:=1};;
        g) resource_group=${OPTARG:=1};;
        h) usage;;
    esac
done

#az deployment group create --resource-group CSML-Credits-Budget-RG --template-file worker.bicep
ending_suffix=$starting_suffix+$num_workers
key=$(<$ssh_pub_key_file)
echo $key

GREEN=$'\e[0;32m'
RED=$'\e[0;31m'
NC=$'\e[0m'

deploy() {
    az deployment group create --resource-group $resource_group --template-file worker.bicep --parameters adminUsername=$user vmName=$prefix$1 adminPasswordOrKey="$key"
}

for (( c=$starting_suffix; c<$ending_suffix; c++ ))
do  
    deploy $c
done
