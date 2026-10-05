#!/usr/bin/env bash
set -euo pipefail

while getopts b:u:c: flag
do
  case "${flag}" in
    b) branch=${OPTARG};;
    u) user=${OPTARG};;
    c) condapath=${OPTARG};;
  esac
done

#!/usr/bin/env bash

##################################################
# Install miniconda without saving the install script
# Borrowed from https://gist.githubusercontent.com/mherkazandjian/cce01cf3e15c0b41c1c4321245a99096/raw/03c86dae9a212446cf5b095643854f029b39c921/miniconda_installer.sh | bash -s -- ~/miniconda
##################################################

# the absolute path of the miniconda installer (e.g Miniconda3-latest-Linux-x86_64.sh)
INSTALLER_PATH=/tmp/Miniconda3-latest-Linux-x86_64.sh

# the installation root directory of miniconda (e.g /opt/conda)
INSTALL_PREFIX=$condapath


function _install_miniconda() {
    wget -q https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ${INSTALLER_PATH}
    chmod +x ${INSTALLER_PATH}
    ${INSTALLER_PATH} -b -p ${INSTALL_PREFIX}
    rm ${INSTALLER_PATH}
}


function step1_install_dependencies {
    git clone https://github.com/emilycantrell/predicting-life-outcomes-ffcws.git
    sudo mkdir /datadrive;
    sudo fdisk /dev/sdc;
    sudo mkfs -t ext4 /dev/sdc1;
    sudo mount /dev/sdc1 /datadrive;
    sudo -i blkid;
    uuid=$(sudo -i blkid | grep sdc1 | awk -F"=" '{print $2}' | awk '{print $1'} |  sed -r 's/^"|"$//g')
    echo $uuid;
    UUID_LINE="${uuid}   /datadrive   ext4   defaults,nofail   1   2";
    echo $UUID_LINE;
    echo $UUID_LINE | sudo tee -a /etc/fstab;
    #sudo reboot
    sudo mkdir /datadrive/experiments;
    sudo chgrp $user /datadrive/experiments/ -R;
    sudo chown $user /datadrive/experiments/ -R;
    sudo mkdir /datadrive/mongodb;
    wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | sudo apt-key add -;
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.2 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.2.list;
    sudo apt-get update;
    sudo apt-get install -y mongodb-org;
    sudo chown mongodb /datadrive/mongodb/ -R;
    sudo chgrp mongodb /datadrive/mongodb/ -R;
    echo "open /etc/mongod.conf and update the dbPath field to /datadrive/mongodb";
    echo "wait";
    sock_file="/tmp/mongodb-27017.sock"
    if test -f "$sock_file"; then
        sudo chown mongodb:mongodb $sock_file
        sudo systemctl restart mongod
    fi
    cd ~
    echo "Install Miniconda"
    _install_miniconda
    source ~/.bashrc
}

function step2_install_monkeys_infra {
    cd ~/predicting-life-outcomes-ffcws
    git checkout $branch
    git pull
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
    conda env create -f environment_pinned_linux.yml
}

function step3_check_installation {
    echo "run tests"
    cd ~/predicting-life-outcomes-ffcws
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
    conda activate million-monkeys
    python -m pytest -v -p no:warnings --regression ./test/desc_files/regression --incorrect ./test/desc_files/incorrect --regression ./test/desc_files/regression_missing --imperfect_regression ./test/desc_files/imperfect_regression --reproducibility ./test/desc_files/reproducibility --metadata_warning ./test/desc_files/metadata_warning --metadata_concepts ./test/desc_files/mm_concepts --metalearning ./test/desc_files/metalearning
    python -m pytest models/
    python -m pytest data/test_preprocessing.py
}

 step1_install_dependencies
 step2_install_monkeys_infra
 step3_check_installation
