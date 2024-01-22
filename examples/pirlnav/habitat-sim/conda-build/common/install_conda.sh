#!/bin/bash

set -ex

# Anaconda
wget -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x  Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
rm Miniconda3-latest-Linux-x86_64.sh
export PATH=/opt/conda/bin:$PATH
conda install -y anaconda-client git gitpython ninja conda-build # conda-build=3.18.9 # last version that works with our setup
conda remove -y --force patchelf
