#!/bin/bash

git submodule update --init

conda create -n pirlnav python=3.7 cmake=3.14.0 -y

conda activate pirlnav

# Install torch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install habitat-lab
cd habitat-lab | pip install -e . | pip install -r requirements.txt | python setup.py develop --all

# Install habitat-sim
conda install habitat-sim=0.2.2 withbullet headless -c conda-forge -c aihabitat

# ln -s /media/disk12tb/carlos/data data
