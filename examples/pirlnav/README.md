# Deploy PIRLNav project in VSN framework


1. Download and install Evaluation of Visual Semantic Navigation Models in Real Robots. Follow our [README.md](README.md)


2. Then, we have to download the **PIRLNav: Pretraining with Imitation and RL Finetuning for ObjetcNav to deploy in a robot**. 

```
git clone https://github.com/Ram81/pirlnav.git
git submodule update --init

conda create -n pirlnav python=3.7 cmake=3.14.0

cd habitat-sim/
pip install -r requirements.txt
./build.sh --headless

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

cd habitat-lab/
pip install -r requirements.txt

pip install -e habitat-lab
pip install -e habitat-baselines

pip install -e .
```

3. Before, run the project you have to download HM3D Scene and Episode Dataset.

- Download the HM3D dataset using the instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) (download the full HM3D dataset for use with habitat)

- Download the ObjectNav HM3D episode dataset from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets).


