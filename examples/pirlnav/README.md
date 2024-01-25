# Deploy PIRLNav project in VSN framework


1. **Follow the installation instructions of our ROS4VSN library.** See our [README.md](README.md)

2. **Install habitat-lab and habitat-sim versions 0.2.2** by executing the following commands:
    ```
   git clone --branch v0.2.2 https://github.com/facebookresearch/habitat-lab.git
   git clone --branch v0.2.2 https://github.com/facebookresearch/habitat-sim.git
    ```
   ```
   conda create -n pirlnav python=3.7 cmake=3.14.0
   conda activate pirlnav
   ```
   ```
   cd habitat-sim/
   pip install -r requirements.txt
   ./build.sh --headless

   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    ```
   ```
   cd habitat-lab/
   pip install -r requirements.txt
   pip install -e .
   ```


3. **The repository should look like this:**  
ros4vsn/  
├── catkin_ws  
│   └── src  
│   │       ├── discrete_move  
│   │       └── vsn  
├── examples  
│   └── pirlnav  
│   │       ├── configs  
│   │       ├── habitat-lab  
│   │       ├── habitat-sim  
│   │       ├── pirlnav  
│   │       ├── ros4pirlnav  
│   │       └── scripts  
├── imgs  
└── scripts  
 

4. **Download the pre-trained PIRLNav model provided in the official PIRLNav repository**. Download it from [here](https://habitat-on-web.s3.amazonaws.com/pirlnav_release/checkpoints/objectnav_rl_ft_hd.ckpt).

5. Before running the project you must **download HM3D Scene and Episode Datasets**.

- Download the HM3D dataset using the instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) (download the full HM3D dataset for use with habitat)

- Download the ObjectNav HM3D episode dataset from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets).

6. **Copy the content inside our folder [ros4pirlnav](ros4pirlnav) in [catkin_ws\vsn](../../catkin_ws/src/vsn/scripts)** .



