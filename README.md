# 3D-CycleGan-Pytorch-Medical-Imaging-Translation

Pytorch pipeline for 3D image domain translation using Cycle-Generative-Adversarial-networks.  
*******************************************************************************
## Requirements
We download the official MONAI DockerHub, with the latest MONAI version. Please visit https://docs.monai.io/en/latest/installation.html
Additional packages can be installed with "pip install -r requirements.txt"
*******************************************************************************
## Python scripts and their function

- organize_folder_structure.py: Organize the data in the folder structure (training,testing) for the network.

- check_loader_patches: Shows example of patches fed to the network during the training.

- options_folder/base_options.py: List of base_options used to train/test the network.  

- options_folder/train_options.py: List of specific options used to train the network.

- options_folder/test_options.py: List of options used to test the network.

- models_folder: the folder contains the scripts with the networks and the cycle-gan training architecture.

- train.py: Runs the training. (Set the base/train options first)

- test.py: It launches the inference on a single input image chosen by the user. (Set the base/train options first)