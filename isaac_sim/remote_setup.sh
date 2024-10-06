#!/bin/bash
set -x

ssh -X -i ~/.ssh/lambda_ssh.pem ubuntu@150.136.222.3
ssh -L 5901:localhost:5901 -C -N -i ~/.ssh/lambda_ssh.pem ubuntu@150.136.222.3

cd robotics
sudo apt-get update && sudo apt-get upgrade -y
sudo dpkg --configure -a
sudo apt-get install -y libfuse2 libglu1-mesa lxde git ubuntu-desktop
sudo dpkg -i startup/virtualgl_3.1.1_amd64.deb
sudo dpkg -i startup/turbovnc_3.1_amd64.deb
sudo /opt/VirtualGL/bin/vglserver_config -config +s +f -t
sudo systemctl restart gdm3
sudo gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 0
gsettings set org.gnome.desktop.screensaver idle-activation-enabled false

echo 'export PATH="/home/ubuntu/robotics/anaconda3/bin:$PATH"' >> ~/.bashrc

alias isaac_python=/home/ubuntu/robotics/omni/library/isaac_sim-2023.1.1/python.sh

echo 'ubuntu:tYwwum1!' | sudo chpasswd
sudo reboot

# Change power - screen blank to NEVER
/opt/TurboVNC/bin/vncserver -kill :1
/opt/TurboVNC/bin/vncserver :1 -depth 24 -geometry 1728x1117

ps aux | grep X
xhost +local:root

sudo groupadd docker
sudo usermod -aG docker ubuntu
newgrp docker
sudo systemctl restart docker
docker login nvcr.io -u '$oauthtoken' -p 'ZHM5aWVjamdxam5wOHR0c3JoMG1qYWhrbGY6OWUxZmNhNmEtOTI4Ny00ZTI1LWE1NWUtZDM0NzAxZDlmOGYx'
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1

echo "alias isaac='docker run --name isaac --entrypoint bash -it --gpus all -e \"ACCEPT_EULA=Y\" --network=host \
    -e \"PRIVACY_CONSENT=Y\" \
    -e DISPLAY=\$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/robotics:/isaac-sim/robotics \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    nvcr.io/nvidia/isaac-sim:2023.1.1'" >> ~/.bashrc


export ISAACSIM_PATH="/home/ubuntu/robotics/omni/library/isaac_sim-2023.1.1"
export ISAACSIM_PYTHON_EXE="/home/ubuntu/robotics/omni/library/isaac_sim-2023.1.1/python.sh"
alias isaac_py="/home/ubuntu/robotics/omni/library/isaac_sim-2023.1.1/python.sh"

conda init
source ~/.bashrc
conda activate isaac-sim
conda activate orbit

docker commit isaac repository/isaac_image
docker save -o ~/robotics.tar nvcr.io/nvidia/isaac-sim:2023.1.1
docker load -i robotics.tar

docker commit a632aa012a20 isaac_image
docker save -o ~/robotics/isaac_image.tar isaac_image
docker load -i ~/robotics/isaac_image.tar

echo "alias isaac2='docker run --name isaac --entrypoint bash -it --gpus all -e \"ACCEPT_EULA=Y\" --network=host \
    -e \"PRIVACY_CONSENT=Y\" \
    -e DISPLAY=\$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/robotics:/isaac-sim/robotics \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    isaac_image'" >> ~/.bashrc
    
source ~/.bashrc

./python.sh -m pip install -e robotics/OmniIsaacGymEnvs/
#Train from scratch
isaac_python robotics/OmniIsaacGymEnvs/omniisaacgymenvs/scripts/rlgames_train.py task=Anymal
# Train from checkpoint
./python.sh robotics/OmniIsaacGymEnvs/omniisaacgymenvs/scripts/rlgames_train.py task=AnymalTerrain checkpoint=robotics/OmniIsaacGymEnvs/omniisaacgymenvs/runs/AnymalTerrain/nn/last_AnymalTerrain_ep_2000_rew_17.841707.pth
# Demo from checkpoint
./python.sh robotics/OmniIsaacGymEnvs/omniisaacgymenvs/scripts/rlgames_demo.py task=Humanoid checkpoint=robotics/OmniIsaacGymEnvs/omniisaacgymenvs/runs/Humanoid/nn/last_Humanoid test=True num_envs=64
./python.sh robotics/OmniIsaacGymEnvs/omniisaacgymenvs/scripts/rlgames_demo.py task=AnymalTerrain checkpoint=robotics/OmniIsaacGymEnvs/omniisaacgymenvs/runs/AnymalTerrain/nn/last_AnymalTerrain_ep_3000_rew_21.921618.pth test=True num_envs=64
# Demo from omniverse checkpoint
./python.sh robotics/OmniIsaacGymEnvs/omniisaacgymenvs/scripts/rlgames_demo.py task=AnymalTerrain num_envs=64 checkpoint=robotics/anymal_terrain.pth 

# EOF

# Notes
# - Use conda for packages
# - See if more of the GPU or CPU's can be utilized for the simulation and training, maybe it is a docker problem (maybe run isaac-sim installed on the host to compare it to docker installation)

docker attach isaac
docker restart isaac
docker rm isaac
ctrl-p ctrl-q
telnet localhost 8223

logs/skrl/franka_reach/2024-04-13_14-48-11/checkpoints/agent_9600.pt
# alias PYTHON_PATH=/isaac-sim/python.sh

# docker exec isaac-sim python /isaac-sim/python.sh
# sudo visudo # ubuntu ALL=(ALL) NOPASSWD: ALL
# sudo nano /etc/gdm3/custom.conf
#start
# AutomaticLoginEnable = true
# AutomaticLogin = ubuntu
#end

# PYTHON_PATH -m pip install --upgrade pip
# PYTHON_PATH -m pip install -e .
# cd omniisaacgymenvs
# PYTHON_PATH scripts/rlgames_train.py task=Anymal