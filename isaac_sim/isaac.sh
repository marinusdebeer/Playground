#!/bin/bash

# SSH into the server and execute commands remotely. This part is better executed manually due to the interactive nature of SSH.

SERVER_IP="129.153.226.168"
ssh -X -i ~/.ssh/lambda_ssh.pem ubuntu@$SERVER_IP << 'EOF'

cd robotics

# Install Chrome
# wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt-get install -f
sudo dpkg -i google-chrome-stable_current_amd64.deb

# Install Git
sudo apt-get update
sudo apt-get install -y git


sudo apt update
sudo apt install libfuse2

# Install VirtualGL
wget https://downloads.sourceforge.net/project/virtualgl/3.1.1/virtualgl_3.1.1_amd64.deb
sudo dpkg -i virtualgl_*.deb
sudo /opt/VirtualGL/bin/vglserver_config

# Install TurboVNC
# wget https://sourceforge.net/projects/turbovnc/files/3.1/turbovnc_3.1_amd64.deb
sudo dpkg -i turbovnc_*.deb

#For the following make sure it starts on display :1
/opt/TurboVNC/bin/vncserver :1 -depth 24 -geometry 1728x1117

# Open a random window to remove the black window issue
sudo apt-get update
sudo apt-get install lxde -y
DISPLAY=:1 startlxde &

# Clone the GitHub repository and set up the environment
git clone https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git
cd OmniIsaacGymEnvs
alias PYTHON_PATH=/home/ubuntu/robotics/isaac_sim-2023.1.1/python.sh
export PYTHON_PATH=$(eval echo $PYTHON_PATH)
source ~/.bashrc
PYTHON_PATH -m pip install -e .


# Run Training
cd omniisaacgymenvs
PYTHON_PATH scripts/rlgames_train.py task=Cartpole

EOF

#Utils
ps aux | grep X
sudo apt install xfce4 xfce4-goodies
sudo apt --fix-broken install

/opt/TurboVNC/bin/vncserver -kill :1
sudo apt-get install gnome-terminal


# Set up Display [OLD]
sudo apt update
sudo apt install xvfb -y
Xvfb :1 -screen 0 1536x960x24 &
export DISPLAY=:1
xhost +
sudo apt install x11vnc -y
sudo apt install xfce4-session
x11vnc -storepasswd qwerty ~/.vnc/passwd
echo 'ubuntu:Muybien1!' | sudo chpasswd
x11vnc -display :1 -auth ~/.Xauthority -forever -loop -noxdamage -repeat -rfbauth ~/.vnc/passwd -rfbport 5901 -shared