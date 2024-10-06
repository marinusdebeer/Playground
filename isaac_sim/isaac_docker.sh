# sudo apt install build-essential -y
# wget https://us.download.nvidia.com/XFree86/Linux-x86_64/525.85.05/NVIDIA-Linux-x86_64-525.85.05.run
# chmod +x NVIDIA-Linux-x86_64-525.85.05.run
# sudo ./NVIDIA-Linux-x86_64-525.85.05.run
# curl -fsSL https://get.docker.com -o get-docker.sh
# sudo sh get-docker.sh

# curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
#   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
#     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
#   && \
#     sudo apt-get update

# sudo apt-get install -y nvidia-container-toolkit
# sudo systemctl restart docker

# Verify NVIDIA Container Toolkit
# docker run --rm --gpus all ubuntu nvidia-smi

sudo groupadd docker
sudo usermod -aG docker ubuntu
newgrp docker
docker login nvcr.io -u '$oauthtoken' -p 'ZHM5aWVjamdxam5wOHR0c3JoMG1qYWhrbGY6OWUxZmNhNmEtOTI4Ny00ZTI1LWE1NWUtZDM0NzAxZDlmOGYx'
docker pull nvcr.io/nvidia/isaac-sim:2023.1.1
docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
    -e "PRIVACY_CONSENT=Y" \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    nvcr.io/nvidia/isaac-sim:2023.1.1


# xhost +
# docker run --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
#   -e "PRIVACY_CONSENT=Y" \
#   -v $HOME/.Xauthority:/root/.Xauthority \
#   -e DISPLAY \
#   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
#   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
#   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
#   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
#   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
#   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
#   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
#   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
#   nvcr.io/nvidia/isaac-sim:2023.1.1 \
#   ./runapp.sh