# Use Isaac Sim image as the base
FROM nvcr.io/nvidia/isaac-sim:2023.1.1

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone your fork of the repository
RUN git clone https://github.com/marinusdebeer/OmniIsaacGymEnvs.git /app

# Install your Python package from the repository
WORKDIR /app
RUN /isaac-sim/python.sh -m pip install -e .

# Your additional setup here

# Set an entrypoint if you wish to run `/isaac-sim/python.sh` directly
ENTRYPOINT ["/isaac-sim/python.sh"]

# Optionally set a default command, for example running a specific script
# CMD ["/app/scripts/your_script.py"]
