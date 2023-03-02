##WINDOWS
conda create --name tfdml_plugin python=3.9 
conda activate tfdml_plugin
pip install tensorflow-cpu==2.10
pip install tensorflow-directml-plugin

#MAC
conda create --name gpu_venv python=3.8
conda activate gpu_venv
conda install -c apple tensorflow-deps==2.10.0
python -m pip install tensorflow-macos==2.10.0
python -m pip install tensorflow-metal==0.6.0