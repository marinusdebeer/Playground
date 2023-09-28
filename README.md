##WINDOWS
conda create --name tfdml_plugin python=3.9 
conda activate tfdml_plugin
pip install tensorflow-cpu==2.10
pip install tensorflow-directml-plugin

#MAC
#https://developer.apple.com/metal/tensorflow-plugin/
pip install tensorflow
pip install tensorflow-macos
pip install tensorflow-metal