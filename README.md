# INSTALL DEPENDENCIES
## FOR MAC

### 1: Install the Edge TPU runtime

`curl -LO https://github.com/google-coral/libedgetpu/releases/download/release-grouper/edgetpu_runtime_20221024.zip`

`unzip edgetpu_runtime_20221024.zip`

`cd edgetpu_runtime`

`sudo bash install.sh`

## 2: Install the PyCoral library
`python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0`

`python3 -m pip install -r requirements.txt`



## FOR WINDOWS

### 1: Install the Edge TPU runtime

https://coral.ai/docs/accelerator/get-started#runtime-on-windows

### 2: Install the PyCoral library

`python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0`

`python3 -m pip install -r requirements.txt`

# Running The Script
`Add a random file to the models folder to skip that model when running multiple evalutation`