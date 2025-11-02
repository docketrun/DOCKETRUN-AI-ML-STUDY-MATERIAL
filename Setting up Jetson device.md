# Prerequisite
```sh
snap download snapd --revision=24724
```
```sh
sudo snap ack snapd_24724.assert
```
```sh
sudo snap install snapd_24724.snap
```
```sh
sudo sudo snap refresh --hold snapd
```
```sh
sudo apt-get install git cmake libpython3-dev python3-numpy
```
```sh
sudo apt-get install libnumpy-dev
```
```sh
wget -O - https://repo.download.nvidia.com/jetson/common/pool/main/n/nvidia-l4t-dla-compiler/nvidia-l4t-dla-compiler_36.4.1-20241119120551_arm64.deb | dpkg-deb --fsys-tarfile - | sudo tar xv --strip-components=5 --directory=/usr/lib/aarch64-linux-gnu/nvidia/ ./usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so
```
# Opencv
```sh
pip3 install opencv-python
```
# Ultralytics
```sh
pip3 install ultralytics
```
```sh
pip3 install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
```
```sh
pip3 install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```
```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
```
```sh
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```
```sh
sudo apt-get update
```
```sh
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```
```sh
pip3 install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```
```sh
pip3 install numpy==1.23.5
```
