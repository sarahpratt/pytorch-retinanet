FROM nvidia/cuda:8.0-devel-ubuntu16.04
RUN apt-get update; apt-get install -y module-init-tools zip vim curl rsync cron lsb-release pciutils wget git  apt-file libpython2.7-dev  python-numpy python-pip cuda-samples-8-0 tk-dev python-tk python-skimage
RUN pip install opencv-python cffi pandas cython requests matplotlib==2.2.2
RUN pip install pycocotools 
RUN pip install torch==0.4.1
RUN pip install torchvision==0.2.1
COPY install_nvidia.sh /root/install_nvidia.sh
RUN /root/install_nvidia.sh
CMD /bin/bash

