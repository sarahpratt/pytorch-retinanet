FROM nvidia/cuda:8.0-devel-ubuntu14.04
RUN apt-get update; apt-get install -y inetutils-ping ssh net-tools isc-dhcp-client traceroute zip vim curl rsync lsof cron lsb-release moreutils xserver-xorg-core pciutils wget git python3-pip protobuf-compiler libprotoc-dev apt-file libboost1.55-dev libgflags-dev libgoogle-glog-dev libatlas-base-dev libhdf5-dev libleveldb-dev liblmdb-dev libopencv-core-dev libopencv-highgui-dev libpython2.7-dev libboost-python1.55-dev libboost-thread1.55-dev libboost-system1.55-dev libboost-filesystem1.55-dev libsnappy-dev libopencv-imgproc-dev python-numpy python-pip gfortran python-skimage python-protobuf cuda-samples-8-0
COPY install_nvidia.sh /root/install_nvidia.sh
RUN /root/install_nvidia.sh
CMD /bin/bash

