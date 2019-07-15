#!/bin/bash
nvidia_version=`cat /proc/driver/nvidia/version |grep 'NVRM version:'| grep -oE "Kernel Module\s+[0-9.]+"| awk {'print $3'}`
driver_filename="NVIDIA-Linux-x86_64-$nvidia_version.run"
#driver_url="http://us.download.nvidia.com/XFree86/Linux-x86_64/$nvidia_version/$driver_filename"
#echo $driver_url
#wget $driver_url -P /root/
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux -P /root/
sh /root/cuda_10.0.130_410.48_linux  --extract=/root
sh /root/$driver_filename -s --no-kernel-module