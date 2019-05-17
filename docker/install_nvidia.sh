#!/bin/bash
nvidia_version=`cat /proc/driver/nvidia/version |grep 'NVRM version:'| grep -oE "Kernel Module\s+[0-9.]+"| awk {'print $3'}`
driver_filename="NVIDIA-Linux-x86_64-$nvidia_version.run"
driver_url="http://us.download.nvidia.com/XFree86/Linux-x86_64/$nvidia_version/$driver_filename"
echo $driver_url
wget $driver_url -P /root/
sh /root/$driver_filename -s --no-kernel-module
