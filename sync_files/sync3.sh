#!/bin/bash

rsync -avz \
  --exclude .idea \
  --exclude __pycache__/ \
  --exclude runs/ \
  --exclude .git/ \
  --exclude .DS_Store \
  --exclude csv_retinanet_12_80.pth \
  --exclude csv_retinanet_20.pth \
  --exclude venv/ \
  --exclude output/ \
  --exclude docker/ \
   ../pytorch-retinanet sarahp@172.16.0.162:/home/sarahp