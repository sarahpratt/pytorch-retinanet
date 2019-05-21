#!/bin/bash

rsync -avz \
  --exclude .idea \
  --exclude __pycache__/ \
  --exclude runs/ \
  --exclude .git/ \
  --exclude .DS_Store \
  --exclude venv/ \
  --exclude data/coco/ \
  --exclude data/custom/images_512/ \
   ../pytorch-retinanet sarahp@172.16.6.248:/home/sarahp