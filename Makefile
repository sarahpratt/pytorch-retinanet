buildc:
	cd docker && docker build -t  ai2/ubuntu-pytorch-retinanet .

runc:
	docker run -v`pwd`:/root/pytorch-retinanet  --privileged -it ai2/ubuntu-pytorch-retinanet
	
