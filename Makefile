buildc:
	cd docker && docker build -t  ai2/ubuntu-pytorch-retinanet .

runc:
	docker run -v`pwd`:/root/pytorch-retinanet  --privileged --shm-size=16g -it ai2/ubuntu-pytorch-retinanet
	
 