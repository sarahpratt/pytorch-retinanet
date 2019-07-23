buildc:
	cd docker && docker build -t  ai2/ubuntu-pytorch-retinanet .

runc:
	docker run -v`pwd`:/root/pytorch-retinanet -p 1234:8501 --privileged --shm-size=16g -it ai2/ubuntu-pytorch-retinanet
	
 