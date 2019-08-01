buildc:
	cd docker && docker build -t  ai2/ubuntu-pytorch-retinanet .

runc:
	docker run -v`pwd`:/root/pytorch-retinanet -v /mnt/mag5tb/sarahp/runs:/root/pytorch-retinanet/runs -p 1234:8501 --privileged --ipc=host -it ai2/ubuntu-pytorch-retinanet
	
 