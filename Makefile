CC = nvcc
CFLAGS = -arch sm_80 -lcublas -lnvidia-ml

all: my_program

my_program:test.cu
	$(CC) -gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_90,code=sm_90  -lcublas -lnvidia-ml -o my_program test.cu

clean:
	rm -f my_program
