CC = nvcc
CFLAGS = -arch sm_80 -lcublas -lnvidia-ml

all: my_program
old: my_program_old
stream:stream
api:api

my_program:test.cu
	$(CC) -gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_90,code=sm_90  -lcublas -lnvidia-ml -o my_program test.cu

api:tc.cu
	$(CC) -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90  -lcublas -lnvidia-ml -o api tc.cu

my_program_old:test.cu
	$(CC) -gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70 -lcublas -lnvidia-ml -o my_program_old test.cu

stream:stream.cu
	$(CC) --std=c++11 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_90,code=sm_90  -lcublas -lnvidia-ml -o stream stream.cu


clean:
	rm -f my_program
