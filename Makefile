CC = nvcc
CFLAGS = -arch sm_80 -lcublas -lnvidia-ml

all: my_program

my_program:test.cu
	$(CC) -arch sm_80 -lcublas -lnvidia-ml -o my_program test.cu

clean:
	rm -f my_program
