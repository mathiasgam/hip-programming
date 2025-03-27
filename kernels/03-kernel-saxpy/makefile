.PHONY: all run

saxpy: saxpy.cpp
	CC -xhip -o saxpy saxpy.cpp 

run: saxpy
	srun --reservation=HIPcourse --account=project_462000877 --partition=small-g --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=1 ./saxpy

all: saxpy