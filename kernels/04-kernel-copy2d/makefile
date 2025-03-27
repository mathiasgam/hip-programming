.PHONY: all run

copy2d: copy2d.cpp
	CC -xhip -o copy2d copy2d.cpp 

run: copy2d
	srun --reservation=HIPcourse --account=project_462000877 --partition=small-g --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=1 ./copy2d

all: copy2d