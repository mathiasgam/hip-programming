.PHONY: all run

copy: copy.cpp
	$(CXX) $(CXXFLAGS) -o copy copy.cpp 

run: copy
	srun --reservation=HIPcourse --account=project_462000877 --partition=small-g --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=1 ./copy

profile: copy
	srun --reservation=HIPcourse --account=project_462000877 --partition=small-g --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=1 rocprof -i metrics.txt -o metrics.csv ./copy

all: copy