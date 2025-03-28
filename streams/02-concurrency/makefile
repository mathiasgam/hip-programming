.PHONY: all run

streams: streams.cpp
	$(CXX) $(CXXFLAGS) -o streams streams.cpp 

run: streams
	srun --reservation=HIPcourse --account=project_462000877 --partition=small-g --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=1 ./streams

all: streams