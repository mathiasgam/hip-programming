

GPU_RUN = srun --account=project_462000877 --partition=dev-g -N1 -n2 --cpus-per-task=1 --gpus-per-task=2 --time=00:15:00

all: pingpong

# ping-pong: ping-pong.o
#	 $(HIPCC) $(LDFLAGS) -o $@ $< $(LIBS)
pingpong: ping-pong.cpp
	CC -xhip -o pingpong ping-pong.cpp

%.o: %.cpp
	# $(MPICXXENV) $(MPICXX) $(MPICXXFLAGS) -c -o $@ $<

.PHONY: clean run
clean:
	rm -f *.o pingpong
	rm -rf bin/

run: pingpong
#	srun --reservation=HIPcourse --account=project_462000877 --partition=small-g --time=00:05:00 --nodes=1 --ntasks-per-node=2 --cpus-per-task=1 --gpus-per-task=1 bin/ping-pong
	MPICH_GPU_SUPPORT_ENABLED=1 srun --account=project_462000877 --partition=dev-g --nodes=2 --tasks-per-node=1 --cpus-per-task=1 --gpus-per-task=2 --time=00:15:00 ./pingpong
