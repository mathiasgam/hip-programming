TARGETS = bin/vector-sum

# GPU_RUN = srun --reservation=HIPcourse --account=project_462000877 --partition=small-g --time=00:05:00 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=2

GPU_RUN = srun --account=project_462000877 --partition=dev-g -N1 -n1 --cpus-per-task=1 --gpus-per-task=2 --time=00:15:00

.PHONY: all clean rebuild run profile
all: $(TARGETS)

clean:
	rm -rf bin/

rebuild: clean all

bin/%: %.cpp
	@mkdir -p bin/
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGETS)
	$(GPU_RUN) bin/vector-sum
