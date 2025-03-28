TARGETS = bin/copy bin/matrix_transpose_naive bin/matrix_transpose_with_SM_nobc bin/matrix_transpose_with_SM

GPU_RUN = srun --reservation=HIPcourse --account=project_462000877 --partition=small-g --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=1

.PHONY: all clean rebuild run profile
all: $(TARGETS)

clean:
	rm -rf bin/

rebuild: clean all

bin/%: %.cpp
	@mkdir -p bin/
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGETS)
	$(GPU_RUN) ./bin/copy
	$(GPU_RUN) ./bin/matrix_transpose_naive
	$(GPU_RUN) ./bin/matrix_transpose_with_SM
	$(GPU_RUN) ./bin/matrix_transpose_with_SM_nobc

profile: $(TARGETS)
	$(GPU_RUN) rocprof -i metrics.txt -o metrics0.csv ./bin/copy
	$(GPU_RUN) rocprof -i metrics.txt -o metrics1.csv ./bin/matrix_transpose_naive
	$(GPU_RUN) rocprof -i metrics.txt -o metrics2.csv ./bin/matrix_transpose_with_SM
	$(GPU_RUN) rocprof -i metrics.txt -o metrics3.csv ./bin/matrix_transpose_with_SM_nobc
