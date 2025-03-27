# My Solution

Important to setup the environment first

```bash
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
```

Then compile using `CC -xhip -o hello helo.cpp`

If runned normally, using `./hello` no GPUs will be available, so using `srun` instead is neccesary.

```bash
srun --reservation=HIPcourse --account=project_462000877 --partition=small-g --time=00:05:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-task=1 ./hello > log.txt
```
This will create a small job on a GPU node, with one GPU allocated. The output is piped into `log.txt`, but can also be outputted directly to the terminal.

Output:
```bash
Hello! I am GPU 0 out of 1 GPUs in total.
```
