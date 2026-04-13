# Accelerated Thermal Simulation of High-Power Electronic Systems

This project simulates **2D anisotropic heat diffusion** in a multi-layer IC/package stack and compares performance across:
- serial CPU
- OpenMP CPU
- CUDA GPU
- hybrid MPI + CUDA + OpenMP (1D Y-slab decomposition)

## Project layout

Main source files live in `src/`:

- `heat.h`  
  Shared header: constants, `Grid` struct, material tables, and function declarations.

- `main.c`  
  Orchestrator: argument parsing, initialization, timestep loop, snapshots, timing, CSV output.

- `stencil_cpu.c`  
  CPU stencil implementations (`apply_stencil_serial`, `apply_stencil_omp`) and material property tables.

- `stencil_cuda.cu`  
  CUDA kernels (naive + tiled) and GPU memory/launch helpers.

- `halo.c`  
  MPI halo exchange routines for top/bottom ghost-row communication between ranks.

- `Makefile`  
  Builds all execution modes and includes a scaling-study target.

- `visualize.py`  
  Reads `results/scaling.csv` and prints summary statistics by variant.

## Build and run in MobaXterm

Open MobaXterm terminal, connect to your Linux host, then run:

```bash
cd ~/Accelerated-Thermal-Simulation-of-High-Power-Electronic-Systems/src
```

### 1) Build all executables

```bash
make clean
make
```

This creates:
- `heat_serial`
- `heat_omp`
- `heat_cuda`
- `heat_hybrid`

### 2) Run each version

#### Serial CPU
```bash
./heat_serial --nx 1024 --ny 1024 --steps 1000 --snap-every 200 --outdir results_serial
```

#### OpenMP CPU
```bash
export OMP_NUM_THREADS=16
./heat_omp --nx 1024 --ny 1024 --steps 1000 --snap-every 200 --outdir results_omp
```

#### CUDA GPU
```bash
./heat_cuda --nx 4096 --ny 4096 --steps 1000 --snap-every 250 --outdir results_cuda
```

#### Hybrid MPI + CUDA + OpenMP
```bash
export OMP_NUM_THREADS=8
mpirun -np 2 ./heat_hybrid --nx 4096 --ny 4096 --steps 1000 --snap-every 250 --outdir results_hybrid
```

## GPU architecture selection

Default is `ARCH=sm_80` (A100). For V100 use `sm_70`:

```bash
make clean
make ARCH=sm_70
```

## Scaling study workflow

Run automated sweep (`nx` = 1024, 2048, 4096) and summarize:

```bash
make scaling-study
```

This appends rows to `results/scaling.csv` and runs:

```bash
python3 visualize.py --mode scaling
```

## Python visualization commands (exact)

Run from `src/`:

```bash
cd ~/Accelerated-Thermal-Simulation-of-High-Power-Electronic-Systems/src
```

Scaling summary mode (reads timing CSV):

```bash
python3 visualize.py --mode scaling --csv results/scaling.csv
```

Heatmap slider mode (interactive temperature-over-time view):

```bash
python3 visualize.py --mode heatmap --outdir results_cuda
```

Hybrid output example:

```bash
python3 visualize.py --mode heatmap --outdir results_hybrid
```

Optional colormap override:

```bash
python3 visualize.py --mode heatmap --outdir results_cuda --cmap inferno
```

## Expected outputs (explicit)

Running the executables produces both **terminal output** and **files on disk**.

### 1) Terminal output (stdout)

Each run prints one CSV-style timing line:

```text
version,nx,ny,nranks,nsteps,wall_time_s
```

Example:

```text
cuda,4096,4096,1,1000,1.234567
```

`version` is one of: `serial`, `omp`, `cuda`, `hybrid`.

### 2) Output files created

When you run with `--outdir <folder>`, the program writes into that folder:

- `metadata.txt`
  - text file with run settings (`nx`, `ny`, `nranks`, `dx`, `dy`, `dt`, `nsteps`, `snap_every`)
- `rank_RRR_step_SSSSSS.bin`
  - binary snapshot files (32-bit float temperatures)
  - one file per rank per snapshot step
  - written at step `0` and then every `snap_every` steps

For example, if `--outdir results_cuda` and one rank:
- `results_cuda/metadata.txt`
- `results_cuda/rank_000_step_000000.bin`
- `results_cuda/rank_000_step_000250.bin`
- `results_cuda/rank_000_step_000500.bin`
- ...

### 3) Scaling-study files

`make scaling-study` additionally creates/appends:

- `results/scaling.csv`  
  Timing rows from serial/omp/cuda/hybrid scaling runs.

## Parameter meanings and how to choose them

These runtime arguments control simulation size, runtime cost, and output frequency:

- `--nx`  
  Number of grid cells in the X direction (left/right).
- `--ny`  
  Number of grid cells in the Y direction (bottom/top).
- `--steps`  
  Number of simulation time steps (how long we march forward in time).
- `--snap-every`  
  Save one snapshot every N steps (I/O frequency).

How they affect performance:

- Larger `nx`/`ny` -> more cells per step -> more compute and memory traffic.
- Larger `steps` -> more loop iterations -> roughly linear increase in runtime.
- Smaller `snap-every` -> more file writes -> higher I/O overhead (can reduce speedup).

How they relate to the heat PDE:

- `nx` and `ny` define spatial discretization resolution of the 2D heat field.
- `steps` defines total simulated time (`t_final ~ steps * dt`).
- `snap-every` does not change PDE physics; it only changes output cadence.

Practical parameter selection:

- Quick debug run: `--nx 512 --ny 512 --steps 200 --snap-every 100`
- CPU comparison run: `--nx 1024 --ny 1024 --steps 1000 --snap-every 200`
- GPU/hybrid benchmark run: `--nx 4096 --ny 4096 --steps 1000 --snap-every 250`
