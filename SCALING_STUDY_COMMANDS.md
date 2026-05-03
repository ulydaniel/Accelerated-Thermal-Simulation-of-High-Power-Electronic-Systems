# Scaling Study Commands

This guide shows exactly how to compile and run the scaling sweep from `src/Makefile`.

## 1) Go to the build directory

```bash
cd src
```

## 2) Compile all executables

```bash
make clean
make all
```

This builds:

- `heat_serial`
- `heat_omp`
- `heat_cuda`
- `heat_hybrid`

## 3) Run scaling study (default settings)

```bash
make scaling-study
```

Default sweep values come from the Makefile:

- `SCALING_NX_LIST="512 1024 2048 4096 8192 16384 32768 65536"`
- `SCALING_STEPS=200`
- `SCALING_SNAP_EVERY=1`
- `SCALING_MPI_NP=2`
- `SCALING_BLOCK_X=16` (defaults to `TILE_WIDTH`)
- `SCALING_BLOCK_Y=16` (defaults to `TILE_WIDTH`)
- `SCALING_VARIANTS="serial omp cuda hybrid"`
- `SCALING_THREADS_LIST="1 2 3 ... 32"`

Results are appended to:

- `results/scaling.csv`
- CSV columns: `version,nx,ny,nranks,nsteps,wall_time_s,omp_threads,block_x,block_y`

## 4) One solid command with custom parameters

Use this single command form to override any scaling-study variable:

```bash
make scaling-study \
  SCALING_VARIANTS="serial omp cuda hybrid" \
  SCALING_NX_LIST="1024 2048 4096 8192" \
  SCALING_STEPS=500 \
  SCALING_SNAP_EVERY=10 \
  SCALING_MPI_NP=4 \
  SCALING_BLOCK_X=32 \
  SCALING_BLOCK_Y=8 \
  SCALING_THREADS_LIST="1 2 4 8 16 32"
```

## 5) Parameter reference

- `SCALING_VARIANTS`
  - Which versions to run.
  - Allowed values: `serial`, `omp`, `cuda`, `hybrid`
  - Example: `SCALING_VARIANTS="cuda hybrid"`

- `SCALING_NX_LIST`
  - Space-separated list of square grid sizes.
  - For each value `nx`, the run uses `--nx nx --ny nx`.
  - Example: `SCALING_NX_LIST="2048 4096 8192"`

- `SCALING_STEPS`
  - Number of time steps passed as `--steps`.
  - Example: `SCALING_STEPS=1000`

- `SCALING_SNAP_EVERY`
  - Snapshot interval passed as `--snap-every`.
  - Lower values write snapshots more frequently.
  - Example: `SCALING_SNAP_EVERY=20`

- `SCALING_MPI_NP`
  - MPI rank count used for the `hybrid` variant (`mpirun -np`).
  - One rank maps to one GPU in this project.
  - Example: `SCALING_MPI_NP=8`

- `SCALING_BLOCK_X`
  - CUDA block X dimension passed as `--block-x` for `cuda` and `hybrid`.
  - Example: `SCALING_BLOCK_X=32`

- `SCALING_BLOCK_Y`
  - CUDA block Y dimension passed as `--block-y` for `cuda` and `hybrid`.
  - Example: `SCALING_BLOCK_Y=8`

- `SCALING_THREADS_LIST`
  - OMP thread counts swept for `omp` and `hybrid` variants.
  - Each value sets `OMP_NUM_THREADS` for one run.
  - Example: `SCALING_THREADS_LIST="2 4 8 16"`

## 6) Common ready-to-run examples

CUDA only, quick run:

```bash
make scaling-study SCALING_VARIANTS="cuda" SCALING_NX_LIST="2048 4096" SCALING_STEPS=100
```

Hybrid only, MPI/OMP sweep:

```bash
make scaling-study SCALING_VARIANTS="hybrid" SCALING_NX_LIST="4096 8192" SCALING_MPI_NP=4 SCALING_BLOCK_X=32 SCALING_BLOCK_Y=8 SCALING_THREADS_LIST="1 2 4 8 16" SCALING_STEPS=300
```

OMP only strong sweep:

```bash
make scaling-study SCALING_VARIANTS="omp" SCALING_NX_LIST="1024 2048 4096 8192 16384" SCALING_THREADS_LIST="1 2 4 8 12 16 24 32" SCALING_STEPS=400
```

CUDA only with explicit block shape:

```bash
make scaling-study SCALING_VARIANTS="cuda" SCALING_NX_LIST="4096 8192" SCALING_BLOCK_X=32 SCALING_BLOCK_Y=8 SCALING_STEPS=250
```

## 7) Serial implementation: ready-to-run commands

Build only the serial executable:

```bash
make heat_serial
```

Run serial with defaults compiled from the Makefile (`NX`, `NY`, `NSTEPS`):

```bash
./heat_serial
```

Run serial with explicit runtime parameters:

```bash
./heat_serial --nx 4096 --ny 4096 --steps 500 --snap-every 10 --outdir results/serial_nx4096_s500
```

Quick serial smoke test (small grid, short run):

```bash
./heat_serial --nx 512 --ny 512 --steps 50 --snap-every 25 --outdir results/serial_smoke
```

Serial strong-size sweep via the scaling target:

```bash
make scaling-study \
  SCALING_VARIANTS="serial" \
  SCALING_NX_LIST="1024 2048 4096 8192" \
  SCALING_STEPS=300 \
  SCALING_SNAP_EVERY=10
```

Serial sweep with one command-line override set:

```bash
make scaling-study SCALING_VARIANTS="serial" SCALING_NX_LIST="2048 4096" SCALING_STEPS=200 SCALING_SNAP_EVERY=1
```

Notes for serial runs:

- `heat_serial` parses: `--nx`, `--ny`, `--steps`, `--snap-every`, `--outdir`
- CSV timing line printed by the executable: `serial,nx,ny,1,nsteps,wall_time_s`
- `make scaling-study ... SCALING_VARIANTS="serial"` appends serial rows to `results/scaling.csv`
