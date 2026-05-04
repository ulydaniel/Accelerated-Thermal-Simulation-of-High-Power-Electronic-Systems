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

## 8) visualize.py command reference (all plots and outputs)

This section collects the full command set for `src/visualize.py`.

### 8.1) Move to the right directory

From project root:

```bash
cd src
```

All commands below assume you are inside `src`.

### 8.2) Scaling plot (group versions together)

Main scaling figure from CSV:

```bash
python visualize.py --mode scaling --csv results/scaling.csv
```

What this plot shows:

- Top panel: runtime vs grid size (`nx`) for each version/thread series
- Bottom panel: speedup vs serial baseline

Print normalized rows only (no figure window):

```bash
python visualize.py --mode scaling --csv results/scaling.csv --no-scaling-plot
```

Save scaling figure to PNG and PDF:

```bash
python visualize.py --mode scaling --csv results/scaling.csv --scaling-save-prefix results/scaling_comparison
```

Open sortable scaling metrics table UI (and still show plot unless disabled):

```bash
python visualize.py --mode scaling --csv results/scaling.csv --show-scaling-table
```

Table only (no scaling plot):

```bash
python visualize.py --mode scaling --csv results/scaling.csv --show-scaling-table --no-scaling-plot
```

### 8.3) 2D heatmap plot (snapshot slider)

Interactive heatmap over saved snapshots:

```bash
python visualize.py --mode heatmap --outdir results --cmap inferno
```

Use another colormap:

```bash
python visualize.py --mode heatmap --outdir results --cmap turbo
```

Prompt for output directory at runtime:

```bash
python visualize.py --mode heatmap --ask-outdir
```

### 8.4) 3D surface plot (MATLAB-like)

Interactive 3D surface with snapshot slider:

```bash
python visualize.py --mode surface --outdir results --cmap turbo --elev 35 --azim 45 --surface-stride 8
```

Save 3D surface view to PNG and PDF:

```bash
python visualize.py --mode surface --outdir results --cmap turbo --elev 35 --azim 45 --surface-stride 8 --save-prefix results/thermal_surface
```

Higher-fidelity surface (less downsampling):

```bash
python visualize.py --mode surface --outdir results --surface-stride 2
```

Faster rendering for very large grids:

```bash
python visualize.py --mode surface --outdir results --surface-stride 16
```

### 8.5) Export field snapshots to plain text files (for MATLAB)

Export latest snapshot as CSV matrix:

```bash
python visualize.py --mode export --outdir results --step -1 --export-format csv --export-path results/field_latest.csv
```

Export a specific step as DAT matrix:

```bash
python visualize.py --mode export --outdir results --step 200 --export-format dat --export-path results/field_step_000200.dat
```

Export as OUT matrix:

```bash
python visualize.py --mode export --outdir results --step 200 --export-format out --export-path results/field_step_000200.out
```

### 8.6) Equivalent commands from project root (without `cd src`)

Scaling:

```bash
python src/visualize.py --mode scaling --csv src/results/scaling.csv
```

Heatmap:

```bash
python src/visualize.py --mode heatmap --outdir src/results --cmap inferno
```

Surface:

```bash
python src/visualize.py --mode surface --outdir src/results --cmap turbo --surface-stride 8
```

Export:

```bash
python src/visualize.py --mode export --outdir src/results --step -1 --export-format dat --export-path src/results/field_latest.dat
```

### 8.7) Python package note

`visualize.py` requires:

- `numpy`
- `matplotlib`

If missing in your environment:

```bash
python -m pip install numpy matplotlib
```
