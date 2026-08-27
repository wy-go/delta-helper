# delta-helper

Slurm sbatch templates and small helpers for NCSA [**Delta**](https://docs.ncsa.illinois.edu/systems/delta/en/latest/index.html) and [**DeltaAI**](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/index.html).

Kept intentionally small: four template scripts you copy into a project and
edit, plus three CLI helpers for interactive shells / status / cancel.

## Install

```bash
git clone <this-repo> ~/github/delta-helper
export PATH="$HOME/github/delta-helper/bin:$PATH"   # add to ~/.bashrc
```

Create a config file with your allocation codes (kept outside the repo, so it
never gets committed):

```bash
mkdir -p ~/.config/delta-helper
cp config.example ~/.config/delta-helper/config
chmod 600 ~/.config/delta-helper/config
$EDITOR ~/.config/delta-helper/config
```

Keys: `DELTA_ACCOUNT_CPU`, `DELTA_ACCOUNT_GPU`, `DELTAAI_ACCOUNT`. Run
`accounts` on a login node to see which codes you have on that cluster.

Resolution order (first non-empty wins):
`--account` flag → env var → `~/.config/delta-helper/config`.
The legacy `DELTA_ACCOUNT` / `DELTAAI_ACCOUNT` env vars are still honored as a
fallback if you'd rather set one code inline.

## Templates

Copy the one you want into your project, edit the header, and `sbatch it`.

| Template | Cluster | Purpose |
|---|---|---|
| [templates/delta_cpu.slurm](templates/delta_cpu.slurm) | Delta | CPU job on the `cpu` partition |
| [templates/delta_gpu.slurm](templates/delta_gpu.slurm) | Delta | Single-node GPU job (A40 / A100 / H200 / MI100 — pick a partition) |
| [templates/delta_gpu_multinode.slurm](templates/delta_gpu_multinode.slurm) | Delta | Multi-node PyTorch DDP (`torchrun`) |
| [templates/deltaai_gpu.slurm](templates/deltaai_gpu.slurm) | DeltaAI | GH200 GPU job on the `ghx4` partition |

Every template has the `--account`, `--partition`, `--time`, and resource lines
grouped at the top so they're easy to find. Comments explain each knob.

## Helpers (in `bin/`)

| Command | What it does |
|---|---|
| `ds-interactive` | `salloc` an interactive shell. Auto-detects cluster from hostname. |
| `ds-status` | `squeue -u $USER` with a wider, more useful format. |
| `ds-cancel` | Cancel by job id, by name pattern, or (with confirm) all of yours. |

Examples:

```bash
ds-interactive                      # 1h shell on this cluster's default GPU partition
ds-interactive --cpus 32 --mem 64g  # bigger interactive shell
ds-interactive --gpu-type a100 --time 2:00:00
ds-interactive --cluster dai --time 1:00:00

ds-status                # your queue
ds-status --watch        # loop every 10s

ds-cancel 1234567        # one job
ds-cancel --name my_run  # every job whose name matches
ds-cancel --all          # every job of yours (prompts for confirmation)
```

Run any helper with `-h` for full options.

## Cluster cheat sheet

|  | Delta | DeltaAI |
|---|---|---|
| Login host prefix | `dt-login*` | `gh-login*` |
| CPU arch | AMD EPYC 7763 (x86_64) | NVIDIA Grace ARM (aarch64) |
| GPU | A40 / A100 40GB / A100 80GB / H200 / MI100 | GH200 (H100 96GB, unified mem) |
| Main partitions | `cpu`, `gpuA40x4`, `gpuA100x4`, `gpuA100x8`, `gpuH200x8`, `gpuMI100x8` (+ `-interactive`, `-preempt`) | `ghx4`, `ghx4-interactive` |
| Node share unit | fully shared; request only what you need | 1 GH200 ≈ 72 cores / 110 GB / 1 GPU |
| Account format | `bbXX-delta-{cpu,gpu}` (partition-scoped) | `bbXX` (one code) |
| Max batch time | 48 hr | 48 hr |
| Max interactive time | 1 hr | 2 hr (8 node-hr/user rolling budget) |
| Default mem-per-core | 1000 MB | 1000 MB |
| Default wall-clock | 30 min | 30 min |
| Filesystem constraint | `--constraint="scratch"` etc. (see below) | not required |
| PyTorch module | `pytorch-conda` | `python/miniforge3_pytorch` |
| NCCL over Slingshot 11 | load `aws-ofi-nccl`; `NCCL_SOCKET_IFNAME=hsn` | loaded automatically; `NCCL_SOCKET_IFNAME=hsn` |
| Preempt queues | yes (`*-preempt`, 0.25–0.5× charge) | not yet available |

### Storage

| Path | Use for | Notes |
|---|---|---|
| `/u/$USER` (HOME) | code, configs, small files | **not** for job I/O; 30-day snapshots |
| `/scratch/<acct>/<user>/` | job I/O, experiment outputs | same Lustre volume as `/work/hdd`; not purged |
| `/projects/<acct>/` | shared long-lived datasets | not purged |
| `/work/nvme/…` | many-small-file I/O | available on request |
| `/tmp` (per node) | in-job scratch | wiped after each job |

`/scratch` and `/work/hdd` are the **same underlying volume** under two names.
Recommended layout for a project is
`/scratch/<account>/<user>/<project>/`. See NCSA
[Data Management docs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html)
for exact quotas and how to request larger allocations.

### Delta filesystem constraints

Add to any batch script that touches `/scratch`, `/projects`, `/work/hdd`,
`/taiga`, or `/ime`. Only your `/u/$USER` (HOME) is assumed by default.

```
#SBATCH --constraint="scratch"          # single
#SBATCH --constraint="scratch&projects" # multiple
```

### Common `sbatch` knobs

```
--account=$DELTA_ACCOUNT       # required
--partition=gpuA40x4           # see table
--time=04:00:00                # hh:mm:ss, capped by partition max
--nodes=1
--ntasks-per-node=1            # 1 for PyTorch single-process; N for MPI
--cpus-per-task=16             # OMP_NUM_THREADS ≈ this
--mem=64g                      # or `--mem=0` for whole-node exclusive
--gpus-per-node=1              # GPU count (Delta + DeltaAI)
--gpu-bind=closest             # PCIe-topology-aware CPU-GPU pinning
--exclusive                    # whole node (charged as such)
--no-requeue                   # opt out of automatic requeue
```

### Docs

- Delta: https://docs.ncsa.illinois.edu/systems/delta/en/latest/index.html
- DeltaAI: https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/index.html
