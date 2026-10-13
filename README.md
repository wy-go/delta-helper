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
| `ds-crowd` | How crowded each partition is right now (pending, running, idle and partly used nodes) and its billing weight. Works on Delta and DeltaAI. |
| `submit-everywhere` | Submit one unit of work to every lane that can run it (Delta partitions, campus cluster and DeltaAI through ssh) as one duplicate group; records every job id in a group file. |
| `dedup-reaper` | One loop (every 2 min, single instance) that cancels a group's extra copies once one runs, and all others once the unit is finished. |
| `pool-run` | Runs independent units in parallel on one machine (login node, job allocation, lent machine) with workers capped at free cores, library threads pinned, per-unit claims, a pool lock and a streamed results log. Also on ICC and CloudLab (plain Linux, Python 3.7+). |
| `load-guard` | One-line alerts for a watchdog: load above cores, too many copies of one script, heavy processes on login nodes, a pool lock held twice, processes stuck in I/O wait on `/scratch`. |

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

### Never wait on one queue: `submit-everywhere` and `dedup-reaper`

```bash
# one unit, every CPU lane on Delta and the campus cluster; the unit is finished when its output exists
submit-everywhere --unit pal_slot3 --lanes delta-cpu --lanes icc-cpu --done-file /projects/bgwo/wanyu/x/slot3.done \
    --expected-minutes 40 job.slurm slot3
# every A100 lane, plus A40 as a second hardware type
submit-everywhere --unit probe --lanes a100 --lanes a40 --hardware a100,a40 job.slurm
# start the reaper once per site (state and lock on shared storage); check it
setsid nohup dedup-reaper > /projects/bgwo/wanyu/coord_reaper/reaper_stdout.log 2>&1 < /dev/null &
dedup-reaper --status
```

- Lanes: `--lane PARTITION:ACCOUNT[:CLUSTER[:HARDWARE[:normal|preempt]]]` or presets (`delta-cpu`, `icc-cpu`, `a100`, `a40`,
  `h200`, `mi100`, `gh200`); `--hardware` keeps only the listed hardware types. Clusters `icc` and `deltaai` are reached by ssh
  from a Delta login node (the campus connection lives on the node the owner logged in from); the script is copied to
  `~/submit_everywhere/<group>/` there unless `--remote-script CLUSTER=PATH` names a copy, and its paths must exist there.
- Every copy is named after the group id and carries `--comment=submit-everywhere:<group>`. The script must set a time limit;
  the two costly interactive partitions are refused; a lane whose sbatch fails (for example the one-job-per-user interactive
  limit) is reported and skipped.
- The reaper (`/projects/bgwo/wanyu/coord_reaper/`: `groups/`, `reaper.lock` with host, pid and heartbeat, `reaper.log`) cancels
  only job ids recorded in a group file, after checking the job's name is still the group id. Rule (site rule in
  /u/wanyu/CLAUDE.md): once a normal copy runs, the other copies go; a preempt copy that had run 5 min or less when the normal
  copy started yields; one that ran longer keeps the normal copy as insurance if 30 min or more remain, else the normal copy
  goes; while only preempt copies run, pending normal copies stay. When the unit is finished (done file, or a copy that
  completed with exit 0 if no done file is given) all other copies are cancelled. Give a done file whenever a copy can exit 0
  without doing the work (for example when another copy holds the unit's claim).
- Tests: `python3 tests/test_dedup_decide.py` (the rule); the end-to-end test of 2026-10-11 is in the commit message.

### Correct parallelism by default: `pool-run` and `load-guard`

Use `pool-run` for every batch of independent units (fits, simulations, evaluations), instead of a sequential loop or a
hand-written `xargs`/`&` launcher; it applies the playbook's "Throughput check before every batch" rule by default.
Run `load-guard` from every watchdog loop and before placing work on a shared or lent machine.

```bash
# units.txt: one unit per line, "ID<TAB>COMMAND"; a unit writes {tmp}, renamed to {out} when it exits 0
printf 'fit_a\tpython fit.py --cfg a --out {tmp}\nfit_b\tpython fit.py --cfg b --out {tmp}\n' > units.txt
pool-run --pool /projects/bgwo/wanyu/myexp/pool --units units.txt --out '/projects/bgwo/wanyu/myexp/out/{id}.json' --time-one
pool-run --pool /projects/bgwo/wanyu/myexp/pool --units units.txt --out '/projects/bgwo/wanyu/myexp/out/{id}.json'
pool-run --pool /projects/bgwo/wanyu/myexp/pool --template 'python fit.py --cfg {item} --out {tmp}' --items cfgs.txt --out 'out/{id}.json'
pool-run --pool /projects/bgwo/wanyu/myexp/pool --units units.txt --status
load-guard --hosts 'local,dt-login03,cloudlab-*' --alerts-only     # exit 1 and one line per alert
```

`pool-run`, what it does by default (`pool-run -h` has every option):

- **Workers** = min(`--workers`, free cores // `--threads`), free cores = cores minus load, rounded down (after 0.25 core of measurement noise). Cores are this
  process's CPUs (inside a Slurm job only the allocation); load is max(1-min load average, CPUs busy now) on a whole
  machine and the CPUs busy now inside an allocation. It never runs more workers than fit, and refuses to start when none
  fits (exit 3), unless `--force`. Login nodes (`dt-login*`, `gh-login*`, `cc-login*`) cap at `--login-max` 4.
  `--mem-per-worker GB` also caps by available memory.
- **Threads pinned**: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, BLIS, vecLib,
  Numba, `LOKY_MAX_CPU_COUNT` (joblib), Rayon, Polars and TensorFlow are set to `--threads` (default 1); LightGBM and
  XGBoost follow `OMP_NUM_THREADS`. Code that passes `n_jobs`/`num_threads` itself should read `$POOL_RUN_THREADS`.
  `--bind` also pins each worker to its own CPUs.
- **Claims**: `<pool>/claims/<id>.json` (host, launcher pid, unit process group, lane), created atomically. On its own
  host a claim is stale once its launcher is gone (unit processes left behind are killed at takeover); from another host,
  once its heartbeat (mtime, every `--heartbeat` 30 s) is older than `--stale-seconds` 180. Copies of one pool on several
  hosts or jobs (a shared `--pool` on `/projects`) split the units and never run a live unit twice; `--linger` keeps a
  launcher waiting to take over stale claims instead of exiting (exit 6) when only other launchers' units remain.
- **Outputs** temp-then-rename (`{tmp}` fsynced and renamed to `{out}` on exit 0); finished units are skipped, so the same
  command resumes. `<pool>/done/<id>.json` marks units without `--out`. A failed unit is retried `--retries` (1) times.
- **One launcher per pool and lane**: `<pool>/locks/<lane>.lock` (host, pid, heartbeat); the lane is the host, plus the
  Slurm job id inside a job. A second launcher exits (exit 4), and so does another lane on the same host outside Slurm.
- **Results stream**: one JSON line per finished unit (unit, status, rc, seconds, host, lane, attempt, out, log) is
  appended to `<pool>/results.jsonl` and printed as `RESULT {...}`; per-unit logs in `<pool>/logs/`.
- **`--time-one`** runs one unit (kept as a real result) and prints its time, units left, workers and the projected wall;
  units measured alone are a lower bound when many share memory bandwidth.
- **Nice 19** outside Slurm jobs (login nodes, lent machines). **dt-login02**: refuses (exit 7) when the pool, units,
  outputs, results or working directory are on `/scratch` (= `/work/hdd`), whose I/O there hung for 4-7 min on
  2026-10-13; run on dt-login03 or in a job (`--allow-scratch-here` overrides; `POOL_RUN_SCRATCH_HANG_HOSTS` sets the
  host list).
- Inside a job, give it the whole allocation: `#SBATCH --cpus-per-task=N` and `pool-run ...` with no `--workers`.

`load-guard` checks each host (`local`, ssh aliases, patterns over `~/.ssh/config`) and prints `ALERT <host> <check>: ...`
lines (exit 1) or `OK <host>: ...`:

| check | alerts when |
|---|---|
| `load` | the 5-min load average (`--load-window`) exceeds the core count |
| `copies` | more than `--max-copies` (default: core count) processes of one script run for the user |
| `login-cpu` | on a login node, one process used more than `--login-cpu-minutes` (10) CPU-minutes at 20% or more of a core on average (`--min-util`); editors and agents skipped (`--ignore`) |
| `pool-lock` | two `pool-run` launchers for one pool and lane on a host, two locks for one pool on a host outside Slurm, or a launcher without its lock; stale locks are `NOTE` lines |
| `scratch-io` | a process in uninterruptible I/O wait (state D) on `/scratch` for more than `--dstate-minutes` (2) without using CPU; on dt-login02 (`--dstate-any-hosts`) every such stuck process is reported. Durations come from successive runs (state in `~/.cache/load-guard/dstate.json`), so run it every 1-5 min |
| `unreachable` | ssh failed |

There is no separate watchdog script in this repo; add `load-guard` to the session watchdog loop (or a Monitor):

```bash
load-guard --hosts 'local,dt-login03,cloudlab-*' --watch 120 --alerts-only   # prints new alerts with a time stamp; repeats every 30 min
# or one pass inside an existing loop:  out=$(load-guard --hosts local,dt-login03 --alerts-only) || echo "$out"
```

As a library: `sys.path.insert(0, "/u/wanyu/github/delta-helper/bin"); from load_guard import check_hosts`
(`check_hosts(["local"])` returns alert objects with `.line()`). Tests: `python3 tests/test_load_guard.py` (synthetic
snapshots) and `python3 tests/test_pool_run.py` (end to end on the current machine; see the commit message for the
2026-10-13 runs). `illinois-helper/bin` holds identical copies of both tools for ICC.

## Which partition starts soonest (and what it costs)

Crowding changes by the hour, so check it live before choosing:

```bash
ds-crowd --gpu-only        # sorted by pending jobs per running job, least crowded first
```

Snapshot 2026-09-28 about 03:00 CDT, with the measured wait of a 1-GPU job (16 cores, 55-90 min wall) submitted to every lane at once.
Billing is per GPU; the regular queue is 1000 = 1x. "Pending per running" above ~5 means expect a long wait.

| cluster | partition | pending | running | pending per running | billing per GPU | measured start of a 1-GPU job |
|---|---|---|---|---|---|---|
| DeltaAI | `ghx4` | 1,098 | 286 | 3.8 | 1000 (1x) | **44 s** |
| DeltaAI | `ghx4-interactive` | 18 | 15 | 1.2 | 2000 (2x) | 0 s, **but do not use: 2x price** |
| Delta | `gpuA100x4` | 2,243 | 173 | 13.0 | 1000 (1x) | **31 min** (small short jobs backfill) |
| Delta | `gpuA100x4-interactive` | 7 | 10 | 0.7 | 2000 (2x) | refused: one job per user per partition, slot held |
| Delta | `gpuA100x8` | 370 | 34 | 10.9 | 1500 (1.5x) | not started after 31 min |
| Delta | `gpuA40x4` | 776 | 311 | 2.5 | 500 (0.5x) | not started after 31 min |
| Delta | `gpuH200x8` | 480 | 12 | 40.0 | 3000 (3x) | not started after 31 min |
| Delta | `gpuH200x8-interactive` | 11 | 1 | 11.0 | 6000 (6x) | not tried (never use) |
| Delta | `gpuA100x4-preempt`, `gpuA40x4-preempt` | ~100 each | 2 each | 50+ | 500 / 250 (0.5x / 0.25x) | not started after 31 min |

What this means in practice:

- **DeltaAI `ghx4` was the fastest GPU start by far**, at the regular price. It is ARM (aarch64): x86 Python environments do
  not run there, so build a separate one on a `gh-login` node. It reads Delta's `/work/hdd` (= Delta `/scratch`) and `/projects`,
  so data needs no copying; the home directory is separate.
- **Never use `ghx4-interactive`**: it costs 2x and `ghx4` starts almost as fast.
- **Delta interactive lanes cost 2x** and allow one job per user per partition; if another of your jobs holds the slot, new
  submissions are refused (`QOSMaxSubmitJobPerUserLimit`).
- **Delta preempt lanes are cheap but rarely start**: about 2 running against 100+ pending.
- **Short, small jobs start sooner everywhere**: a tight `--time` and one GPU let Slurm backfill the job into gaps.
- **H200 on Delta is the most crowded and the most expensive** (3x, 6x interactive).
- Campus Cluster (ICC) availability is covered in `illinois-helper`. On 2026-09-28 all its A100/H200 GPUs were busy (a 1-GPU job
  had not started after 33 min), while a 2 x A10 job on `eng-research-gpu` started in 44 s.

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
