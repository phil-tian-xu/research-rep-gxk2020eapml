### 1. Log in from the local PC

```
ssh s1155269217@scrp-login.econ.cuhk.edu.hk
```

### 2. Request computing resources

Run on the login node:

```
compute -c 8 --mem=32G -t 02:00:00 bash
```

This requests **8 CPUs, 32 GB RAM, and 2 hours**. Adjust these values for the workload. Wait for allocation before continuing.

### 3. Enter the repository and set CPU thread limits

```
cd ~/research-rep-gxk2020eapml

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
```

Repeat these settings in each new compute session.

### 4. Run Python

Use the server’s project environment:

```
.venv/bin/python -u -B path/to/script.py
```

Or run a project module:

```
.venv/bin/python -u -B -m core.module_name
```

Example: full-sample PCR test for 1987, fixed K=10:

```
.venv/bin/python -u -B -m core.data_examine.pcr_full_memory \
  --output-dir "$HOME/large-data/GKX2020/results/pcr_full_memory"
```

Keep the SSH connection open while this interactive job runs.

### 5. Release resources

After Python finishes:

```
exit
```

This returns to the login node.

### Guidance for agents

When providing server execution instructions, include the resource request, thread settings, and Python command. Keep the repository name `research-rep-gxk2020eapml`, use `.venv/bin/python`, and select resources appropriate to the workload. Run substantial computation on allocated compute nodes. For unattended jobs, use `sbatch`.

## PCR production batch job

`run_pcr_scrp.sh` submits the existing full-universe PCR workflow. It requests
one node, one Python process, 8 CPUs, 192 GiB RAM and two days in partition
`scrp`, account `pg`, QoS `c32g4`. This workload uses CPUs, so no GPU is requested.
The memory/time requests are initial estimates, not measured requirements for
the entire horizon. Check `qos` and `scrp-info` on SCRP before submission.
Resource options can be overridden before the script name in `sbatch`.
See the [official Slurm guide](https://scrp.econ.cuhk.edu.hk/guide/slurm).

### Files and prerequisites

Update the server repository with `run_pcr_scrp.sh`, `.gitattributes`, and
`config/gkx_pcr_scrp.yaml`. Keep the existing Python source/config dependencies
at the same version throughout a run. The shell script must have LF line endings;
`.gitattributes` preserves these when checking out through Git. Direct SFTP
uploads should preserve the file bytes. No executable bit is needed for `sbatch`.

The script expects:

- Repository: `~/research-rep-gxk2020eapml`.
- Python: `.venv/bin/python`, with the project dependencies including NumPy,
  SciPy and Matplotlib installed. The script checks their imports before fitting.
- Full prepared dataset: `data/processed/gkx_parquet` accessible from that
  repository, through the existing data directory/link.
- GNU `/usr/bin/time` installed on the compute node.

The SCRP config inherits `gkx_pcr_full_v5` unchanged except for `RUN.OUTPUT_DIR`,
set to `/data/users/s1155269217/GKX2020/results/gkx_pcr_full_v5`.
It retains all stocks, PCR only, test years 1987-2016, K=0..100 selected annually
by validation MSE, float64 arrays and the original dense estimator.
The account-specific data root matches Phil's recorded SCRP storage location.

### Submit in three stages

Run these commands in Bash on the SCRP login node. Do not run `compute` first;
`sbatch` requests the compute allocation itself. Once submitted, the job does
not depend on keeping the SSH connection open.

```bash
cd ~/research-rep-gxk2020eapml
sbatch run_pcr_scrp.sh --stop-after-year 1996
```

This starts a fresh SCRP run for 1987-1996. Slurm prints the job ID.
`slurm-JOBID.out` in the submission directory points to the detailed log:

```text
/data/users/s1155269217/GKX2020/results/slurm/pcr-JOBID/run.log
```

Find `Run directory:` in that log. After the stage finishes, copy its exact
timestamped directory into `runDir`; do not use the parent results directory.

```bash
runDir='/data/users/s1155269217/GKX2020/results/gkx_pcr_full_v5/REPLACE_WITH_TIMESTAMP'
sbatch run_pcr_scrp.sh --resume "$runDir" --stop-after-year 2006
```

After that job finishes, continue through 2016:

```bash
sbatch run_pcr_scrp.sh --resume "$runDir" --stop-after-year 2016
```

Submit stages sequentially. Reassign `runDir` if opening a new terminal.
Completed annual checkpoints are skipped. If a job fails or reaches its time
limit, use the same resume command; an unfinished annual fit starts again.
Do not edit the saved configuration or bypass compatibility checks. Start with
a fresh SCRP run rather than assuming Windows checkpoints are compatible.

Alternatively, submit all 30 years in one job:

```bash
sbatch run_pcr_scrp.sh
```

This creates a new run; use `--resume "$runDir"` to continue an existing one.

### Memory, logs and replication results

Each job saves `run.log`, `packages.txt`, `submitted_script.sh` and `time.txt`
under its `results/slurm/pcr-JOBID/` directory. `time.txt` is written by GNU time
when the process exits and includes `Maximum resident set size (kbytes)`;
divide this Linux value by 1,048,576 to express it in GiB. It measures that
invocation's process peak, not an annual breakdown. A forced job termination
may prevent the final timing report; use available Slurm accounting in that case.
Python failures propagate as a failed job rather than a success message.

This script uses GNU time instead of `--memory`, whose current counters support
Windows only. The estimation code is unchanged.

At the end of each successful stage, the timestamped run's `table1_figure3/`
directory contains:

- `table1_all.csv`: pooled out-of-sample R2 and R2 percent for all stocks.
- `annual_metrics.csv`: annual losses, R2 and sample counts.
- `complexity.csv`: annual selected/fitted component counts and grid-boundary flags.
- `figure3_complexity.png` and `.svg`: PCR component counts over time.
- `tuning_curves.csv` and `validation_YEAR.png`: validation losses across K.

Partial periods are flagged as incomplete. The Table 1 report covers the All
row; large/small-stock rows are outside this workflow. The run root also retains
annual checkpoints, combined predictions, configuration and data/source hashes.
