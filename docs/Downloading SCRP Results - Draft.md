Check Job Status:
```bash
squeue -u "$USER" -r
```

Check Experiment Status:
```bash
.venv/bin/python -B run_experiment.py list --refresh
```

Collect results:
```bash
.venv/bin/python -B run_experiment.py collect EXPERIMENT_ID
```

# Downloading completed SCRP results to 
your own PC

**Status: draft for Phil to review.**

This guide covers experiments created through `run_experiment.py`: find the experiment, confirm completion, collect results on SCRP, check the local inventory, download missing experiments, and confirm they appear locally. Detailed content validation, result analysis, and paper comparisons belong to a separate workflow.

Run server commands in **Bash from the original project checkout on SCRP**, and local commands in **PowerShell from your local repository root**. The examples use the ENet+H and OLS-3+H experiments identified during our walkthrough; replace their IDs for future runs.

## 1. On SCRP: find experiments and check their status

```bash
.venv/bin/python -B run_experiment.py list --refresh
```

Start here even if you forgot the experiment or job ID.

| Column | Meaning |
| --- | --- |
| `experiment` | Experiment ID and normally its folder name |
| `model`, `sample`, `years` | Research specification and requested period |
| `saved` | Annual checkpoint files present; `30/30` for a complete 1987–2016 run |
| `reports` | `ready` means the combined report marker exists; `pending` means it does not |
| `last_action` | Last recorded launch action; may remain `submitted` after completion |
| `job_ids` | Slurm submission IDs, including retries |
| `scheduler_now` | Scheduler/accounting status for those submissions |

The **experiment ID stays the same across retries**; each retry receives a new **Slurm job ID**. The listing is an inventory, not a detailed validation of saved results.

If an experiment is missing, explicitly select the results root:

```bash
.venv/bin/python -B run_experiment.py list --refresh --root /data/users/s1155269217/GKX2020/results/experiments
```

Only experiments with unified-launcher records are automatically listed. Older manually launched runs require their original paths and commands. Do not repeat a new-experiment command just to check or continue an existing run.

## 2. On SCRP: confirm completion or investigate unfinished tasks

For a selected job, replace the ID and use a start date before submission:

```bash
sacct -X -j 398186 --starttime 2026-09-26 --format=JobID%22,State%24,ExitCode,Elapsed
```

| Status | Action |
| --- | --- |
| `PENDING`, `RUNNING`, `COMPLETING` | Wait |
| `COMPLETED` with `ExitCode=0:0` | Execution succeeded; proceed to collection when all requested years are saved |
| `FAILED`, `OUT_OF_MEMORY`, `TIMEOUT`, `CANCELLED` | Inspect the affected task's logs |
| No accounting information | Completion is unknown |

A `0:0` exit code beside a running or pending task is not a final success indication. An empty queue also does not establish success. For a 1987–2016 experiment, expect 30 completed annual results. If there were retries, inspect their job IDs too; earlier failed attempts can remain in accounting after their replacement tasks succeed.

To inspect active tasks:

```bash
squeue -r -j 398186 -o "%.20i %.12T %.10M %R"
```

If a task failed, set its experiment path, submission ID, and task index. Task index 0 corresponds to the first requested year:

```bash
experiment="/data/users/s1155269217/GKX2020/results/experiments/enet_full_1987-2016_20260926T080544288119Z"
job=398186
task=0
```

Read the worker log:

```bash
tail -n 80 "$experiment"/annual/logs/"${job}_${task}_"*/run.log
```

If no worker log exists, inspect scheduler output:

```bash
tail -n 80 "$experiment"/submissions/*/"slurm-${job}_${task}.out"
```

After addressing the cause, and once no tasks for that experiment remain queued or running:

```bash
.venv/bin/python -B run_experiment.py resume "$experiment"
```

Resume retains validated completed checkpoints and retries unfinished years. Keep the original execution code and environment for retries. Re-enter the shell variables after opening a new terminal, and use the retry's new job ID when inspecting its logs.

## 3. On SCRP: collect completed results

Run once per completed experiment, replacing `EXPERIMENT_ID` with the ID from the listing:

```bash
.venv/bin/python -B run_experiment.py collect EXPERIMENT_ID
```

For our ENet+H and OLS-3+H examples, run these one at a time:

```bash
.venv/bin/python -B run_experiment.py collect enet_full_1987-2016_20260926T080544288119Z
```

```bash
.venv/bin/python -B run_experiment.py collect ols3_full_1987-2016_20260926T140325486437Z
```

Collection validates the saved annual results and generates combined tables and plots under `annual/combined/`. It does not retrain models. Collect at the original server location before downloading.

Check the updated inventory:

```bash
.venv/bin/python -B run_experiment.py list --refresh
```

The completed experiments should now show the expected saved count and `reports=ready`. If collection says **“Reports already saved,”** it returns the existing location without repeating validation. If collection fails, inspect the error and resolve it before downloading the experiment as a completed result.

## 3a. On your own PC: check what is already downloaded

Before downloading, run:

```powershell
.venv/Scripts/python.exe -B run_experiment.py list --root results/scrp_imports
```

Compare **exact experiment IDs** with the server listing. This avoids confusing a pilot or earlier experiment with the desired full run. Use this command without `--refresh`; it reads local records.

For our example, the local inventory before downloading was:

| Experiment | Local inventory | Decision |
| --- | --- | --- |
| PCR, full 1987–2016 | Matching ID present, `30/30`, `ready` | Skip |
| PLS, full 1987–2016 | Matching ID present, `30/30`, `ready` | Skip |
| ENet+H, full 1987–2016 | Missing | Download |
| OLS-3+H, full 1987–2016 | Missing | Download |

Local pilot runs are stored separately. To inspect those when useful:

```powershell
.venv/Scripts/python.exe -B run_experiment.py list --root results/experiments
```

## 4. On your own PC: download the missing experiments

Connect to CUHK VPN if off campus. Run the following commands from the local repository root.

The destination is `results/scrp_imports`. If setting up this workflow on a new PC, create the directory first:

```powershell
New-Item -ItemType Directory -Force -Path 'results/scrp_imports' | Out-Null
```

For our example, download both complete experiment folders with this **single PowerShell line**:

```powershell
@('lcd results/scrp_imports', 'cd /data/users/s1155269217/GKX2020/results/experiments', 'get -R enet_full_1987-2016_20260926T080544288119Z', 'get -R ols3_full_1987-2016_20260926T140325486437Z', 'bye') | sftp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 s1155269217@scrp-login.econ.cuhk.edu.hk
```

Enter your password when prompted. Wait for both transfers to finish and inspect any transfer errors. An interrupted transfer is not complete.

For future experiments, replace the folder names after `get -R`. Download the whole experiment folder so its reports, checkpoints, configurations, and logs remain together. This copies files; it does not provide incremental synchronization.

**Backup plan:** if direct downloading repeatedly disconnects, use the campus PC as an intermediary, then transfer the downloaded folders to your own PC.

## 5. On your own PC: verify the download through the local inventory

You do **not** need to retrain or recollect to read the downloaded reports.

```powershell
.venv/Scripts/python.exe -B run_experiment.py list --root results/scrp_imports
```

Use this without `--refresh`: local inspection reads saved records; it does not connect to SCRP.

For our example, expect **PCR, PLS, ENet+H, and OLS-3+H**, each showing:

- The intended experiment ID, full sample, and 1987–2016 period.
- `saved=30/30`.
- `reports=ready`.

This is the inventory-level download check. Detailed content validation and research analysis will be handled in the separate results-review workflow.

## Review notes

During the walkthrough, Phil confirmed server listings with ENet+H and OLS-3+H changing from `pending` to `ready`, and a local listing containing the existing PCR and PLS imports. Successful transfer and the post-download listing for ENet+H and OLS-3+H have not yet been confirmed in this conversation.

The server listing displayed the same PCR experiment ID twice. The cause remains unresolved; this does not establish that training ran twice.

## References

- [Running Experiments](Running%20Experiments.md): existing launcher and reporting guide.
- [Annual Parallel Runs](working_notes/Annual%20Parallel%20Runs.md): underlying annual collection workflow.
- [SCRP account access and file transfer](https://scrp.econ.cuhk.edu.hk/guide/account-and-access).
- [Slurm accounting documentation](https://slurm.schedmd.com/sacct.html).
- [SFTP command documentation](https://man.openbsd.org/sftp).
