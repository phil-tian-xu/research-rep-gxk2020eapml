# Research experiment workflow

This guide follows an experiment from local development to server execution and
back to local analysis. The workflow is reusable; the commands use this project's
launcher, accounts and datasets as examples. Last reviewed: 2026-10-29.

## Workflow at a glance

**Prepare locally → test locally → synchronize code and data → test on the
server → run and monitor full experiments → collect and verify → transfer back
→ analyze locally.**

| Step | Work | Where | Ready to proceed when |
| --- | --- | --- | --- |
| [1. Prepare data](#1-prepare-data-on-the-main-pc) | Preserve inputs, prepare the panel and fixed development samples | Main PC | Data versions and sample definitions are clear |
| [2. Develop and test](#2-develop-and-test-locally) | Modify the method and check small-sample results | Main PC | Training, predictions and diagnostics have been checked |
| [3. Synchronize code](#3-record-and-synchronize-the-tested-code) | Commit/push tested changes and pull on the servers | Main PC and servers | Each experiment has a known code version |
| [4. Transfer inputs](#4-transfer-input-data-through-the-campus-pc) | Main PC → Netdisk → campus PC → server | Both PCs and servers | Required datasets are complete on the target server |
| [5. Test on the server](#5-run-a-small-server-test) | Run a small sample with the intended server setup | SCRP / AI Hub | The pilot finishes and its diagnostics are reviewed |
| [6. Run the full experiment](#6-launch-the-full-experiments) | Submit the planned methods, sample and test years | SCRP / AI Hub | Experiment IDs and intended coverage are recorded |
| [7. Monitor execution](#7-monitor-progress-and-handle-interruptions) | Check queues, saved progress, logs and failures | SCRP / AI Hub | All planned annual fits are saved |
| [8. Collect and verify](#8-collect-and-verify-on-the-server) | Build pooled reports and complete result bundles | SCRP / AI Hub | Coverage and saved files pass verification |
| [9. Bring results home](#9-transfer-results-back-and-verify-locally) | Server → campus PC → Netdisk → main PC | Servers and both PCs | Archives and extracted results pass local verification |
| [10. Analyze and iterate](#10-analyze-document-and-start-the-next-round) | Compare methods, investigate differences and document findings | Main PC | Reviewed outputs and the next research questions are recorded |

Monitoring continues throughout server execution. Input transfers are needed when
deploying a dataset for the first time or introducing a new data version.

| Location/service | Role |
| --- | --- |
| Main PC | Data preparation, model development, local tests and research analysis |
| Campus PC | Transfer bridge; use `D:\data_transfer` for staging |
| SCRP / AI Hub | Server pilots, full experiments and storage of completed runs |
| Git | Synchronize code and experiment settings |
| Netdisk | Exchange datasets and result archives between the two PCs |

## Using the commands

- **PowerShell** blocks run on the specified Windows PC.
- **Bash** blocks run after logging into the specified server.
- Each project uses its own `.venv`. Replace values beginning with `PASTE_` or
  `PATH/TO/` before running a block.
- Re-enter the project and reset the relevant variables after opening a new
  terminal or reconnecting. Initial server checkout/data-path setup is assumed;
  use the linked site guides when setting up a server for the first time.
- An **experiment ID** is used by `resume`, `status` and `collect`. A **Slurm job
  ID** is used by `squeue`, `sacct` and `scancel`.
- Examples that name a method create a new experiment. Use its existing ID for
  subsequent monitoring, resumption and collection.

For detailed site commands, see [SCRP](run_scrp.md), [AI Hub](run_aihub.md) and the
[project run guide](../Running%20Experiments.md).

## 1. Prepare data on the main PC

Preserve the raw inputs. Keep raw data, prepared panels and experiment outputs
separate. Use the same prepared data and fixed development samples locally and
on the servers, with consistent dates, features and target definitions.

**Main PC — PowerShell:**

```powershell
$projectDir = "E:\Workstation\coding_projects\coding_research\paper_replication\research-rep-gxk2020eapml"
Set-Location -LiteralPath $projectDir

Get-ChildItem -LiteralPath .\data\raw
Get-Item .\data\processed\gkx_parquet\metadata.json
Get-Item .\data\processed\gkx_dev_p05_seed2020_parquet\metadata.json
```

This project's full panel is `gkx_parquet`; its fixed 5% development panel is
`gkx_dev_p05_seed2020_parquet`. Reuse existing verified panels. When creating new
data, follow [panel preparation](../working_notes/GKX%20Panel%20Preparation.md),
[development samples](../working_notes/GKX%20Development%20Samples.md) and
[Parquet conversion](../working_notes/Parquet%20Conversion.md).

## 2. Develop and test locally

Implement or revise the research method on the main PC. Start with a small
sample and a limited test period; expand the check when the change warrants it.
Review training behavior, validation selection, predictions and evaluation
results. Record the research choices and unresolved questions.

**Main PC — PowerShell, from the project folder:**

```powershell
# Install/update this project's locked environment when needed; NN extra is optional.
uv sync --locked --extra nn

# Example: one NN1 test year on the fixed 5% sample, using the local GPU.
.\.venv\Scripts\python.exe -B run_experiment.py nn1 --local --sample p05 --years 1987 --cpus 4

.\.venv\Scripts\python.exe -B run_experiment.py list
```

Copy the experiment ID printed by the run:

```powershell
$run = "PASTE_LOCAL_EXPERIMENT_ID"
.\.venv\Scripts\python.exe -B run_experiment.py status $run
.\.venv\Scripts\python.exe -B run_experiment.py collect $run
```

The launcher supports `pcr`, `pls`, `enet`, `ols3`, `glm`, `rf` and `nn1` through
`nn5`. Select the method and configuration appropriate to the research question.
A completed pilot establishes execution and diagnostic evidence; predictive
performance and agreement with the paper still require research review.

## 3. Record and synchronize the tested code

Commit the tested implementation and settings, then pull that version on each
server used for the next experiment. Preserve the recorded code/environment for
existing experiments that may need resumption. Start a new experiment when the
research specification changes.

**Main PC — PowerShell:**

```powershell
git status --short
git diff

# Replace with the specific files belonging to this change.
git add -- "PATH/TO/CHANGED_FILE"
git diff --cached
git commit -m "Describe the tested research change"
git push
```

Stage code, settings and relevant notes deliberately. Keep datasets, generated
result archives and credentials outside Git.

From the campus PC, choose the server connection you need:

```powershell
# SCRP
ssh s1155269217@scrp-login.econ.cuhk.edu.hk
```

```powershell
# AI Hub; use your current working connection if the site address changes.
ssh tian.xu@137.189.75.110
```

**SCRP or AI Hub — Bash, after login and before preparing new experiments:**

```bash
project_dir="$HOME/research-rep-gxk2020eapml"
cd "$project_dir"

git status --short
git pull --ff-only
uv sync --locked --extra nn
git rev-parse HEAD
```

Use the project's required extras; `--extra nn` installs its NN dependency.
Install dependencies on the login/head node. Preserve the original checkout when
queued/running/resumable experiments depend on it. Git changes do not transfer
the research datasets.

## 4. Transfer input data through the campus PC

**Main PC → Netdisk → campus PC → SCRP / AI Hub**

Use the Netdisk desktop client for the PC-to-PC stages. Keep each archive and its
checksum file together. The campus PC uploads them to the selected server using
SFTP. Use this step only for required datasets that the server does not already
have in the correct version.

### 4.1 Package the dataset on the main PC

**Main PC — PowerShell:**

```powershell
$transferDir = "D:\data_transfer"
New-Item -ItemType Directory -Path $transferDir -Force | Out-Null

# Choose one payload relative to the project's data directory.
$payload = "processed/gkx_dev_p05_seed2020_parquet"
# Full panel: $payload = "processed/gkx_parquet"
# Raw inputs: $payload = "raw"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$payloadLabel = $payload.Replace('/', '_')
$archiveName = "input_$payloadLabel-$stamp.tar.gz"
$archivePath = Join-Path $transferDir $archiveName

tar -czf $archivePath -C (Join-Path $projectDir "data") $payload
if ($LASTEXITCODE -ne 0) { throw "Dataset archive creation failed." }

$hash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText("$archivePath.sha256", "$hash  $archiveName`n", [Text.Encoding]::ASCII)
```

Upload the `.tar.gz` and `.sha256` files to Netdisk. On the campus PC, download
both to `D:\data_transfer`. Transfer prepared panels when the server can directly
reuse them; transfer raw inputs when server-side preparation requires them.

### 4.2 Verify the archive on the campus PC

**Campus PC — PowerShell:**

```powershell
$transferDir = "D:\data_transfer"
$archiveName = "PASTE_INPUT_ARCHIVE_NAME.tar.gz"
$archivePath = Join-Path $transferDir $archiveName

$expected = ((Get-Content -LiteralPath "$archivePath.sha256" -Raw).Trim() -split '\s+')[0]
$actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
if ($expected -notmatch '^[0-9a-fA-F]{64}$' -or $actual -ine $expected) {
    throw "Archive checksum mismatch. Obtain the intact archive before continuing."
}
Write-Host "Archive SHA-256 verified."
```

SHA-256 checks that the transferred archive matches the packaged file. Dataset
and experiment checks after extraction provide the next layer of verification.

### 4.3 Upload from the campus PC

Choose **one** server profile. The AI Hub address below is listed in its
[official guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide#direct-login-using-ssh-command);
use your working saved connection if the address changes. SCRP access follows
its [account/access guide](https://scrp.econ.cuhk.edu.hk/guide/account-and-access).

**Campus PC — PowerShell, SCRP profile:**

```powershell
$cluster = "s1155269217@scrp-login.econ.cuhk.edu.hk"
$remoteData = "/data/users/s1155269217/GKX2020/data"
```

**Campus PC — PowerShell, AI Hub profile:**

```powershell
$cluster = "tian.xu@137.189.75.110"
$remoteData = "/hpchome/tian.xu/research-rep-gxk2020eapml/data"
```

The remote data directory must exist. Run the upload block after selecting the
profile and setting `$archiveName` in Section 4.2:

```powershell
@(
    'lcd D:/data_transfer'
    "cd $remoteData"
    "put $archiveName"
    "put $archiveName.sha256"
    'bye'
) | sftp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $cluster
```

Enter authentication when prompted. Keep archives intact on the campus PC.

### 4.4 Verify and extract on the target server

**Target server — Bash:**

```bash
# SCRP example; on AI Hub use the remoteData path selected above.
data_root="/data/users/s1155269217/GKX2020/data"
archive="PASTE_INPUT_ARCHIVE_NAME.tar.gz"
payload="processed/gkx_dev_p05_seed2020_parquet"

cd "$data_root"
if [ -e "$data_root/$payload" ]; then
  printf 'Dataset destination exists. Verify/use the existing version or choose a new versioned destination.\n'
elif sha256sum -c "$archive.sha256"; then
  tar -xzf "$archive" -C "$data_root"
fi
```

Set `payload` to the same relative path packaged in Section 4.1. Check that the
expected metadata/part files are present and use the same data version as the
local pilot. The normal model reader also validates the dataset's saved metadata
and partition hashes as it reads them. Preserve existing raw/prepared datasets.

## 5. Run a small server test

Run a pilot on each server before a new full experiment. Confirm execution,
data/sample identity, training diagnostics and consistency with the local test.
GPU differences can cause small numerical differences; investigate material ones.

**SCRP — Bash, project folder:**

```bash
.venv/bin/python -B run_experiment.py nn1 --scrp \
  --sample p05 --years 1987 \
  --gpu-type a100 --gpus 1 --parallel 1 \
  --cpus 4 --mem 8G --time 00:30:00
```

**AI Hub — Bash, project folder:**

```bash
.venv/bin/python -B run_experiment.py nn1 --aihub \
  --sample p05 --years 1987 \
  --partition gpuh --gpu-type any --gpus 1 --parallel 1 \
  --cpus 4 --mem 8G --time 00:30:00
```

`gpuh` is the partition observed on this account; check current `sinfo` output.
See [monitoring](#7-monitor-progress-and-handle-interruptions) and
[collection](#8-collect-and-verify-on-the-server) for the pilot's next steps.
Save its experiment ID and review its results before launching the full sample.

Adding `--dry-run` prepares an experiment without submitting it. Submit that
prepared experiment through its printed ID using `resume`; repeating the model
command creates a different experiment.

## 6. Launch the full experiments

Record the model/specification, sample, intended period, code version, server and
experiment ID. Keep pilots separate from full-sample production results. Resource
requests are examples to adjust after the pilot and current account checks.

### 6.1 SCRP example

**SCRP — Bash:**

```bash
qos
sinfo -N -O NodeList:20,Partition:12,StateComplete:35,Gres:25,GresUsed:50

.venv/bin/python -B run_experiment.py nn3 --scrp \
  --sample full --years 1987:2016 \
  --gpu-type rtx3090 --gpus 1 --parallel 2 \
  --cpus 8 --mem 96G --time 12:00:00
```

Use `--gpu-type a100` when requesting A100 tasks. CPU methods omit the GPU flags
and use appropriate CPU/RAM allocations. Check the
[SCRP resource guide](https://scrp.econ.cuhk.edu.hk/guide/slurm) and live account
output when changing requests.

### 6.2 AI Hub example

**AI Hub — Bash:**

```bash
sinfo -N -O NodeList:20,Partition:12,StateComplete:35,Gres:25,GresUsed:50

.venv/bin/python -B run_experiment.py nn2 --aihub \
  --sample full --years 1987:2016 \
  --partition gpuh --gpu-type any --gpus 1 --parallel 3 \
  --cpus 8 --mem 80G --time 12:00:00
```

Here `--gpus 1` is per annual task; `--parallel 3` allows three simultaneous
tasks. RAM and CPU requests are also per task: this example totals at most 240G
and 24 CPUs. Check other active work and the
[AI Hub resource guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide#resource-quota)
before increasing concurrency or allocations. The launcher also enforces its
recorded student resource budget.

### 6.3 Split runs and combined IDs

To divide one model across GPU types/year blocks, prepare each source experiment
with its own resources and disjoint years using `--dry-run`. For example, one
SCRP source may cover 1987-2003 on RTX 3090 and another 2004-2016 on A100.
Register the two printed source IDs:

**SCRP — Bash:**

```bash
early="PASTE_EARLY_SOURCE_ID"
late="PASTE_LATE_SOURCE_ID"
.venv/bin/python -B run_experiment.py combine "$early" "$late" --years 1987:2016

run="PASTE_COMBINED_ID"
.venv/bin/python -B run_experiment.py resume "$run"
```

Use the combined ID for later status and collection. Each source retains its
resource settings. On AI Hub, add `--aihub` to combine/resume and ID lookup.
The current launcher combines sources from the same backend with compatible
settings/environments and complete, non-overlapping coverage. It does not combine
SCRP and AI Hub source jobs into one execution ID. Compare their completed model
results together during local research analysis. Detailed preparation examples
are in the site command references linked above.

## 7. Monitor progress and handle interruptions

Check both the scheduler and saved research progress throughout execution.

**SCRP — Bash:**

```bash
run="PASTE_EXPERIMENT_OR_COMBINED_ID"
.venv/bin/python -B run_experiment.py list
.venv/bin/python -B run_experiment.py status "$run" --refresh
```

**AI Hub — Bash:**

```bash
run="PASTE_EXPERIMENT_OR_COMBINED_ID"
.venv/bin/python -B run_experiment.py list --aihub
.venv/bin/python -B run_experiment.py status "$run" --aihub --refresh
```

**Either server — Bash:**

```bash
job="PASTE_SLURM_JOB_ID"
squeue -u "$USER"
squeue -r -j "$job" -o "%.18i %.12T %.10M %R"
scontrol show job "$job"
sacct -X --array -j "$job" --format=JobID%22,State%20,ExitCode,Elapsed,Start,End
```

Pending jobs are waiting. An empty queue does not establish successful completion.
If accounting is unavailable, use `status` without `--refresh` and inspect logs.
`last_action=submitted` records a past action; saved-year counts show the files
currently present.

While a source experiment is executing, inspect its directory:

```bash
exp="/FULL/PATH/TO/SOURCE_EXPERIMENT"
task=0
tail -n 80 "$exp"/annual/logs/"$job"_"$task"_*/run.log
tail -n 80 "$exp"/submissions/*/"slurm-$job"_"$task.out"
```

To resume an existing run after resolving an interruption:

```bash
# SCRP; add --aihub for an AI Hub ID.
.venv/bin/python -B run_experiment.py resume "$run"
```

Completed annual fits are retained. If the research specification changes,
prepare a new experiment. To deliberately cancel a job with an incorrect request:

```bash
scancel "$job"
```

## 8. Collect and verify on the server

After all planned annual tasks have finished, collect the run using its existing
ID. Collection builds pooled reports from saved fits. For this replication, the
full-period comparison requires full-universe results for all 1987-2016 test years.

**SCRP — Bash:**

```bash
.venv/bin/python -B run_experiment.py collect "$run"
```

**AI Hub — Bash:**

```bash
.venv/bin/python -B run_experiment.py collect "$run" --aihub
```

For a combined ID, collection stores full sub-experiment snapshots, including
saved models, forecasts, diagnostics, settings and logs, inside that one combined
folder. Download that complete folder. Original sub-runs remain preserved.

For older/incomplete combined bundles, repair from the intact originals on the
server, then verify all embedded files:

```bash
# SCRP; add --aihub to each command for an AI Hub ID.
.venv/bin/python -B run_experiment.py collect "$run" --repair-sources
.venv/bin/python -B run_experiment.py status "$run" --verify
```

Repair requires the recorded inventory and matching saved bytes; it does not
retrain models or replace results with a different fit. New/repaired bundles use
short `sources/00`, `sources/01` directories. Allow temporary space for a second
embedded bundle during repair.

| Check | Meaning |
| --- | --- |
| `saved=30/30` | All planned annual checkpoint files are present for 1987-2016 |
| `reports=ready` | Aggregate reports are available and combined source presence checks pass |
| `sources=present` | Combined source file inventory was checked |
| `sources=verified` | Every embedded source file also passed its SHA-256 check |
| `sources=incomplete` / `reports=incomplete` | Restore the damaged combined bundle before using it |

`sources` verification applies to combined bundles; ordinary single experiments
show `sources=n/a`. File integrity and job completion are distinct from research
validation. Review scores and training diagnostics as part of analysis.

To inspect the collected reports:

```bash
exp="/FULL/PATH/TO/COLLECTED_EXPERIMENT"
cat "$exp/annual/combined/table1_all.csv"
head -n 15 "$exp/annual/combined/annual_metrics.csv"
# Combined NN diagnostics:
head -n 15 "$exp/annual/combined/nn_training_summary.csv"
```

## 9. Transfer results back and verify locally

**SCRP / AI Hub → campus PC → Netdisk → main PC**

Create fresh archives after the final collection/repair. Earlier archives retain
their earlier contents. Transfer whole archives between PCs; extract the model
folders only when ready to verify and use them locally.

### 9.1 Package one complete experiment on its server

**SCRP — Bash, choose this results root:**

```bash
results_root="/data/users/s1155269217/GKX2020/results/experiments"
```

**AI Hub — Bash, choose this results root instead:**

```bash
results_root="$project_dir/results/aihub_experiments"
```

**Selected server — Bash:**

```bash
run="PASTE_COLLECTED_EXPERIMENT_OR_COMBINED_ID"
export_root="$results_root/transfer_archives"
mkdir -p "$export_root"
cd "$export_root"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
archive="$run-results-$stamp.tar.gz"

if tar -czf "$archive" -C "$results_root" "$run"; then
  sha256sum "$archive" > "$archive.sha256"
  sha256sum -c "$archive.sha256"
fi
```

Record the archive filename and export directory. Repeat for other completed
experiments. For combined runs, include their entire `sources/` tree. Raw and
prepared research datasets remain separate dependencies.

### 9.2 Download to the campus PC

Choose the matching server/export profile.

**Campus PC — PowerShell, SCRP:**

```powershell
$cluster = "s1155269217@scrp-login.econ.cuhk.edu.hk"
$remoteExport = "/data/users/s1155269217/GKX2020/results/experiments/transfer_archives"
```

**Campus PC — PowerShell, AI Hub:**

```powershell
$cluster = "tian.xu@137.189.75.110"
$remoteExport = "/hpchome/tian.xu/research-rep-gxk2020eapml/results/aihub_experiments/transfer_archives"
```

**Campus PC — PowerShell, after selecting the profile:**

```powershell
$transferDir = "D:\data_transfer"
New-Item -ItemType Directory -Path $transferDir -Force | Out-Null
$archiveName = "PASTE_RESULTS_ARCHIVE_NAME.tar.gz"

@(
    'lcd D:/data_transfer'
    "cd $remoteExport"
    "get $archiveName"
    "get $archiveName.sha256"
    'bye'
) | sftp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $cluster
```

Verify the archive using the PowerShell checksum block in Section 4.2 with this
results archive's filename. Upload both files to Netdisk using its desktop client.

### 9.3 Download and extract on the main PC

Download the same archive and `.sha256` file from Netdisk to `D:\data_transfer`.
Repeat the archive checksum check from Section 4.2 on the main PC. Then extract
to a new staging directory:

**Main PC — PowerShell:**

```powershell
$transferDir = "D:\data_transfer"
$archiveName = "PASTE_RESULTS_ARCHIVE_NAME.tar.gz"
$extractDir = Join-Path $transferDir ("results_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $extractDir -ErrorAction Stop | Out-Null

tar -xzf (Join-Path $transferDir $archiveName) -C $extractDir
if ($LASTEXITCODE -ne 0) { throw "Result extraction failed." }
```

Use a short destination and a fresh extraction directory. Keep the archive intact
until the extracted files pass verification. An archive checksum alone cannot
detect a tool dropping files during extraction.

### 9.4 Verify the extracted combined experiment

**Main PC — PowerShell, from the updated project folder:**

```powershell
Set-Location -LiteralPath $projectDir
$run = "PASTE_COMBINED_ID"
$extractedRun = Join-Path $extractDir $run

.\.venv\Scripts\python.exe -B run_experiment.py status $extractedRun --verify
if ($LASTEXITCODE -ne 0) { throw "Extracted bundle verification failed." }
```

For a full-period combined run, require `saved=30/30`, `reports=ready` and
`sources=verified`. The command verifies extracted files without the original
server directories. Use `status` on an ordinary single experiment to check saved
coverage/report presence; its archive checksum verifies the transferred package.

### 9.5 Place verified results in the local project

Analysis can use an extracted experiment's absolute path directly. To retain it
under the project, copy the whole folder into the appropriate import location.

**Main PC — PowerShell:**

```powershell
# AI Hub imports may use results\aihub_imports instead.
$importRoot = Join-Path $projectDir "results\scrp_imports"
New-Item -ItemType Directory -Path $importRoot -Force | Out-Null
$destination = Join-Path $importRoot $run

if (Test-Path -LiteralPath $destination) {
    throw "Import destination exists. Use a new dated import root and preserve the earlier copy."
}
Copy-Item -LiteralPath $extractedRun -Destination $destination -Recurse -ErrorAction Stop

.\.venv\Scripts\python.exe -B run_experiment.py status $destination --verify
if ($LASTEXITCODE -ne 0) { throw "Project import verification failed." }
```

The example is for combined runs. For a replacement download, use a fresh dated
import root such as `results\scrp_imports_20261029` and explicitly select those
paths during analysis. Preserve the previous import until the replacement has
passed verification. Keep the archive/checksum as the transfer record.

## 10. Analyze, document and start the next round

Perform the research comparison on the main PC using the verified saved results.

- Check model, dataset, period coverage and observations before pooling/comparing.
- Use full-universe, full-period runs for the main 1987-2016 replication table;
  report pilots or partial-period diagnostics separately.
- Review training/validation behavior and prediction diagnostics alongside scores.
- Generate comparison tables and figures, investigate differences from the paper,
  and document research choices and unresolved issues.
- Save derived analysis separately from imported run folders, for example under
  `results/replication_comparison`. Retain the experiment IDs and data/code versions
  used by each analysis.
- Keep the training, validation and test roles fixed. Document revisions motivated
  by observed results so later comparisons retain their research context.
- Preserve the completed first-round results as a reference. Make subsequent
  method changes locally, repeat the pilot checks and create new experiments.

An experiment is ready for research analysis after computation, collection,
transfer and local integrity checks have all finished. Agreement with the paper
is a separate conclusion supported by the analysis.

## References

- [SCRP command reference](run_scrp.md)
- [AI Hub command reference](run_aihub.md)
- [Project experiment guide](../Running%20Experiments.md)
- [SCRP access and transfer](https://scrp.econ.cuhk.edu.hk/guide/account-and-access)
- [SCRP Slurm guide](https://scrp.econ.cuhk.edu.hk/guide/slurm)
- [SCRP storage guide](https://scrp.econ.cuhk.edu.hk/guide/storage)
- [AI Hub official guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide)
