# Research experiment workflow template

This guide follows an experiment from local development to server execution and back to local analysis. It keeps the same workflow across projects; replace the example project paths, datasets, methods, configurations and periods with those used by the project. The SCRP and AI Hub accounts and transfer routes remain available across projects. Last reviewed: 2026-10-30.

## Workflow at a glance

**Prepare locally → test locally → synchronize code and data → test on the server → run and monitor full experiments → collect and verify → transfer back → analyze locally.**

| Step                                                                     | Work                                                             | Where                | Ready to proceed when                                         |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------- | -------------------- | ------------------------------------------------------------- |
| [1. Prepare data](#1-prepare-data-on-the-main-pc)                        | Choose the sample and inputs; prepare that dataset | Main PC              | Dataset, features and source versions are clear               |
| [2. Develop and test](#2-develop-and-test-locally)                       | Modify the method and check small-sample results                 | Main PC              | Training, predictions and diagnostics have been checked       |
| [3. Synchronize code](#3-record-and-synchronize-the-tested-code)         | Commit/push tested changes and pull on the servers               | Main PC and servers  | Each experiment has a known code version                      |
| [4. Transfer inputs](#4-transfer-required-input-data)                    | Main PC → server, directly or via the campus PC                  | Selected PC and server | Required datasets are complete on the target server         |
| [5. Test on the server](#5-run-a-small-server-test)                      | Run a small sample with the intended server setup                | SCRP / AI Hub        | The pilot finishes and its diagnostics are reviewed           |
| [6. Run the full experiment](#6-launch-the-full-experiments)             | Submit each planned model, dataset, config and test period       | SCRP / AI Hub        | Experiment IDs and intended coverage are recorded             |
| [7. Monitor execution](#7-monitor-progress-and-handle-interruptions)     | Check queues, saved progress, logs and failures                  | SCRP / AI Hub        | All planned tasks and their outputs are saved                             |
| [8. Collect and verify](#8-collect-and-verify-on-the-server)             | Build pooled reports and complete result bundles                 | SCRP / AI Hub        | Coverage and saved files pass verification                    |
| [9. Bring results home](#9-transfer-results-back-and-verify-locally)     | Server → main PC, directly or via the campus PC                  | Server and selected PCs | Archives and extracted results pass local verification     |
| [10. Analyze and iterate](#10-analyze-document-and-start-the-next-round) | Compare methods, investigate differences and document findings   | Main PC              | Reviewed outputs and the next research questions are recorded |

Monitoring continues throughout server execution. Input transfers are needed when deploying a dataset for the first time or introducing a new data version.

Follow the requested transfer route. Otherwise, choose according to the actual file/archive size and connection reliability: small files can go directly between the main PC and server; the campus PC is an optional intermediary for large files or unreliable direct connections. If the intended route is unclear, present both options with commands labeled for the PC on which they run. Measure the archive before describing it as small; the number of input variables alone does not determine result size.

**Campus PC: use `D:\data_transfer` for every file transfer**, including datasets, result archives and checksums. Create the folder before downloading; keep any transfer subfolders beneath it. Main-PC paths are separate: the examples use `E:\data_transfer`, but use the actual folder if files are already stored elsewhere. Creating a directory does not copy files into it.

| Location/service | Role                                                                   |
| ---------------- | ---------------------------------------------------------------------- |
| Main PC          | Data preparation, model development, local tests and research analysis |
| Campus PC        | Optional transfer bridge; all transfer staging uses `D:\data_transfer` |
| SCRP / AI Hub    | Server pilots, full experiments and storage of completed runs          |
| Git              | Synchronize code and experiment settings                               |
| Netdisk          | Optional exchange of datasets and result archives between the two PCs |

## Using the commands

- **PowerShell** blocks run on the specified Windows PC.
- **Bash** blocks run after logging into the specified server.
- Run separate setup, transfer and verification blocks in order, using the same terminal for variables shared between blocks. Each block states its purpose and the PC/server on which it runs.
- Each project uses its own environment. The examples use `.venv` and `uv sync --locked`; retain the project's existing environment setup if it differs, and include any required dependency extras.
- Replace `PROJECT_NAME`, `PATH/TO/...`, `/FULL/PATH/TO/...` and values beginning with `PASTE_` before running a block. Server storage paths are layout examples; confirm the actual destinations.
- `dataset_dev_v1` and `dataset_full_v1` are example development/full datasets; `model_a`, `model_b` and their configuration names are example methods. Replace them with the project's actual identifiers.
- The Python examples use `run_experiment.py` as an illustrative launcher with `list`, `status`, `resume`, `collect` and optional `combine` commands. Adapt the entry point, flags and saved-file names to the project's actual interface. These examples do not imply that another project already implements those commands.
- The example pilot uses 2020 and the example full period uses 2020–2024. Replace the dates and `--years` selector with the project's intended periods or other task identifiers.
- Re-enter the project and reset the relevant variables after opening a new terminal or reconnecting. Initial server checkout/data-path setup is assumed; use the linked site guides when setting up a server for the first time.
- An **experiment ID** identifies a research run and its saved outputs. A **Slurm job ID** identifies a scheduler submission and is used by `squeue`, `sacct` and `scancel`.
- Examples that name a method create a new experiment. Use its existing ID for subsequent monitoring, resumption and collection.
- Optional features such as `--dry-run`, `--verify`, combined IDs and resumption apply when supported by the project. Use its documented equivalents.

The numbered steps, labeled command blocks and explicit replacement values are intended to be readable by both people and agents. Add links to the project's own run, data and configuration guides when adapting this template. Official SCRP and AI Hub guides are linked under [References](#references).

## 1. Prepare data on the main PC

Preserve the raw inputs. Keep raw data, prepared datasets and experiment outputs separate. Choose both the sample and the input specification before testing. Use matching prepared datasets locally and on the servers, with consistent dates, variables and outcome definitions.

**Main PC — PowerShell:**

```powershell
$projectDir = "E:\PATH\TO\PROJECT_NAME"
Set-Location -LiteralPath $projectDir

# Install/update this project's locked environment when needed.
uv sync --locked
```

### 1.1 Choose the dataset

| Dataset name | Sample | Input specification |
| --- | --- | --- |
| `dataset_dev_v1` | Reproducible development sample | Main specification |
| `dataset_full_v1` | Full research sample | The same main specification |
| `dataset_alternative_dev_v1` | The same development sample | Alternative specification, when needed |
| `dataset_alternative_full_v1` | The same full research sample | The same alternative specification |

These names are examples. Define the actual sample, variables, transformations, information timing and dataset version in the project's data notes. When comparing input specifications, retain matching observations and outcomes unless the research design calls for different samples.

Changing an experiment's evaluation period does not automatically change its sample or input specification. Keep those choices explicit.

### 1.2 Prepare only missing datasets

Reuse existing verified datasets. For a new dataset, follow the project's preparation and development-sample instructions. Use a reproducible sampling rule and preserve source identifiers and dates.

**Main PC — PowerShell, choose the dataset to prepare:**

```powershell
$dataset = "dataset_full_v1"
$datasetDir = Join-Path $projectDir "data\processed\$dataset"

if (Test-Path -LiteralPath $datasetDir) {
    Write-Host "Dataset folder exists. Check its completion metadata before reusing it."
} else {
    # Replace this with the project's actual preparation command.
    .\.venv\Scripts\python.exe -B PATH/TO/PREPARE_DATA.py --dataset $dataset
    if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed." }
}
```

Check whether the experiment launcher prepares data automatically or requires this separate preparation step. Preserve existing and interrupted builds for inspection instead of overwriting them.

Record the required source files and dependencies for each prepared dataset. Verify row counts, identifiers, dates, variables, missing values and other properties relevant to the research before marking preparation complete.

### 1.3 Check the selected dataset and retain its identity

**Main PC — PowerShell:**

```powershell
$dataset = "dataset_dev_v1"  # Select the dataset for the next pilot or transfer.
$datasetDir = Join-Path $projectDir "data\processed\$dataset"
$metadataPath = Join-Path $datasetDir "metadata.json"

# Use this project's actual metadata/manifest file and format.
Get-Content -LiteralPath $metadataPath -Raw -ErrorAction Stop | ConvertFrom-Json | Format-List
Get-FileHash -LiteralPath $metadataPath -Algorithm SHA256
```

Check that the metadata identifies the intended dataset version and completed preparation. Use the project's data-validation command where available; a metadata hash alone does not validate every data file.

Record the dataset name, input specification and metadata hash with the experiment plan. Transfer the entire required dataset folder, including metadata, provenance and data files. Keep the original preparation inputs for reproducibility.

## 2. Develop and test locally

Implement or revise the research method on the main PC. Test the intended specification on a reproducible development sample and a limited period before preparing production runs. Review fitting behavior, parameter selection, predictions and evaluation results. Record the research choices and unresolved questions.

**Main PC — PowerShell, from the project folder:**

```powershell
# One example evaluation year on the development dataset.
$dataset = "dataset_dev_v1"
.\.venv\Scripts\python.exe -B run_experiment.py model_a --local --dataset $dataset --config model_a_v1 --years 2020 --cpus 4

.\.venv\Scripts\python.exe -B run_experiment.py list
```

Copy the experiment ID printed by the run:

```powershell
$run = "PASTE_LOCAL_EXPERIMENT_ID"
.\.venv\Scripts\python.exe -B run_experiment.py status $run
.\.venv\Scripts\python.exe -B run_experiment.py collect $run
```

For an alternative input specification, repeat the local command with its corresponding development dataset, retaining the intended method, configuration and period. Each invocation creates its own experiment ID. Keep the comparison protocol aligned while allowing any planned parameter selection to use its designated data.

Use the method and configuration names supported by the project. For example:

| Command | Research role |
| --- | --- |
| `model_a` | Main method or baseline |
| `model_b` | Comparison method |

**Optional separate comparison run — PowerShell:**

```powershell
.\.venv\Scripts\python.exe -B run_experiment.py model_b --local --dataset dataset_dev_v1 --config model_b_v1 --years 2020 --cpus 4
```

Review the saved launch record and resolved settings for the actual dataset, method and configuration. Check important calculations and saved outputs, including restoration or resumption when later work depends on them. A completed pilot establishes execution and diagnostic evidence; substantive research conclusions still require review.

## 3. Record and synchronize the tested code

Commit the tested implementation and settings, then pull that version on each server used for the next experiment. Preserve the recorded code/environment for existing experiments that may need resumption. Start a new experiment when the research specification changes.

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

Stage code, settings and relevant notes deliberately. Keep datasets, generated result archives and credentials outside Git.

From the main PC or campus PC with a working server connection, choose the server you need:

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
project_dir="$HOME/PROJECT_NAME"
cd "$project_dir"

git status --short
git pull --ff-only
uv sync --locked
git rev-parse HEAD
```

Install the project's required dependencies and extras on the login/head node. Preserve the original checkout and environment when queued/running/resumable experiments depend on them. Record the code revision and environment used by each experiment; changes that invalidate an existing run's research or execution identity require a new experiment. Git changes do not transfer or prepare research datasets.

## 4. Transfer required input data

**Direct: main PC → SCRP / AI Hub. Optional intermediary: main PC → Netdisk → campus PC → SCRP / AI Hub.**

Use SFTP from the PC that currently holds the source files. Small inputs can be uploaded directly from the main PC's actual project folder; no archive or intermediary is required. See the [SCRP access and transfer guide](https://scrp.econ.cuhk.edu.hk/guide/account-and-access) and [AI Hub guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide). Check that the local files exist and the remote destination is correct before uploading. Use this step only for required files or datasets that the server does not already have in the correct version.

The archive workflow below supports both routes. For direct uploads, keep the archive and checksum on the main PC and use that PC in Sections 4.2–4.3. For the intermediary route, use the Baidu Netdisk desktop client to download both files into `D:\data_transfer` on the campus PC first, then verify and upload there. Earlier disconnections during large direct transfers make the campus PC useful as a fallback; they do not make it mandatory for every transfer.

Prepare each selected dataset once on the main PC and transfer that same completed folder to the servers that need it. Transfer the development version for the server pilot, then its full-sample counterpart for production. Include any additional inputs required by the experiment. Server-side data preparation is an alternative when needed; it requires the source data and preparation inputs, and substantial preparation belongs on an allocated compute node.

### 4.1 Package the dataset on the main PC

**Main PC — PowerShell:**

```powershell
$transferDir = "E:\data_transfer"  # Main-PC staging folder; choose another local folder if needed.
New-Item -ItemType Directory -Path $transferDir -Force | Out-Null

# Choose the prepared dataset being deployed; repeat for other required datasets.
$dataset = "dataset_dev_v1"
# Production example: dataset_full_v1.
$payload = "processed/$dataset"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$payloadLabel = $payload.Replace('/', '_')
$archiveName = "input_$payloadLabel-$stamp.tar.gz"
$archivePath = Join-Path $transferDir $archiveName

tar -czf $archivePath -C (Join-Path $projectDir "data") $payload
if ($LASTEXITCODE -ne 0) { throw "Dataset archive creation failed." }

$hash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText("$archivePath.sha256", "$hash  $archiveName`n", [Text.Encoding]::ASCII)
```

For a direct upload, use the `.tar.gz` and `.sha256` files already on the main PC. For the intermediary route, upload both to Baidu Netdisk, create `D:\data_transfer` on the campus PC and download both files there. Transfer prepared datasets when the server can directly reuse them; transfer raw inputs when server-side preparation requires them. To package raw inputs instead, set `$payload = "raw"` before creating the archive. Keep preparation inputs and finished dataset archives clearly labeled. Changing a folder's name does not change the registered dataset identity in its metadata.

### 4.2 Verify the archive on the uploading PC

Select the folder on the PC doing the upload.

**Campus PC for intermediary uploads — PowerShell:**

```powershell
$transferDir = "D:\data_transfer"
New-Item -ItemType Directory -Path $transferDir -Force | Out-Null
```

**Main PC for direct uploads — PowerShell, choose this block instead:**

```powershell
$transferDir = "E:\data_transfer"  # Use the archive's actual folder on the main PC.
```

**Selected uploading PC — PowerShell, check the archive:**

```powershell
$archiveName = "PASTE_INPUT_ARCHIVE_NAME.tar.gz"
$archivePath = Join-Path $transferDir $archiveName

$expected = ((Get-Content -LiteralPath "$archivePath.sha256" -Raw).Trim() -split '\s+')[0]
$actual = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
if ($expected -notmatch '^[0-9a-fA-F]{64}$' -or $actual -ine $expected) {
    throw "Archive checksum mismatch. Obtain the intact archive before continuing."
}
Write-Host "Archive SHA-256 verified."
```

SHA-256 checks that the transferred archive matches the packaged file. Dataset and experiment checks after extraction provide the next layer of verification.

### 4.3 Upload from the selected PC

Choose **one** server profile. The AI Hub address below is listed in its [official guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide#direct-login-using-ssh-command); use your working saved connection if the address changes. SCRP access follows its [account/access guide](https://scrp.econ.cuhk.edu.hk/guide/account-and-access).

**Uploading PC — PowerShell, SCRP profile:**

```powershell
$cluster = "s1155269217@scrp-login.econ.cuhk.edu.hk"
$remoteData = "/data/users/s1155269217/PROJECT_NAME/data"
```

**Uploading PC — PowerShell, AI Hub profile:**

```powershell
$cluster = "tian.xu@137.189.75.110"
$remoteData = "/hpchome/tian.xu/PROJECT_NAME/data"
```

The remote data directory must exist. Run the upload block after selecting the profile and setting `$archiveName` in Section 4.2:

```powershell
@(
    ('lcd "{0}"' -f $transferDir.Replace('\', '/'))
    "cd $remoteData"
    "put $archiveName"
    "put $archiveName.sha256"
    'bye'
) | sftp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $cluster
```

Enter authentication when prompted. Keep archives intact on the uploading PC. `$transferDir` must contain the archive and checksum on that PC; creating a folder does not copy files into it.

### 4.4 Verify and extract on the target server

**Target server — Bash:**

```bash
# SCRP example; on AI Hub use the remoteData path selected above.
data_root="/data/users/s1155269217/PROJECT_NAME/data"
archive="PASTE_INPUT_ARCHIVE_NAME.tar.gz"
dataset="dataset_dev_v1"
payload="processed/$dataset"

cd "$data_root"
if [ -e "$data_root/$payload" ]; then
  printf 'Dataset destination exists. Verify/use the existing version or choose a new versioned destination.\n'
elif sha256sum -c "$archive.sha256"; then
  tar -xzf "$archive" -C "$data_root"
fi

# For a prepared dataset, use its actual metadata file and compare with Section 1.3.
sha256sum "$data_root/$payload/metadata.json"
```

Set `payload` to the same relative path packaged in Section 4.1. Check that the expected metadata and data files are present and use the same version as the local pilot. Run the project's data integrity checks after extraction. Preserve existing raw/prepared datasets. For raw-only archives, omit the prepared-dataset metadata hash command. The example paths use `data/processed/DATASET_NAME`, possibly through a server data symlink; verify that the upload destination and project reader refer to the same location.

## 5. Run a small server test

Run a pilot on each server with the intended dataset family and method configuration before its full experiment. Match the local pilot's development dataset, configuration and evaluation period. Confirm data identity, fitting diagnostics and numerical consistency; investigate material cross-device differences.

**SCRP — Bash, project folder:**

```bash
dataset="dataset_dev_v1"
.venv/bin/python -B run_experiment.py model_a --scrp \
  --dataset "$dataset" --config model_a_v1 --years 2020 \
  --gpu-type a100 --gpus 1 --parallel 1 \
  --cpus 4 --mem 8G --time 00:30:00
```

**AI Hub — Bash, project folder:**

```bash
dataset="dataset_dev_v1"
.venv/bin/python -B run_experiment.py model_a --aihub \
  --dataset "$dataset" --config model_a_v1 --years 2020 \
  --partition gpuh --gpu-type any --gpus 1 --parallel 1 \
  --cpus 4 --mem 8G --time 00:30:00
```

These are GPU examples; CPU methods omit GPU flags and use a suitable CPU partition. `gpuh` is the partition previously observed on this AI Hub account; check current `sinfo` output. Resource requests are initial examples to adjust from measured requirements and current site limits.

See [monitoring](#7-monitor-progress-and-handle-interruptions) and [collection](#8-collect-and-verify-on-the-server) for the pilot's next steps. Save its experiment ID and review its results before launching the full sample. Pilot another method or input specification when its execution requirements differ.

If the launcher supports `--dry-run`, use it to prepare an experiment without submitting. Submit that prepared experiment through its printed ID using the documented command; repeating a new-experiment command creates a different experiment.

## 6. Launch the full experiments

Record the method, dataset name, input specification, configuration, intended coverage, code version, server and experiment ID. Prepare and transfer the selected full-sample dataset first. Keep pilots separate from production results. Resource requests are examples to adjust after the pilot and current account checks.

For example, one production case is `model_a` + `dataset_full_v1` + `model_a_v1` + `2020:2024`. Its comparison method or alternative dataset is a separate case. A different research specification starts a new experiment; it is not a change to an existing ID.

### 6.1 SCRP example

**SCRP — Bash:**

```bash
qos
sinfo -N -O NodeList:20,Partition:12,StateComplete:35,Gres:25,GresUsed:50

dataset="dataset_full_v1"
.venv/bin/python -B run_experiment.py model_a --scrp \
  --dataset "$dataset" --config model_a_v1 --years 2020:2024 \
  --gpu-type rtx3090 --gpus 1 --parallel 2 \
  --cpus 4 --mem 16G --time 02:00:00
```

Use `--gpu-type a100` when requesting A100 tasks. CPU methods omit the GPU flags and use appropriate CPU/RAM allocations. Check the [SCRP resource guide](https://scrp.econ.cuhk.edu.hk/guide/slurm) and live account output when changing requests.

### 6.2 AI Hub example

**AI Hub — Bash:**

```bash
sinfo -N -O NodeList:20,Partition:12,StateComplete:35,Gres:25,GresUsed:50

dataset="dataset_full_v1"
.venv/bin/python -B run_experiment.py model_b --aihub \
  --dataset "$dataset" --config model_b_v1 --years 2020:2024 \
  --partition gpuh --gpu-type any --gpus 1 --parallel 2 \
  --cpus 4 --mem 16G --time 02:00:00
```

Here `--gpus 1` is per task; `--parallel 2` allows two simultaneous tasks. RAM and CPU requests are also per task: this example totals at most two GPUs, 32G and eight CPUs. Check other active work and the [AI Hub resource guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide#resource-quota) before increasing concurrency or allocations. Confirm actual account limits rather than assuming that the project's launcher enforces them.

### 6.3 Split runs and combined IDs

If the project supports split runs and combined IDs, divide one method/dataset/configuration case into non-overlapping task blocks. The yearly example below uses two GPU types; adapt the split to the project's task definition. Both sources must use compatible data, inputs, research settings and execution environments. For example:

**SCRP — Bash, prepare two sources without submitting:**

```bash
dataset="dataset_full_v1"

.venv/bin/python -B run_experiment.py model_a --scrp \
  --dataset "$dataset" --config model_a_v1 --years 2020:2022 \
  --gpu-type rtx3090 --gpus 1 --parallel 2 \
  --cpus 4 --mem 16G --time 02:00:00 --dry-run

.venv/bin/python -B run_experiment.py model_a --scrp \
  --dataset "$dataset" --config model_a_v1 --years 2023:2024 \
  --gpu-type a100 --gpus 1 --parallel 2 \
  --cpus 4 --mem 16G --time 02:00:00 --dry-run
```

Register the two printed source IDs and submit them through one combined ID:

**SCRP — Bash:**

```bash
early="PASTE_EARLY_SOURCE_ID"
late="PASTE_LATE_SOURCE_ID"
.venv/bin/python -B run_experiment.py combine "$early" "$late" --years 2020:2024

run="PASTE_COMBINED_ID"
.venv/bin/python -B run_experiment.py resume "$run"
```

Use the combined ID for later status and collection. Each source retains its resource settings. On AI Hub, add `--aihub` to combine/resume and ID lookup.

Different methods, datasets and research specifications remain separate cases. Combine only compatible parts of the same planned case, checking for missing or duplicate tasks. Account for the total resources across all active source runs.

Check the project's supported combination and cross-server rules before using this example. If it has no combined-ID facility, retain the source IDs and use its documented collection procedure. Keep source provenance visible when comparing results from SCRP and AI Hub.

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

Pending jobs are waiting. An empty queue does not establish successful completion. If accounting is unavailable, inspect the project's saved progress and logs. A recorded submission is a past action; saved-task counts describe the outputs currently present. Use the saved launch record and resolved configuration to identify the specification rather than inferring it solely from a method name or completion count.

While a source experiment is executing, inspect its directory:

```bash
exp="/FULL/PATH/TO/SOURCE_EXPERIMENT"
# Replace these example log paths with those recorded by the project.
tail -n 80 "$exp/logs/run.log"
tail -n 80 "$exp/logs/slurm-$job.out"
```

To resume an existing run after resolving an interruption:

```bash
# SCRP; add --aihub for an AI Hub ID.
.venv/bin/python -B run_experiment.py resume "$run"
```

Use resumption only when supported and compatible with the saved experiment; confirm that it retains completed tasks and avoids duplicate active work. If the research specification changes, prepare a new experiment. To deliberately cancel a job with an incorrect request:

```bash
scancel "$job"
```

## 8. Collect and verify on the server

After all planned tasks have finished, collect the run using its existing ID. Collection builds aggregate reports from saved outputs. Check the intended sample and full period or task coverage for the main research comparison. Collect each method/dataset/configuration case separately. Use its combined ID when supported and when its task blocks were split.

**SCRP — Bash:**

```bash
.venv/bin/python -B run_experiment.py collect "$run"
```

**AI Hub — Bash:**

```bash
.venv/bin/python -B run_experiment.py collect "$run" --aihub
```

Check what collection includes. A combined folder may contain source snapshots or may only reference the original sub-runs. Ensure the package includes the required saved models or estimates, predictions or other outputs, diagnostics, settings and logs. Preserve the original sub-runs.

**Verify a collected experiment before packaging — Bash:**

```bash
# SCRP; add --aihub for an AI Hub ID.
# Use the project's actual integrity-verification command if its interface differs.
.venv/bin/python -B run_experiment.py status "$run" --verify
```

For an incomplete or damaged bundle, preserve the existing files, use the project's supported recovery procedure and verify again. Keep the restored files consistent with the recorded experiment identity.

| Check | Meaning |
| --- | --- |
| Saved task coverage | All planned tasks have their required outputs; no missing or duplicate tasks |
| Reports ready | Aggregate reports exist and pass the project's consistency checks |
| Required source files present | Combined/source dependencies needed for later use are included |
| File hashes verified | Packaged files match their recorded inventory |
| Incomplete or inconsistent outputs | Resolve the issue before treating the collection as complete |

File integrity and job completion are distinct from research validation. Review scores and fitting diagnostics as part of analysis.

To inspect the collected reports, adapt these example file names to the project's outputs:

```bash
exp="/FULL/PATH/TO/COLLECTED_EXPERIMENT"
cat "$exp/reports/summary.csv"
head -n 15 "$exp/reports/period_metrics.csv"
head -n 15 "$exp/reports/diagnostics.csv"
```

Keep the dataset, method and configuration identifiers visible when recording results. Check the resolved settings and source provenance when confirming a combined run's specification.

## 9. Transfer results back and verify locally

**Direct: SCRP / AI Hub → main PC. Optional intermediary: SCRP / AI Hub → campus PC → Netdisk → main PC.**

Create fresh archives after the final collection/repair. Earlier archives retain their earlier contents. Follow the requested route; when recommending one, check the archive's actual size and connection reliability. For the campus route, download into `D:\data_transfer` on the campus PC, then use Baidu Netdisk to transfer the archive and checksum to the main PC. Keep both files together and verify the checksum on each receiving PC before extraction.

### 9.1 Package one complete experiment on its server

Select only the experiment requested for analysis. A complete experiment means all files inside that one experiment folder, including saved estimates, outputs, settings, diagnostics and logs. Record its ID before packaging; repeat for additional experiments only when they are requested.

**SCRP — Bash, choose this results root:**

```bash
results_root="/data/users/s1155269217/PROJECT_NAME/results/experiments"
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
  ls -lh "$archive" "$archive.sha256"
fi
```

Record the archive filename, export directory and displayed size. The example assumes each experiment folder has the same name as its ID; use the actual folder name if it differs. For combined runs, include all required source snapshots or sub-run folders. Retain the exact dataset versions and any audit/raw inputs needed by later analysis; record dependencies kept outside the result package. Different research specifications retain their own result archives and experiment IDs.

### 9.2 Download to the selected PC

Choose the matching server/export profile. Run on the main PC for a direct download, or on the campus PC for the intermediary route.

**Downloading PC — PowerShell, SCRP:**

```powershell
$cluster = "s1155269217@scrp-login.econ.cuhk.edu.hk"
$remoteExport = "/data/users/s1155269217/PROJECT_NAME/results/experiments/transfer_archives"
```

**Downloading PC — PowerShell, AI Hub:**

```powershell
$cluster = "tian.xu@137.189.75.110"
$remoteExport = "/hpchome/tian.xu/PROJECT_NAME/results/aihub_experiments/transfer_archives"
```

Select the download folder on the PC receiving the archive.

**Campus PC for the intermediary route — PowerShell:**

```powershell
$transferDir = "D:\data_transfer"
```

**Main PC for a direct download — PowerShell, choose this block instead:**

```powershell
$transferDir = "E:\data_transfer"  # Choose a short local folder on the main PC.
```

**Downloading PC — PowerShell, prepare the folder and filename:**

```powershell
New-Item -ItemType Directory -Path $transferDir -Force | Out-Null
$archiveName = "PASTE_RESULTS_ARCHIVE_NAME.tar.gz"
```

**Downloading PC — PowerShell, transfer the archive and checksum:**

```powershell
@(
    ('lcd "{0}"' -f $transferDir.Replace('\', '/'))
    "cd $remoteExport"
    "get $archiveName"
    "get $archiveName.sha256"
    'bye'
) | sftp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 $cluster
```

Verify the archive using the PowerShell checksum block in Section 4.2 with this results archive's filename and the receiving PC's `$transferDir`. For a direct download, proceed to Section 9.3 on the main PC using the same files and folder. For the intermediary route, upload both verified files from `D:\data_transfer` to Baidu Netdisk using its desktop client.

### 9.3 Download and extract on the main PC

For the intermediary route, create the main-PC transfer folder below, then download the same archive and `.sha256` file from Baidu Netdisk into it, keeping their filenames unchanged. For the direct route, use the files already downloaded there and retain their actual folder. Set the variables, repeat the archive checksum block from Section 4.2 on the main PC, and proceed to extraction only after it passes.

**Main PC — PowerShell, prepare the receiving folder and filename:**

```powershell
$transferDir = "E:\data_transfer"  # Use the actual main-PC download folder.
New-Item -ItemType Directory -Path $transferDir -Force | Out-Null
$archiveName = "PASTE_RESULTS_ARCHIVE_NAME.tar.gz"
```

**Main PC — PowerShell, after downloading and passing the checksum check:**

```powershell
$extractDir = Join-Path $transferDir ("results_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $extractDir -ErrorAction Stop | Out-Null

tar -xzf (Join-Path $transferDir $archiveName) -C $extractDir
if ($LASTEXITCODE -ne 0) { throw "Result extraction failed." }
```

Use a short destination and a fresh extraction directory. Keep the archive intact until the extracted files pass verification. An archive checksum alone cannot detect a tool dropping files during extraction. If copying under a long project path encounters Windows path limits, keep the verified run under the short staging location and analyze it through its absolute path. Project import in Section 9.5 is optional.

### 9.4 Verify the extracted experiment

**Main PC — PowerShell, from the updated project folder:**

```powershell
Set-Location -LiteralPath $projectDir
$run = "PASTE_EXPERIMENT_OR_COMBINED_ID"
$extractedRun = Join-Path $extractDir $run

.\.venv\Scripts\python.exe -B run_experiment.py status $extractedRun --verify
if ($LASTEXITCODE -ne 0) { throw "Extracted bundle verification failed." }
```

Require the intended task coverage, complete reports and matching file hashes. Use the project's verifier against the extracted folder and confirm that it can locate all required files without relying on unavailable server paths. For a combined experiment, verify its required sources too. An archive checksum verifies the transferred package; extracted-file checks verify the usable local copy.

### 9.5 Place verified results in the local project

Analysis can use an extracted experiment's absolute path directly. To retain it under the project, copy the whole folder into the appropriate import location.

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

Use the project's verification command for both ordinary and combined runs. For a replacement download, use a fresh dated import root such as `results\scrp_imports_PASTE_IMPORT_DATE` and explicitly select those paths during analysis. Preserve the previous import until the replacement has passed verification. Keep the archive/checksum as the transfer record.

## 10. Analyze, document and start the next round

Perform the research comparison on the main PC using the verified saved results.

- Check the method, dataset name, input specification, configuration, period or task coverage and observations before comparing results. Comparable cases should retain matching samples, outcomes and evaluation rules where required by the research design.
- Use the intended full-sample, full-coverage runs for the main comparison; report pilots and partial-run diagnostics separately. Label different methods, input specifications and estimation choices clearly.
- Review fitting and parameter-selection behavior and output diagnostics alongside summary scores.
- Generate comparison tables and figures, investigate differences from the research reference, and document choices and unresolved issues.
- Save derived analysis separately from imported run folders, for example under `results/analysis`. Retain the experiment IDs and data/code versions used by each analysis.
- Keep the estimation, selection and evaluation roles fixed. Document revisions motivated by observed results so later comparisons retain their research context.
- Preserve completed results as a reference. Make subsequent method changes locally, repeat the pilot checks and create new experiments.

For a comparison of methods or input specifications, keep the evaluation protocol aligned and allow each case to select its parameters using the designated selection data. Review aggregate and period/task-level results together with diagnostics. Interpret the comparison according to the research question and avoid repeatedly revising a method against the same held-out evaluation results.

An experiment is ready for research analysis after computation, collection, transfer and local integrity checks have all finished. Agreement with a paper, support for a hypothesis or another research conclusion requires the subsequent analysis.

## References

- [SCRP access and transfer](https://scrp.econ.cuhk.edu.hk/guide/account-and-access)
- [SCRP Slurm guide](https://scrp.econ.cuhk.edu.hk/guide/slurm)
- [SCRP storage guide](https://scrp.econ.cuhk.edu.hk/guide/storage)
- [AI Hub official guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide)

Add links to the project's own experiment guide, data preparation notes and configuration reference when adapting this template.
