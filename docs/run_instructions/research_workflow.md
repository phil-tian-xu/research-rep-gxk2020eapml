# Research experiment workflow

This guide follows an experiment from local development to server execution and back to local analysis. The workflow is reusable; the commands use this project's launcher, accounts and datasets as examples. Last reviewed: 2026-10-30.

## Workflow at a glance

**Prepare locally → test locally → synchronize code and data → test on the server → run and monitor full experiments → collect and verify → transfer back → analyze locally.**

| Step                                                                     | Work                                                             | Where                | Ready to proceed when                                         |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------- | -------------------- | ------------------------------------------------------------- |
| [1. Prepare data](#1-prepare-data-on-the-main-pc)                        | Choose the security sample and predictors; prepare that dataset | Main PC              | Dataset, features and source versions are clear               |
| [2. Develop and test](#2-develop-and-test-locally)                       | Modify the method and check small-sample results                 | Main PC              | Training, predictions and diagnostics have been checked       |
| [3. Synchronize code](#3-record-and-synchronize-the-tested-code)         | Commit/push tested changes and pull on the servers               | Main PC and servers  | Each experiment has a known code version                      |
| [4. Transfer inputs](#4-transfer-required-input-data)                    | Main PC → server, directly or via the campus PC                  | Selected PC and server | Required datasets are complete on the target server         |
| [5. Test on the server](#5-run-a-small-server-test)                      | Run a small sample with the intended server setup                | SCRP / AI Hub        | The pilot finishes and its diagnostics are reviewed           |
| [6. Run the full experiment](#6-launch-the-full-experiments)             | Submit each planned model, dataset, config and test period       | SCRP / AI Hub        | Experiment IDs and intended coverage are recorded             |
| [7. Monitor execution](#7-monitor-progress-and-handle-interruptions)     | Check queues, saved progress, logs and failures                  | SCRP / AI Hub        | All planned annual fits are saved                             |
| [8. Collect and verify](#8-collect-and-verify-on-the-server)             | Build pooled reports and complete result bundles                 | SCRP / AI Hub        | Coverage and saved files pass verification                    |
| [9. Bring results home](#9-transfer-results-back-and-verify-locally)     | Server → main PC, directly or via the campus PC                  | Server and selected PCs | Archives and extracted results pass local verification     |
| [10. Analyze and iterate](#10-analyze-document-and-start-the-next-round) | Compare methods, investigate differences and document findings   | Main PC              | Reviewed outputs and the next research questions are recorded |

Monitoring continues throughout server execution. Input transfers are needed when deploying a dataset for the first time or introducing a new data version.

Follow the requested transfer route. Otherwise, choose according to the actual file/archive size and connection reliability: small files can go directly between the main PC and server; the campus PC is an optional intermediary for large files or unreliable direct connections. If the intended route is unclear, present both options with commands labeled for the PC on which they run. Measure the archive before describing it as small; predictor count alone does not determine result size.

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
- Each project uses its own `.venv`. Replace values beginning with `PASTE_` or  `PATH/TO/` before running a block.
- Re-enter the project and reset the relevant variables after opening a new terminal or reconnecting. Initial server checkout/data-path setup is assumed; use the linked site guides when setting up a server for the first time.
- An **experiment ID** is used by `resume`, `status` and `collect`. A **Slurm job ID** is used by `squeue`, `sacct` and `scancel`.
- Examples that name a method create a new experiment. Use its existing ID for subsequent monitoring, resumption and collection.
- Select either `--dataset NAME` or `--sample p05/full` for a new experiment. The examples below use explicit dataset names so the feature specification is visible. Resume and collection use the saved ID without selecting the dataset again.

For detailed site commands, see [SCRP](run_scrp.md), [AI Hub](run_aihub.md) and the [project run guide](../Running%20Experiments.md).

## 1. Prepare data on the main PC

Preserve the raw inputs. Keep raw data, prepared panels and experiment outputs separate. Choose both the security sample and the predictor specification before testing. Use matching prepared datasets locally and on the servers, with consistent dates, features and target definitions.

**Main PC — PowerShell:**

```powershell
$projectDir = "E:\Workstation\coding_projects\coding_research\paper_replication\research-rep-gxk2020eapml"
Set-Location -LiteralPath $projectDir

# Install/update this project's locked environment when needed; the NN extra is optional.
uv sync --locked --extra nn
```

### 1.1 Choose the dataset

| Dataset name | Security sample | Predictor specification |
| --- | --- | --- |
| `gkx_parquet` | Full universe | Original 920 inputs |
| `gkx_dev_p05_seed2020_parquet` | Fixed 5% sample | Original 920 inputs |
| `gkx_compact_full` | Full universe | Compact 176 inputs |
| `gkx_compact_p05` | The same fixed 5% sample | Compact 176 inputs |
| `gkx_expanded_full` | Full universe | Explicit copy of the original 920 inputs |
| `gkx_expanded_p05` | The same fixed 5% sample | Explicit copy of the original 920 inputs |

The compact specification contains 94 characteristics, eight standalone macros and 74 industry indicators. The original/expanded specification contains 94 characteristics, 752 characteristic–macro products and 74 industry indicators. Expanded preparation preserves the original predictor values and order; it adds no standalone macros. These counts describe our current industry vocabulary; preparation preserves the source vocabulary.

`--sample p05` still selects `gkx_dev_p05_seed2020_parquet`, and `--sample full` still selects `gkx_parquet`. A compact full-universe run requires `--dataset gkx_compact_full`. Changing the test years does not change the security sample or predictors. The original full panel can remain the 920-input comparison baseline without creating a second expanded copy; use the explicit expanded name when that is the planned experiment identity.

### 1.2 Prepare only missing datasets

Reuse existing verified panels. For a new source panel, follow [panel preparation](../working_notes/GKX%20Panel%20Preparation.md), [development samples](../working_notes/GKX%20Development%20Samples.md) and [Parquet conversion](../working_notes/Parquet%20Conversion.md). The development sample selects securities once with its fixed seed; keep those securities and their observations consistent across feature specifications.

For the derived datasets, `_p05` uses `gkx_dev_p05_seed2020_parquet`; `_full` uses `gkx_parquet`. Compact preparation also reads `data/raw/macro_predictor.csv` and `data/raw/risk_free_rate_fama_french.csv` through the existing macro helper. Expanded preparation needs only the source Parquet panel. Both preserve observations, targets, identifiers and row positions. Compact macros align to `feature_month`, one month before `return_month`, without additional scaling or security resampling.

**Main PC — PowerShell, choose the dataset to prepare:**

```powershell
$dataset = "gkx_compact_full"
# Other derived choices: gkx_compact_p05, gkx_expanded_p05, gkx_expanded_full.
$datasetDir = Join-Path $projectDir "data\processed\$dataset"

if (Test-Path -LiteralPath $datasetDir) {
    Write-Host "Dataset folder exists. Check its completion metadata before reusing it."
} else {
    .\.venv\Scripts\python.exe -B -m core.data_cleaning.feature_gkx --dataset $dataset
    if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed." }
}
```

Preparation is a separate step; `run_experiment.py` does not build datasets automatically. The builder checks source files, reads back its output and publishes completion metadata last. It refuses existing destinations. Preserve any existing or interrupted build for inspection instead of overwriting it. See [compact feature experiments](../working_notes/Compact%20Feature%20Experiments.md) and the [feature audit](../working_notes/Compact%20Feature%20Audit.md) for the exact definitions and checks.

At the local check on 2026-10-29, the original full/5% panels and corrected compact/expanded 5% panels had completion manifests. The compact and expanded full-universe variants had not been prepared. Recheck the files before the next deployment; this is a dated snapshot.

### 1.3 Check the selected dataset and retain its identity

**Main PC — PowerShell:**

```powershell
$dataset = "gkx_compact_p05"  # Select the panel for the next pilot or transfer.
$datasetDir = Join-Path $projectDir "data\processed\$dataset"
$metadataPath = Join-Path $datasetDir "metadata.json"
$metadata = Get-Content -LiteralPath $metadataPath -Raw -ErrorAction Stop | ConvertFrom-Json
if ($metadata.dataset_name -ne $dataset -or $metadata.status -ne "complete" -or $metadata.format -ne "parquet") {
    throw "The selected dataset does not have matching, complete Parquet metadata."
}
[pscustomobject]@{
    Dataset = $metadata.dataset_name
    Rows = $metadata.n_rows
    Predictors = @($metadata.feature_columns).Count
    MetadataSHA256 = (Get-FileHash -LiteralPath $metadataPath -Algorithm SHA256).Hash
}
```

Record the dataset name, predictor specification and metadata hash with the experiment plan. This is a metadata check; the data reader verifies partition hashes as it reads them. Transfer the entire prepared dataset folder, including its metadata, provenance files and all partitions. Keep the original preparation inputs for reproducibility.

## 2. Develop and test locally

Implement or revise the research method on the main PC. Test the intended predictor specification on its fixed 5% sample and a limited test period before preparing production runs. Review training behavior, validation selection, predictions and evaluation results. Record the research choices and unresolved questions.

**Main PC — PowerShell, from the project folder:**

```powershell
# One NN3 test year on the compact fixed 5% sample, using the local GPU.
$dataset = "gkx_compact_p05"
.\.venv\Scripts\python.exe -B run_experiment.py nn3 --local --dataset $dataset --config gkx_nn3_v1 --years 1987 --cpus 4

.\.venv\Scripts\python.exe -B run_experiment.py list
```

Copy the experiment ID printed by the run:

```powershell
$run = "PASTE_LOCAL_EXPERIMENT_ID"
.\.venv\Scripts\python.exe -B run_experiment.py status $run
.\.venv\Scripts\python.exe -B run_experiment.py collect $run
```

For the 920-input NN3 comparison, repeat the local model command with `gkx_expanded_p05` or the original `gkx_dev_p05_seed2020_parquet`, retaining the chosen model config and years. Each invocation creates its own experiment ID. Both feature specifications tune separately on validation data; using the same config keeps the candidate settings aligned without forcing the same selected parameters.

The launcher supports `pcr`, `pls`, `enet`, `ols3`, `ols`, `ols3-mse`, `glm`, `rf` and `nn1` through `nn10`. NN6-NN10 are the approved deeper, wider research extensions; NN1-NN5 retain the paper's architectures. `--config NAME` selects a versioned research configuration and must match the requested method; dataset and years are selected explicitly by the launcher. Refer to the [configuration reference](../../config/README.md) for defaults.

| Command | Regression benchmark |
| --- | --- |
| `ols` | Squared-error OLS using all declared dataset predictors |
| `ols3-mse` | Squared-error OLS on `mvel1`, `bm`, `mom12m` by default |
| `ols3` | Existing Huber regression on those three characteristics |

**Optional separate local benchmark runs — PowerShell:**

```powershell
.\.venv\Scripts\python.exe -B run_experiment.py ols --local --dataset gkx_compact_p05 --years 1987 --cpus 4
.\.venv\Scripts\python.exe -B run_experiment.py ols3-mse --local --dataset gkx_compact_p05 --years 1987 --cpus 4
```

Review the saved `launch.json` and resolved `research.yaml` for the actual dataset, model and settings. A completed pilot establishes execution and diagnostic evidence; predictive performance and agreement with the paper still require research review.

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
project_dir="$HOME/research-rep-gxk2020eapml"
cd "$project_dir"

git status --short
git pull --ff-only
uv sync --locked --extra nn
git rev-parse HEAD
```

Use the project's required extras; `--extra nn` installs its NN dependency. Install dependencies on the login/head node. Preserve the original checkout when queued/running/resumable experiments depend on it. The compact-feature changes affect execution hashes, so use fresh IDs for new experiments and retain the original code/environment for unfinished historical runs. Git changes do not transfer or prepare research datasets.

## 4. Transfer required input data

**Direct: main PC → SCRP / AI Hub. Optional intermediary: main PC → Netdisk → campus PC → SCRP / AI Hub.**

Use SFTP from the PC that currently holds the source files. For small inputs, such as the macro and risk-free CSV files used by compact preparation, upload directly from the main PC's actual project folder; no archive or intermediary is required. See the [direct and intermediary SFTP examples](../../SCRP_AGENTS.md#sftp-file-transfer). Check that the local files exist and the remote destination is correct before uploading. Use this step only for required files or datasets that the server does not already have in the correct version.

The archive workflow below supports both routes. For direct uploads, keep the archive and checksum on the main PC and use that PC in Sections 4.2–4.3. For the intermediary route, use the Baidu Netdisk desktop client to download both files into `D:\data_transfer` on the campus PC first, then verify and upload there. Earlier disconnections during large direct transfers make the campus PC useful as a fallback; they do not make it mandatory for every transfer.

Prepare each selected panel once on the main PC and transfer that same completed folder to the servers that need it. Transfer the 5% variant for the server pilot, then its full-universe counterpart for production. A complete compact or expanded panel can be fitted without its parent panel or raw macro files on that server. Server-side data preparation is an alternative when needed; it requires the source panel and preparation inputs, and substantial preparation belongs on an allocated compute node.

### 4.1 Package the dataset on the main PC

**Main PC — PowerShell:**

```powershell
$transferDir = "E:\data_transfer"  # Main-PC staging folder; choose another local folder if needed.
New-Item -ItemType Directory -Path $transferDir -Force | Out-Null

# Choose the prepared panel being deployed; repeat for other required panels.
$dataset = "gkx_compact_p05"
# Production: gkx_compact_full. Original baseline: gkx_parquet.
# Explicit expanded baseline: gkx_expanded_p05 or gkx_expanded_full.
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

For a direct upload, use the `.tar.gz` and `.sha256` files already on the main PC. For the intermediary route, upload both to Baidu Netdisk, create `D:\data_transfer` on the campus PC and download both files there. Transfer prepared panels when the server can directly reuse them; transfer raw inputs when server-side preparation requires them. To package raw inputs instead, set `$payload = "raw"` before creating the archive. Keep preparation inputs and finished panel archives clearly labeled. Changing a folder's name does not change the registered dataset identity in its metadata.

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
$remoteData = "/data/users/s1155269217/GKX2020/data"
```

**Uploading PC — PowerShell, AI Hub profile:**

```powershell
$cluster = "tian.xu@137.189.75.110"
$remoteData = "/hpchome/tian.xu/research-rep-gxk2020eapml/data"
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
data_root="/data/users/s1155269217/GKX2020/data"
archive="PASTE_INPUT_ARCHIVE_NAME.tar.gz"
dataset="gkx_compact_p05"
payload="processed/$dataset"

cd "$data_root"
if [ -e "$data_root/$payload" ]; then
  printf 'Dataset destination exists. Verify/use the existing version or choose a new versioned destination.\n'
elif sha256sum -c "$archive.sha256"; then
  tar -xzf "$archive" -C "$data_root"
fi

# For a prepared panel, compare this with the main-PC metadata hash from Section 1.3.
sha256sum "$data_root/$payload/metadata.json"
```

Set `payload` to the same relative path packaged in Section 4.1. Check that the expected metadata/part files are present and use the same data version as the local pilot. The normal model reader also validates the dataset's saved metadata and partition hashes as it reads them. Preserve existing raw/prepared datasets. For raw-only archives, omit the prepared-panel metadata hash command. The registered paths are `data/processed/DATASET_NAME` within the project, possibly through the existing server data symlink; verify that the upload destination and project reader refer to the same location.

## 5. Run a small server test

Run a pilot on each server with the intended dataset family and model configuration before its full experiment. Match the local pilot's 5% dataset, config and years. Confirm data identity, training diagnostics and prediction consistency; investigate material cross-device differences.

**SCRP — Bash, project folder:**

```bash
dataset="gkx_compact_p05"
.venv/bin/python -B run_experiment.py nn3 --scrp \
  --dataset "$dataset" --config gkx_nn3_v1 --years 1987 \
  --gpu-type a100 --gpus 1 --parallel 1 \
  --cpus 4 --mem 8G --time 00:30:00
```

**AI Hub — Bash, project folder:**

```bash
dataset="gkx_compact_p05"
.venv/bin/python -B run_experiment.py nn3 --aihub \
  --dataset "$dataset" --config gkx_nn3_v1 --years 1987 \
  --partition gpuh --gpu-type any --gpus 1 --parallel 1 \
  --cpus 4 --mem 8G --time 00:30:00
```

`gpuh` is the partition observed on this account; check current `sinfo` output. See [monitoring](#7-monitor-progress-and-handle-interruptions) and [collection](#8-collect-and-verify-on-the-server) for the pilot's next steps. Save its experiment ID and review its results before launching the full sample. For another predictor specification, change the dataset to its matching 5% name and repeat the pilot. If AI Hub will run NN2, pilot NN2 with `--config gkx_nn2_v1` before its NN2 production run. Pilot RAM/time requests remain initial examples to adjust from observed requirements.

Adding `--dry-run` prepares an experiment without submitting it. Submit that prepared experiment through its printed ID using `resume`; repeating the model command creates a different experiment.

## 6. Launch the full experiments

Record the model, dataset name, feature specification, config, intended period, code version, server and experiment ID. Prepare and transfer the selected full-universe panel first. Keep pilots separate from production results. Resource requests are examples to adjust after the pilot and current account checks.

For example, the compact NN3 production case is `nn3` + `gkx_compact_full` + `gkx_nn3_v1` + `1987:2016`. Its 920-input comparison is a separate case using `gkx_parquet` or `gkx_expanded_full`. A different dataset/config starts a new experiment; it is not a change to an existing ID.

### 6.1 SCRP example

**SCRP — Bash:**

```bash
qos
sinfo -N -O NodeList:20,Partition:12,StateComplete:35,Gres:25,GresUsed:50

dataset="gkx_compact_full"
.venv/bin/python -B run_experiment.py nn3 --scrp \
  --dataset "$dataset" --config gkx_nn3_v1 --years 1987:2016 \
  --gpu-type rtx3090 --gpus 1 --parallel 2 \
  --cpus 8 --mem 96G --time 12:00:00
```

Use `--gpu-type a100` when requesting A100 tasks. CPU methods omit the GPU flags and use appropriate CPU/RAM allocations. Check the [SCRP resource guide](https://scrp.econ.cuhk.edu.hk/guide/slurm) and live account output when changing requests.

### 6.2 AI Hub example

**AI Hub — Bash:**

```bash
sinfo -N -O NodeList:20,Partition:12,StateComplete:35,Gres:25,GresUsed:50

dataset="gkx_compact_full"
.venv/bin/python -B run_experiment.py nn2 --aihub \
  --dataset "$dataset" --config gkx_nn2_v1 --years 1987:2016 \
  --partition gpuh --gpu-type any --gpus 1 --parallel 3 \
  --cpus 8 --mem 80G --time 12:00:00
```

Here `--gpus 1` is per annual task; `--parallel 3` allows three simultaneous tasks. RAM and CPU requests are also per task: this example totals at most 240G and 24 CPUs. Check other active work and the [AI Hub resource guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide#resource-quota) before increasing concurrency or allocations. The launcher also enforces its recorded student resource budget.

### 6.3 Split runs and combined IDs

To divide one model/dataset/config case across GPU types and year blocks, prepare sources with their own resources and disjoint years using `--dry-run`. Both sources must use the same dataset name, compatible data/features, model configuration and execution environment. For example:

**SCRP — Bash, prepare two sources without submitting:**

```bash
dataset="gkx_compact_full"

.venv/bin/python -B run_experiment.py nn3 --scrp \
  --dataset "$dataset" --config gkx_nn3_v1 --years 1987:2003 \
  --gpu-type rtx3090 --gpus 1 --parallel 2 \
  --cpus 8 --mem 96G --time 12:00:00 --dry-run

.venv/bin/python -B run_experiment.py nn3 --scrp \
  --dataset "$dataset" --config gkx_nn3_v1 --years 2004:2016 \
  --gpu-type a100 --gpus 1 --parallel 2 \
  --cpus 8 --mem 96G --time 12:00:00 --dry-run
```

Register the two printed source IDs and submit them through one combined ID:

**SCRP — Bash:**

```bash
early="PASTE_EARLY_SOURCE_ID"
late="PASTE_LATE_SOURCE_ID"
.venv/bin/python -B run_experiment.py combine "$early" "$late" --years 1987:2016

run="PASTE_COMBINED_ID"
.venv/bin/python -B run_experiment.py resume "$run"
```

Use the combined ID for later status and collection. Each source retains its resource settings. On AI Hub, add `--aihub` to combine/resume and ID lookup.

Compact and expanded runs are separate research cases and must not be combined with each other. The original and explicit expanded dataset names also remain distinct experiment identities even though their predictor values match. Create a separate combined ID for each case that is split across years.

The current launcher combines sources from the same backend with compatible settings/environments and complete, non-overlapping coverage. It does not combine SCRP and AI Hub source jobs into one execution ID. Compare their completed model results together during local research analysis. Detailed preparation examples are in the site command references linked above.

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

Pending jobs are waiting. An empty queue does not establish successful completion. If accounting is unavailable, use `status` without `--refresh` and inspect logs. `last_action=submitted` records a past action; saved-year counts show the files currently present. With explicit dataset names, the listing's `sample` field may contain the dataset name rather than `p05` or `full`. Use the saved `launch.json` dataset/config fields and `research.yaml` to identify the specification; do not infer it solely from `30/30` or the model name.

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

Completed annual fits are retained. If the research specification changes, prepare a new experiment. To deliberately cancel a job with an incorrect request:

```bash
scancel "$job"
```

## 8. Collect and verify on the server

After all planned annual tasks have finished, collect the run using its existing ID. Collection builds pooled reports from saved fits. For this replication, the full-period comparison requires full-universe results for all 1987-2016 test years. Collect each model/dataset/config case separately. Use its combined ID when its year blocks were split. Collection pools compatible years within that case; it does not merge compact/expanded results into one fitted model. Later comparison tables can display those cases together with explicit labels.

**SCRP — Bash:**

```bash
.venv/bin/python -B run_experiment.py collect "$run"
```

**AI Hub — Bash:**

```bash
.venv/bin/python -B run_experiment.py collect "$run" --aihub
```

For a combined ID, collection stores full sub-experiment snapshots, including saved models, forecasts, diagnostics, settings and logs, inside that one combined folder. Download that complete folder. Original sub-runs remain preserved.

**Verify a collected combined bundle before packaging — Bash:**

```bash
# SCRP; add --aihub for an AI Hub ID.
.venv/bin/python -B run_experiment.py status "$run" --verify
```

For older/incomplete combined bundles, repair from the intact originals on the server, then verify all embedded files:

```bash
# SCRP; add --aihub to each command for an AI Hub ID.
.venv/bin/python -B run_experiment.py collect "$run" --repair-sources
.venv/bin/python -B run_experiment.py status "$run" --verify
```

Repair requires the recorded inventory and matching saved bytes; it does not retrain models or replace results with a different fit. New/repaired bundles use short `sources/00`, `sources/01` directories. Allow temporary space for a second embedded bundle during repair.

| Check | Meaning |
| --- | --- |
| `saved=30/30` | All planned annual checkpoint files are present for 1987-2016 |
| `reports=ready` | Aggregate reports are available and combined source presence checks pass |
| `sources=present` | Combined source file inventory was checked |
| `sources=verified` | Every embedded source file also passed its SHA-256 check |
| `sources=incomplete` / `reports=incomplete` | Restore the damaged combined bundle before using it |

`sources` verification applies to combined bundles; ordinary single experiments show `sources=n/a`. File integrity and job completion are distinct from research validation. Review scores and training diagnostics as part of analysis.

To inspect the collected reports:

```bash
exp="/FULL/PATH/TO/COLLECTED_EXPERIMENT"
cat "$exp/annual/combined/table1_all.csv"
head -n 15 "$exp/annual/combined/annual_metrics.csv"
# Combined NN diagnostics:
head -n 15 "$exp/annual/combined/nn_training_summary.csv"
```

Keep the dataset identifier visible when recording these scores. A completed 176-input experiment is a compact-specification result; the 920-input specification is the input baseline for comparison with the paper. Inspect `research.yaml` and source manifests when confirming a combined run's specification.

## 9. Transfer results back and verify locally

**Direct: SCRP / AI Hub → main PC. Optional intermediary: SCRP / AI Hub → campus PC → Netdisk → main PC.**

Create fresh archives after the final collection/repair. Earlier archives retain their earlier contents. Follow the requested route; when recommending one, check the archive's actual size and connection reliability. For the campus route, download into `D:\data_transfer` on the campus PC, then use Baidu Netdisk to transfer the archive and checksum to the main PC. Keep both files together and verify the checksum on each receiving PC before extraction.

### 9.1 Package one complete experiment on its server

Select only the experiment requested for analysis. A complete experiment means all files inside that one experiment folder, including fits, forecasts, settings, diagnostics and logs. Record its ID before packaging; repeat for additional experiments only when they are requested.

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
  ls -lh "$archive" "$archive.sha256"
fi
```

Record the archive filename, export directory and displayed size. For combined runs, include their entire `sources/` tree. Raw and prepared research datasets remain separate dependencies. Retain the exact dataset versions on the main PC for prediction restoration and later analyses that read predictors. Some analyses also need original audit/raw inputs, such as lagged size for the largest/smallest-stock groups. A result bundle does not embed those shared datasets. Different predictor specifications should retain their own result archives and experiment IDs.

### 9.2 Download to the selected PC

Choose the matching server/export profile. Run on the main PC for a direct download, or on the campus PC for the intermediary route.

**Downloading PC — PowerShell, SCRP:**

```powershell
$cluster = "s1155269217@scrp-login.econ.cuhk.edu.hk"
$remoteExport = "/data/users/s1155269217/GKX2020/results/experiments/transfer_archives"
```

**Downloading PC — PowerShell, AI Hub:**

```powershell
$cluster = "tian.xu@137.189.75.110"
$remoteExport = "/hpchome/tian.xu/research-rep-gxk2020eapml/results/aihub_experiments/transfer_archives"
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

Use a short destination and a fresh extraction directory. Keep the archive intact until the extracted files pass verification. An archive checksum alone cannot detect a tool dropping files during extraction. Dataset-based experiment IDs can be longer than the first-round IDs. If copying under the long project path encounters Windows path limits, keep the verified run under the short staging location and analyze it through its absolute path. Project import in Section 9.5 is optional.

### 9.4 Verify the extracted combined experiment

**Main PC — PowerShell, from the updated project folder:**

```powershell
Set-Location -LiteralPath $projectDir
$run = "PASTE_COMBINED_ID"
$extractedRun = Join-Path $extractDir $run

.\.venv\Scripts\python.exe -B run_experiment.py status $extractedRun --verify
if ($LASTEXITCODE -ne 0) { throw "Extracted bundle verification failed." }
```

For a full-period combined run, require `saved=30/30`, `reports=ready` and `sources=verified`. The command verifies extracted files without the original server directories. Use `status` on an ordinary single experiment to check saved coverage/report presence; its archive checksum verifies the transferred package.

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

The example is for combined runs. For a replacement download, use a fresh dated import root such as `results\scrp_imports_20261029` and explicitly select those paths during analysis. Preserve the previous import until the replacement has passed verification. Keep the archive/checksum as the transfer record.

## 10. Analyze, document and start the next round

Perform the research comparison on the main PC using the verified saved results.

- Check model, dataset name, predictor specification, config, period coverage and observations before comparing results. Compact and expanded cases should retain matching security samples, targets and chronology.
- Use full-universe, full-period runs for the main 1987-2016 comparison; report pilots or partial-period diagnostics separately. Label compact 176-input results and original/expanded 920-input results separately, and distinguish squared-error OLS/OLS-3 from the Huber versions.
- Review training/validation behavior and prediction diagnostics alongside scores.
- Generate comparison tables and figures, investigate differences from the paper, and document research choices and unresolved issues.
- Save derived analysis separately from imported run folders, for example under `results/replication_comparison`. Retain the experiment IDs and data/code versions used by each analysis.
- Keep the training, validation and test roles fixed. Document revisions motivated by observed results so later comparisons retain their research context.
- Preserve the completed first-round results as a reference. Make subsequent method changes locally, repeat the pilot checks and create new experiments.

For a compact-versus-expanded NN comparison, keep the model settings and evaluation protocol aligned and allow each case to select its parameters on validation data. Review pooled and annual out-of-sample scores together with training diagnostics. A performance comparison measures the predictive effect of the input specification; it does not by itself demonstrate recovery of the paper's interaction structure. Preserve the recorded macro-timing assumptions and avoid revising the model repeatedly against the same test results.

An experiment is ready for research analysis after computation, collection, transfer and local integrity checks have all finished. Agreement with the paper is a separate conclusion supported by the analysis.

## References

- [SCRP command reference](run_scrp.md)
- [AI Hub command reference](run_aihub.md)
- [Project experiment guide](../Running%20Experiments.md)
- [Dataset definitions and preparation](../working_notes/Compact%20Feature%20Experiments.md)
- [Compact feature audit](../working_notes/Compact%20Feature%20Audit.md)
- [Model configuration reference](../../config/README.md)
- [SCRP access and transfer](https://scrp.econ.cuhk.edu.hk/guide/account-and-access)
- [SCRP Slurm guide](https://scrp.econ.cuhk.edu.hk/guide/slurm)
- [SCRP storage guide](https://scrp.econ.cuhk.edu.hk/guide/storage)
- [AI Hub official guide](https://github.com/CUHK-Business-School-AI-Hub/ai_hub_HPC_user_guide)
