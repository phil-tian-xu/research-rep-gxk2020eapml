# SCRP computing cluster

## Purpose and scope
Phil is a Finance PhD student at CUHK and uses the Department of Economics' SCRP computing cluster for research workloads exceeding his local resources.
Read this file when planning cluster storage, environments, job submission, monitoring, or migration. General project instructions remain in `AGENTS.md`; replication methods and experiment status belong in `PROJECT_AGENTS.md`.

Follow Phil's explicit instructions over this guidance.

Preserve the research method when moving execution to SCRP. Prefer a simple, reproducible setup. More available hardware does not justify changing models, precision, samples, validation rules, or introducing distributed training.

## Sources and verification
Official documentation, checked on 2026-09-24:
- [SCRP website](https://scrp.econ.cuhk.edu.hk/)
- [Account, access and file transfer](https://scrp.econ.cuhk.edu.hk/guide/account-and-access)
- [Slurm and resource requests](https://scrp.econ.cuhk.edu.hk/guide/slurm)
- [Storage](https://scrp.econ.cuhk.edu.hk/guide/storage)

Consult the official documentation for cluster usage questions. Treat the account output Phil supplied as an observed snapshot, not permanent policy or
guaranteed availability. 

## Access

- Phil already uses MobaXterm on Windows for SSH and SFTP.
- Login hosts: `scrp-login.econ.cuhk.edu.hk` and
  `scrp-login-2.econ.cuhk.edu.hk`.
- The access guide requires campus network access or CUHK VPN for SSH/SFTP.
- MobaXterm's SFTP browser can upload directly to the chosen remote data folder.
- Keep passwords, private keys and authentication tokens out of this repository.

## Account snapshot supplied by Phil

The following values come from terminal output in the conversation provided on 2026-09-24; the original command execution time was not supplied. No live cluster
connection was made to verify them when drafting this file.

Account: `pg`. 
Default QoS: `c32g4`. 
Available QoS: `c32g4`, `c16-long`.

| QoS | Observed resource ceilings | Maximum job duration |
| --- | --- | --- |
| `c32g4` | 128 logical CPUs; 4 GPUs total, including at most 2 labelled `a100`; memory field truncated | 5 days |
| `c16-long` | 16 logical CPUs; 1 GPU; memory field `1600G` | 30 days |

These resource ceilings were shown under `MaxTRESPU` (per user), not as a promise of that allocation for each job. Partition limits also apply. The truncated
default-QoS memory field is unknown; do not interpret it as unlimited.

| Storage                        | Observed quota                  |
| ------------------------------ | ------------------------------- |
| Home `~/`                      | 30 GiB soft / 33 GiB hard       |
| Archive `~/archive`            | 1,024 GiB soft / 1,100 GiB hard |
| Distributed large-data storage | 1,000 GiB hard                  |

The supplied `scrp-quota` output also reported a stale backup mount handle. That message is not evidence of an exhausted quota; if still present, investigate
with SCRP support rather than assuming backup health.
## Storage and transfers
Use `~/large-data` for large datasets and substantial generated outputs. Home storage is suitable for code, environment files and small important results.
The official guide describes home backups; large-data and archive have no automatic backup. Keep independent copies of irreplaceable data and results.

Node-local `/tmp` is temporary space, not the sole location for checkpoints.

Observed layout from Phil's supplied SCRP command output:
```text
~/research-rep-gxk2020eapml/                         repository
~/research-rep-gxk2020eapml/data/processed            symlink to ~/large-data/GKX2020/data/processed
~/large-data/GKX2020/data/processed/                 prepared datasets
~/large-data/GKX2020/results/experiments/             experiment outputs
```

The verified `data/processed` link resolves to `/data/users/s1155269217/GKX2020/data/processed`. Only `data/processed` was linked in that output; project `data/raw` is a separate location and initially did not exist. Check raw inputs separately before server-side preparation. Verify current link targets and existing contents before changing paths; never replace a nonempty directory without preserving its contents. Use the home-relative large-data entry point and verify its resolved path instead of assuming a physical mount path.
## Running computation

This repository includes workloads that need CPU resources and workloads that need GPU resources. Not every task needs both. Follow Phil's request and the
specific workload when selecting CPUs, GPUs, RAM and runtime; do not apply one fixed resource allocation to every experiment.

Login nodes have limited resources. Use them for lightweight preparation, file management and job submission. Run substantial computation on compute nodes
with resources allocated through Slurm.

Use `compute` for interactive work on allocated resources and `sbatch` for unattended runs. Check current account limits and resource availability before
submitting a job, and request resources appropriate to the task.

## Data and result transfers
Phil's preference, clarified on 2026-10-30: choose the transfer route according to file size, connectivity and the current request. Direct uploads from the main PC are acceptable, especially for small files. The campus PC is an optional intermediary. When the intended route is unclear, present both options with commands labeled for the PC on which they run; do not assume that an intermediary is required.

**Campus PC transfer directory: `D:\data_transfer`.** Phil requires this directory for all campus-PC file transfers, including uploads, downloads, datasets, result archives and checksums, especially large files. Create it before downloading and keep any transfer subfolders beneath it. Main-PC commands use the files' actual local location; the campus directory does not determine the main PC's path. Verify that staged files exist before uploading; creating the directory does not populate it.

**Campus PC — PowerShell, prepare the transfer directory:**

```powershell
$transferDir = 'D:\data_transfer'
New-Item -ItemType Directory -Path $transferDir -Force | Out-Null
```

### Transfer routes

1. **Direct: main PC → SCRP.** Upload from the files' actual location on the main PC using SFTP. Prefer this simple route for small inputs when the connection works; no staging copy or Netdisk transfer is required.
2. **Via an intermediary: main PC → Baidu Netdisk → campus PC → SCRP.** This remains useful for large transfers or unreliable direct connections. Download the files into `D:\data_transfer` on the campus PC before uploading them from there. For results, reverse the route: SCRP → campus PC (`D:\data_transfer`) → Baidu Netdisk → main PC. Keep the archive and its checksum together through both transfer stages.

Earlier large transfers from the primary PC repeatedly disconnected, while the campus-PC test was fast and successful (2026-09-24). The cause was not established. That observation supports the intermediary option for unreliable transfers; it does not rule out direct uploads of other files.

For either route, verify the local source files and remote destination before transferring. Verify completeness and, where available, checksums before using the uploaded data. Preserve the original local copies.

Follow Phil's requested transfer route. When recommending a route, measure the actual file/archive size and use the observed connection reliability; predictor count alone does not establish that a result archive is small. Transfer only the requested dataset or experiment. A complete experiment download includes all files inside that one selected experiment folder. See [the research workflow](docs/run_instructions/research_workflow.md#9-transfer-results-back-and-verify-locally) for packaging, checksums, download and extraction.

### Connection and destination
- **Server:** `scrp-login.econ.cuhk.edu.hk`
- **Username:** `s1155269217`
- **Campus PC source/download directory:** `D:\data_transfer`; verify the required files have been downloaded there before uploading.
- **Main PC source/download directory:** the files' actual local folder, set explicitly in the command.
- **Remote project data root:** `/data/users/s1155269217/GKX2020/data/`
- The remote subdirectory depends on the dataset and its purpose. Prepared datasets currently go under `processed/`.

### SFTP File Transfer
Provide separate command blocks for setup, transfer and verification, with each block's purpose, PC and shell stated. The SFTP block can supply its instructions automatically through PowerShell. Prompt for authentication; do not embed passwords. Clearly distinguish local Windows paths from remote SCRP paths.

Direct-upload example, run on the main PC from its actual project data folder. The remote project `data/raw` directory must already exist:

```powershell
$rawDir = 'E:/Workstation/coding_projects/coding_research/paper_replication/research-rep-gxk2020eapml/data/raw'
if ((Test-Path -LiteralPath "$rawDir/macro_predictor.csv" -PathType Leaf) -and
    (Test-Path -LiteralPath "$rawDir/risk_free_rate_fama_french.csv" -PathType Leaf)) {
    @(
        'cd /home/users/s1155269217/research-rep-gxk2020eapml/data/raw'
        ('lcd "{0}"' -f $rawDir)
        'put macro_predictor.csv'
        'put risk_free_rate_fama_french.csv'
        'ls -l'
        'bye'
    ) | sftp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 s1155269217@scrp-login.econ.cuhk.edu.hk
} else {
    throw "Required CSV files are missing from $rawDir on this PC. Use their actual local folder."
}
```

Intermediary-upload example, run on the campus PC after downloading `gkx_dev_p05_seed2020_parquet` to `D:/data_transfer/`; the local folder must be present before running the command:

```powershell
$transferDir = 'D:\data_transfer'
$dataset = 'gkx_dev_p05_seed2020_parquet'
if (-not (Test-Path -LiteralPath (Join-Path $transferDir $dataset) -PathType Container)) {
    throw 'Download the dataset to D:\data_transfer before uploading.'
}

@(
    ('lcd "{0}"' -f $transferDir.Replace('\', '/'))
    'cd /data/users/s1155269217/GKX2020/data/processed'
    "put -R $dataset"
    'bye'
) | sftp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 s1155269217@scrp-login.econ.cuhk.edu.hk
```

The remote parent directory must already exist. If the dataset directory already exists, its `mkdir` error is expected; inspect any other errors. `put -R` uploads recursively and overwrites matching remote filenames—it does not provide incremental synchronization.

For future transfers, select either route and adapt the actual source folder and remote destination. If the route or source PC is uncertain, give both direct and intermediary options instead of fixing one route.

## Running Project on SCRP
Please refer to `docs/Running Python on SCRP` for detailed instructions.
