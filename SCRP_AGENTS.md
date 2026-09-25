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

Proposed layout, not yet confirmed as configured:
```text
~/research-rep-gxk2020eapml/                 repository
~/large-data/research-rep-gxk2020eapml/data/ actual datasets
~/research-rep-gxk2020eapml/data            symlink to that data directory
```

This preserves the repository's `data/processed/...` paths. Large results can use the same arrangement for `results/` if needed. Verify the actual targets
and existing contents before creating links; never replace a nonempty directory without preserving its contents. Use the home-relative large-data entry point and verify its resolved path instead of assuming a physical mount path.
## Running computation

This repository includes workloads that need CPU resources and workloads that need GPU resources. Not every task needs both. Follow Phil's request and the
specific workload when selecting CPUs, GPUs, RAM and runtime; do not apply one fixed resource allocation to every experiment.

Login nodes have limited resources. Use them for lightweight preparation, file management and job submission. Run substantial computation on compute nodes
with resources allocated through Slurm.

Use `compute` for interactive work on allocated resources and `sbatch` for unattended runs. Check current account limits and resource availability before
submitting a job, and request resources appropriate to the task.

## Data transfer to SCRP
Agreed on 2026-09-24: use a separate **PC on the CUHK campus network as the intermediary for data uploads**. Transfers from the primary PC repeatedly disconnected; the campus-PC test was fast and successful. The exact cause of the earlier connection resets was not established.
### Standard workflow
1. Transfer data from the primary PC to the campus PC using **Baidu Netdisk**.
2. Place the downloaded files or directories under **`D:/data_transfer/`** on the campus PC.
3. Upload from the campus PC to SCRP using **SFTP**, the preferred transfer method.
4. Verify completeness and, where available, checksums before using the uploaded data. Preserve the original local copies.

### Connection and destination
- **Server:** `scrp-login.econ.cuhk.edu.hk`
- **Username:** `s1155269217`
- **Local staging directory:** `D:/data_transfer/`
- **Remote project data root:** `/data/users/s1155269217/GKX2020/data/`
- The remote subdirectory depends on the dataset and its purpose. Prepared datasets currently go under `processed/`.

### SFTP File Transfer
Prefer a **single PowerShell command run on the campus PC**, supplying the SFTP instructions automatically. Prompt for authentication; do not embed passwords.

Example: upload `D:/data_transfer/gkx_dev_p05_seed2020_parquet` into the remote `processed/` directory:

```
@('cd /data/users/s1155269217/GKX2020/data/processed', 'put -R D:/data_transfer/gkx_dev_p05_seed2020_parquet', 'bye') | sftp -o ServerAliveInterval=30 -o ServerAliveCountMax=6 s1155269217@scrp-login.econ.cuhk.edu.hk
```

The remote parent directory must already exist. If the dataset directory already exists, its `mkdir` error is expected; inspect any other errors. `put -R` uploads recursively and overwrites matching remote filenames—it does not provide incremental synchronization.

For future transfers, adapt the source folder and remote destination while retaining this campus-PC workflow.

## Running Project on SCRP
Please refer to `docs/Running Python on SCRP` for detailed instructions.
