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
