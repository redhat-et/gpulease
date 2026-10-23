# gpulease

[![ci](https://github.com/redhat-et/gpulease/actions/workflows/ci.yml/badge.svg)](https://github.com/redhat-et/gpulease/actions/workflows/ci.yml)
[![license: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)

GPU leases with **automatic adoption of unmanaged jobs**, for shared single-host
AMD/ROCm boxes.

`gpulease` gives agents and people a polite way to check out 1..N GPUs — and, unlike
cooperative-only locks, it also watches what is actually running and books it. Jobs
launched from virtualenvs, `uv` interpreters, absolute paths, or `nohup` daemons are
adopted automatically via `/proc/<pid>/fdinfo`, so they cannot be double-allocated by
`gpulease` users even if they never call `gpulease` themselves.

## Why

On a shared box, "just use GPUs 0-3" works until someone's agent runs
`/opt/venv/bin/python -m vllm ... &` and nobody remembers which GPUs are free.
Environment-variable locks and PATH shims fail against venvs and daemonized servers.
`gpulease` treats the hardware as the source of truth:

- **leases** for jobs that cooperate: queue politely, get `HIP_VISIBLE_DEVICES`
  exported, release on exit or crash;
- **adoption** for jobs that do not: real VRAM usage per process is mapped to GPU
  indices through DRM fdinfo and PCI addresses, then booked as an `adopted` lease;
- **foreign reservations** for VRAM that belongs to other users;
- **status** that overlays leases with live `rocm-smi` numbers.

Adoption cannot stop two jobs that both bypass `gpulease` from colliding with each
other — that requires device-level isolation (cgroups, containers, or a real
scheduler). It does guarantee that everything launched through `gpulease` avoids
everything else, and it makes unmanaged usage visible.

## Install

Requires Linux, Python 3.10+, and ROCm (`rocm-smi` on `PATH`).

```sh
uv tool install git+https://github.com/redhat-et/gpulease
# or
pipx install git+https://github.com/redhat-et/gpulease
# or from a checkout:
./deploy/install.sh
```

`deploy/install.sh` also installs and enables the `gpulease-watch` systemd user unit.
Without systemd:

```sh
nohup gpulease watch >>"${XDG_STATE_HOME:-$HOME/.local/state}/gpulease/watch.log" 2>&1 &
```

## Quick start

```sh
gpulease pool                       # detected GPUs, pool, reservations
gpulease status                     # leases + live VRAM
gpulease run -n 2 -- python bench.py
gpulease run -n 1 --prefer 6 -- python repro.py
gpulease run --gpu 5,6 -- python tp2.py
```

Long-lived servers:

```sh
eval "$(gpulease checkout -n 4 --ttl 240)"
# launch vllm etc. with the printed HIP_VISIBLE_DEVICES
gpulease release <lease-id>
```

Reserve GPUs for someone else (removed from the pool until unreserved):

```sh
gpulease reserve 0-3 --for alice
gpulease unreserve 0-3
```

## Commands

| Command | Description |
|---|---|
| `run -n K [--prefer SPEC] [--gpu SPEC] [--wait S] -- CMD` | lease K GPUs for the command's lifetime |
| `checkout -n K [--ttl MIN]` | lease GPUs for a long-lived job; prints env exports |
| `release ID` / `renew ID` / `env ID` | manage checkout leases |
| `status [--json]` | leases, adopted/foreign usage, live VRAM |
| `pool` | resolved config |
| `reserve SPEC [--for NAME]` / `unreserve SPEC` | pool exclusions |
| `reap` | drop stale lease records |
| `watch [--once]` | adoption watcher loop |

`--prefer 7,6` orders currently-free GPUs by preference and silently falls back to
the rest of the pool. `--gpu 5,6` requires exactly those GPUs (waits per `--wait`,
preserves order for rank mapping, and infers `-n`).

## Configuration

`$XDG_STATE_HOME/gpulease/config.json` (default
`~/.local/state/gpulease/config.json`). Missing keys fall back to defaults; GPU count
and pool are auto-detected from `rocm-smi` when not set.

```json
{
  "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
  "pool": [4, 5, 6, 7],
  "reserved": {"0-3": "alice"},
  "idle_check": true,
  "idle_threshold_gib": 2.0,
  "ttl_minutes": 240,
  "watch_interval_seconds": 5,
  "adopt_grace_seconds": 30,
  "default_wait_seconds": 3600
}
```

| Key | Meaning |
|---|---|
| `gpus` | all GPU indices (default: detected) |
| `pool` | allocatable indices (default: all non-reserved) |
| `reserved` | `{range: note}` excluded from the pool |
| `idle_check` | treat unleased GPUs above the VRAM threshold as busy |
| `idle_threshold_gib` | busy threshold; idle cards sit near 0.3 GiB |
| `ttl_minutes` | default checkout TTL |
| `watch_interval_seconds` | adoption poll interval |
| `adopt_grace_seconds` | release delay after unmanaged usage drains |
| `default_wait_seconds` | default `--wait` (`-1` = forever) |

Environment overrides: `GPULEASE_STATE_DIR`, `GPULEASE_ROCM_SMI`,
`GPULEASE_PROC_ROOT`, `GPULEASE_OWNER`.

## How adoption works

Every few seconds the watcher:

1. scans `/proc/<pid>/fdinfo/*` for processes owned by you, reading `drm-pdev`
   (PCI BDF) and `drm-memory-vram`;
2. maps BDFs to `rocm-smi` device indices and keeps GPUs above the VRAM threshold
   (HIP opens every render node at init, so open file descriptors are not usage —
   only allocated VRAM is);
3. groups usage by process ancestry into one lease per job, with holder PIDs for
   liveness (`adopted` leases disappear when their PIDs die);
4. flags VRAM not explained by your processes on pool GPUs as a `foreign`
   reservation.

`run` and `checkout` consider leased, adopted, foreign, and (optionally) VRAM-busy
GPUs unavailable, so cooperative jobs never collide with unmanaged ones.

## Agent integration

Paste [`docs/agent-prompt.md`](docs/agent-prompt.md) into whatever instruction
surface your agents honor (AGENTS.md, context files, task templates). With the
watcher running, adoption keeps bypassing agents safe anyway; the prompt turns
collisions into queueing.

## Limitations

- ROCm/AMD only (fdinfo `drm-pdev` + `rocm-smi`). NVIDIA support would need a
  different probe.
- Single-user state directory. Multiple Unix users need a shared state location and
  permissions of their own.
- Whole-GPU leases; no fractional/shared-GPU accounting.
- No preemption: adopted jobs are respected, not killed.
- Adoption relies on VRAM allocation above the threshold; tiny jobs may not be
  attributed. `idle_check` covers most of the gap.
- GPUs are addressed by HIP/`rocm-smi` index, never by render-node number
  (on MI300X/MI355X each GPU exposes multiple `renderD*` nodes for its XCDs).

## Alternatives

| Project | What it does | Notes |
|---|---|---|
| [elektricM/gpuq](https://github.com/elektricM/gpuq) | single-node scheduler: leases, priorities, VRAM sharing, cooperative preemption, AMD+NVIDIA | closest existing tool; Rust, GPL-3.0, cooperative jobs only |
| [lzy248/gpuq](https://github.com/lzy248/gpuq) | per-user opportunistic queue that starts jobs when VRAM frees up | personal backlog queue, not multi-agent arbitration |
| [mdenil/gpu_lock](https://github.com/mdenil/gpu_lock) | cooperative file locks in `/var/lock/gpu`, dead-lock reaping | simple, multi-user, no adoption/status |
| [Lixue9jiu/gpu-locks](https://github.com/Lixue9jiu/gpu-locks) | cooperative locks across containers/hosts plus an audit | agents must cooperate |
| [Slurm](https://slurm.schedmd.com/) | full HPC scheduler with GRES, accounting, cgroups | robust but heavy; manual AMD GRES config; jobs must use `srun`/`sbatch` |
| Kubernetes + AMD device plugin / HAMi | cluster-level GPU scheduling and sharing | overkill for one workstation |

To our knowledge, no existing tool adopts unmanaged processes, which is why `gpulease` exists.

## Development

```sh
python3 -m unittest discover -s tests -v
python3 tests/integration_onbox.sh   # 8-GPU ROCm host; bookkeeping-only checks
```

Unit tests use a fake `rocm-smi`, fake `/proc` trees, and a temporary state
directory; they do not need a GPU.

## Uninstall

```sh
systemctl --user disable --now gpulease-watch
rm ~/.config/systemd/user/gpulease-watch.service
rm ~/.local/bin/gpulease
rm -rf "${XDG_STATE_HOME:-$HOME/.local/state}/gpulease"
```

## License

Apache License 2.0 © 2026 Red Hat, Inc. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
