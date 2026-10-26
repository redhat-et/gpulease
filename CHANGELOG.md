# Changelog

## 0.2.0

- Shared multi-user state: `/var/lib/gpulease` with group + ACL access, discovered
  via `.shared`; per-user watchers write into the shared lease table.
- Owner-scoped claims (`claim`/`unclaim`): claimed GPUs are excluded from other
  users' pools and hard-blocked for their `--gpu`/`--prefer` requests.
- Break-glass `--ignore-claims --reason TEXT` for run/checkout; respects active
  leases/VRAM, records `claims_ignored` + reason, and appends to `audit.log`.
- Leases carry uid; `release`/`renew` are owner-only (root override).
- Single shared `foreign` record, coverage-aware across all users' leases.
- Per-user watcher logs; `status`/`pool` show claims and the caller's effective pool.
- `deploy/install-shared.sh`, multi-user README section, tests for shared/claims.

## 0.1.0

Initial public release.

- GPU leases for `1..N` GPUs with crash-safe `flock` and PID-liveness reaping.
- `gpulease run` wrapper and `checkout`/`release`/`renew` leases for long-lived servers.
- Hard `--gpu` and soft `--prefer` GPU selection.
- Adoption watcher: books unmanaged GPU usage from `/proc/<pid>/fdinfo`
  (`drm-pdev` + `drm-memory-vram`), grouped by process ancestry.
- Foreign reservations for VRAM owned by other users.
- Pool reservations (`reserve`/`unreserve`), idle VRAM checks, `status` with JSON output.
- systemd user unit for the watcher, install script, and agent prompt snippet.
