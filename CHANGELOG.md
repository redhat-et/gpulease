# Changelog

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
