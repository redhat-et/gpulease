# Validating NVIDIA support

`tests/validate_nvidia.sh` runs hardware checks for the NVIDIA path. It is generic,
uses a temporary gpulease state directory, and never needs root except for one
optional foreign-process check.

## Prerequisites

- Linux with an NVIDIA driver and `nvidia-smi` on `PATH`
- Python 3.10+; `nvcc` is recommended for the CUDA probe (otherwise CUDA-specific
  checks are skipped)
- A free GPU; the script allocates a few GiB and briefly holds the device
- Optional: passwordless `sudo` for the root-owned/foreign test

## Run

From the repo root:

```sh
GPULEASE="$PWD/gpulease.py" tests/validate_nvidia.sh
# test a specific card:
TARGET_GPU=1 GPULEASE="$PWD/gpulease.py" tests/validate_nvidia.sh
```

Useful overrides: `PYTHON`, `ALLOC_MIB` (default 4096), `HOLD_SECS` (default 20).

The script creates a temporary state directory (`GPULEASE_STATE_DIR`) and a
short-grace config, so it does not touch real leases or a shared `/var/lib/gpulease`.

## What it checks

- `nvidia-smi` discovery and GPU/UUID/PCI information
- `vendor=nvidia` and `occupancy_mode` selection
- `run` environment: `CUDA_VISIBLE_DEVICES` + `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- `checkout`/`release` lifecycle
- CUDA index ↔ PCI mapping using a small compiled CUDA probe
- VRAM threshold behavior (VRAM mode only)
- Adoption of an own-uid unmanaged CUDA process
- Foreign booking for a root-owned process, and cleanup
- Claims blocking a `--gpu` request and `--ignore-claims --reason` break-glass

## Expected results

- RTX PRO / discrete NVIDIA: `occupancy=vram`, card memory fields populated.
- GB10 / unified memory: `occupancy=process`, memory fields may be `N/A`; the script
  uses compute-process presence and skips VRAM-specific assertions.
- MIG must be disabled; gpulease refuses to run otherwise.

Exit code is non-zero if any check fails. Do not run this on a shared GPU while
someone else is using it: the script intentionally allocates memory.
