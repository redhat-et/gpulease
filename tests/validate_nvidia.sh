#!/usr/bin/env bash
# Copyright 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Validate gpulease NVIDIA support on a real GPU host.
# Generic: no cloud/account assumptions, no secrets.
#
# Usage:
#   GPULEASE=/path/to/gpulease.py tests/validate_nvidia.sh
#   TARGET_GPU=1 tests/validate_nvidia.sh
#
# Environment:
#   GPULEASE       path to gpulease.py (default: repo gpulease.py)
#   PYTHON         interpreter (default: python3)
#   TARGET_GPU     GPU index to test (default: 0)
#   ALLOC_MIB      allocation for busy-workload tests (default: 4096)
#   HOLD_SECS      how long workloads hold the GPU (default: 20)

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPULEASE="${GPULEASE:-$ROOT/gpulease.py}"
PYTHON="${PYTHON:-python3}"
TARGET_GPU="${TARGET_GPU:-0}"
ALLOC_MIB="${ALLOC_MIB:-4096}"
HOLD_SECS="${HOLD_SECS:-20}"

STATE_DIR="$(mktemp -d /tmp/gpulease-nvidia-state.XXXXXX)"
WORK="$(mktemp -d /tmp/gpulease-nvidia-work.XXXXXX)"
export GPULEASE_STATE_DIR="$STATE_DIR"
FAILS=0
PIDS=()
CUDA_PROBE=""

log() { printf '\n== %s ==\n' "$*"; }
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; FAILS=$((FAILS + 1)); }
gq() { "$PYTHON" "$GPULEASE" "$@"; }

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    if [ -n "$CUDA_PROBE" ]; then
        sudo -n pkill -f "$CUDA_PROBE" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    rm -rf "$STATE_DIR" "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$STATE_DIR"
cat > "$STATE_DIR/config.json" <<'JSON'
{"occupancy_mode": "auto", "adopt_grace_seconds": 2, "idle_threshold_gib": 2.0}
JSON

assert_json() {
    local file="$1"
    local expr="$2"
    if "$PYTHON" - "$file" "$expr" "$TARGET_GPU" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
expr = sys.argv[2]
target = int(sys.argv[3])

def gpu_state(gpu):
    for row in data["gpus"]:
        if row["gpu"] == gpu:
            return row["state"]
    raise AssertionError(f"gpu {gpu} missing")

def adopted_with_pid(pid):
    return any(
        lease.get("type") == "adopted"
        and pid in lease.get("pids", [])
        and target in lease.get("gpus", [])
        for lease in data.get("leases", [])
    )

def foreign_on(gpu):
    return any(
        lease.get("type") == "foreign" and gpu in lease.get("gpus", [])
        for lease in data.get("leases", [])
    )

scope = {
    "vendor": data.get("vendor"),
    "occupancy": data.get("occupancy_mode"),
    "gpu_count": len(data.get("gpus", [])),
    "state": gpu_state(target) if any(r["gpu"] == target for r in data["gpus"]) else None,
    "adopted": adopted_with_pid,
    "foreign": foreign_on,
}
if not eval(expr):
    raise AssertionError(f"assertion failed: {expr}\nscope={scope}")
PY
    then
        pass "$expr"
    else
        fail "$expr"
    fi
}

log "environment"
"$PYTHON" "$GPULEASE" --version || fail "gpulease --version"
nvidia-smi -L || fail "nvidia-smi -L"
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used,mig.mode.current \
    --format=csv,noheader,nounits || fail "GPU query"
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory \
    --format=csv,noheader,nounits || true

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
if [ "$TARGET_GPU" -ge "$GPU_COUNT" ]; then
    echo "FAIL: TARGET_GPU=$TARGET_GPU but only $GPU_COUNT GPU(s) found"
    exit 1
fi

if command -v nvcc >/dev/null 2>&1; then
    cat > "$WORK/probe.c" <<'C'
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char **argv) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { printf("cudaGetDevice failed\n"); return 1; }
    char pci[64] = "";
    cudaDeviceGetPCIBusId(pci, sizeof(pci), dev);
    size_t mib = argc > 1 ? strtoull(argv[1], NULL, 10) : 1024;
    void *p = NULL;
    if (cudaMalloc(&p, mib * 1024 * 1024) != cudaSuccess) {
        printf("cudaMalloc failed\n"); return 2;
    }
    cudaMemset(p, 1, mib * 1024 * 1024);
    cudaDeviceSynchronize();
    printf("pci=%s visible=%s mib=%zu\n", pci,
           getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "", mib);
    fflush(stdout);
    if (argc > 2) sleep(atoi(argv[2]));
    cudaFree(p);
    return 0;
}
C
    if nvcc -O2 -o "$WORK/cudaprobe" "$WORK/probe.c" >/dev/null 2>&1; then
        CUDA_PROBE="$WORK/cudaprobe"
        pass "compiled CUDA probe"
    else
        echo "WARN: nvcc present but CUDA probe failed to compile; CUDA-specific checks skipped"
    fi
else
    echo "WARN: nvcc not found; CUDA-specific checks skipped"
fi

log "gpulease detection"
gq status --json > "$WORK/status.json" || fail "status --json"
assert_json "$WORK/status.json" 'data["vendor"] == "nvidia"'
assert_json "$WORK/status.json" 'data["occupancy_mode"] in ("vram", "process")'
assert_json "$WORK/status.json" 'target in [r["gpu"] for r in data["gpus"]]'

log "basic run/checkout"
OUT=$(gq run -n 1 --gpu "$TARGET_GPU" --wait 0 -- \
    sh -c 'echo visible=$CUDA_VISIBLE_DEVICES order=$CUDA_DEVICE_ORDER gpus=$GPULEASE_GPUS' \
    2>"$WORK/run.err") || fail "run -- true"
echo "$OUT"
case "$OUT" in
    *"visible=$TARGET_GPU"*) pass "CUDA_VISIBLE_DEVICES" ;;
    *) fail "CUDA_VISIBLE_DEVICES mismatch" ;;
esac
case "$OUT" in
    *"order=PCI_BUS_ID"*) pass "CUDA_DEVICE_ORDER" ;;
    *) fail "CUDA_DEVICE_ORDER mismatch" ;;
esac

OUT=$(gq checkout -n 1 --gpu "$TARGET_GPU" --ttl 1 --wait 0 2>"$WORK/checkout.err") || fail "checkout"
if eval "$OUT"; then
    [ -n "${GPULEASE_LEASE_ID:-}" ] && pass "checkout lease id" || fail "checkout lease id"
    gq release "$GPULEASE_LEASE_ID" || fail "checkout release"
    unset GPULEASE_LEASE_ID GPULEASE_GPUS CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER
else
    fail "checkout exports"
fi

if [ -n "$CUDA_PROBE" ]; then
    log "CUDA index/PCI mapping"
    OUT=$(gq run -n 1 --gpu "$TARGET_GPU" --wait 0 -- "$CUDA_PROBE" 256 0 2>"$WORK/map.err")
    RC=$?
    echo "$OUT"
    if [ "$RC" -ne 0 ]; then
        fail "CUDA probe under run (rc=$RC)"
    else
        PCI=$(echo "$OUT" | sed -n 's/.*pci=\([^ ]*\).*/\1/p')
        SMI_PCI=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader,nounits | sed -n "$((TARGET_GPU + 1))p")
        NORM_CUDA=$("$PYTHON" -c 'import re,sys; s=sys.argv[1].lower(); m=re.match(r"([0-9a-f]+):([0-9a-f]+):([0-9a-f]+)\.([0-9a-f]+)",s); print(f"{int(m.group(1),16):04x}:{int(m.group(2),16):02x}:{int(m.group(3),16):02x}.{m.group(4)}" if m else s)' "$PCI")
        NORM_SMI=$("$PYTHON" -c 'import re,sys; s=sys.argv[1].lower(); m=re.match(r"([0-9a-f]+):([0-9a-f]+):([0-9a-f]+)\.([0-9a-f]+)",s); print(f"{int(m.group(1),16):04x}:{int(m.group(2),16):02x}:{int(m.group(3),16):02x}.{m.group(4)}" if m else s)' "$SMI_PCI")
        if [ "$NORM_CUDA" = "$NORM_SMI" ]; then
            pass "CUDA index $TARGET_GPU maps to PCI $NORM_SMI"
        else
            fail "CUDA PCI $NORM_CUDA != nvidia-smi PCI $NORM_SMI"
        fi
    fi

    log "VRAM threshold"
    if [ "$(gq status --json | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["occupancy_mode"])')" = "vram" ]; then
        rm -f "$STATE_DIR"/leases/*.json 2>/dev/null || true
        CUDA_VISIBLE_DEVICES="$TARGET_GPU" "$CUDA_PROBE" "$ALLOC_MIB" 10 >"$WORK/big.out" 2>&1 & BIG=$!
        PIDS+=("$BIG")
        sleep 3
        gq status --json > "$WORK/status_big.json"
        assert_json "$WORK/status_big.json" 'gpu_state(target) in ("busy", "reserved-busy")'
        kill "$BIG" 2>/dev/null || true
        wait "$BIG" 2>/dev/null || true
        sleep 5
        gq status --json > "$WORK/status_after_big.json"
        assert_json "$WORK/status_after_big.json" 'gpu_state(target) in ("free", "claimed")'
    else
        echo "SKIP: VRAM threshold (occupancy_mode=process)"
    fi

    log "adoption of own unmanaged job"
    rm -f "$STATE_DIR"/leases/*.json 2>/dev/null || true
    CUDA_VISIBLE_DEVICES="$TARGET_GPU" "$CUDA_PROBE" 1024 "$HOLD_SECS" >"$WORK/own.out" 2>&1 & OWN=$!
    PIDS+=("$OWN")
    sleep 3
    gq watch --once >/dev/null 2>&1
    gq status --json > "$WORK/status_own.json"
    assert_json "$WORK/status_own.json" "adopted_with_pid($OWN)"
    kill "$OWN" 2>/dev/null || true
    wait "$OWN" 2>/dev/null || true
    sleep 3
    gq watch --once >/dev/null 2>&1
    gq status --json > "$WORK/status_own_after.json"
    assert_json "$WORK/status_own_after.json" "not adopted_with_pid($OWN)"

    log "foreign booking for root-owned job"
    if sudo -n true 2>/dev/null; then
        rm -f "$STATE_DIR"/leases/*.json 2>/dev/null || true
        sudo -n env CUDA_VISIBLE_DEVICES="$TARGET_GPU" "$CUDA_PROBE" "$ALLOC_MIB" "$HOLD_SECS" \
            >"$WORK/root.out" 2>&1 &
        sleep 3
        gq watch --once >/dev/null 2>&1
        gq status --json > "$WORK/status_root.json"
        assert_json "$WORK/status_root.json" "foreign_on(target)"
        sudo -n pkill -f "$CUDA_PROBE" 2>/dev/null || true
        sleep 3
        gq watch --once >/dev/null 2>&1
        gq status --json > "$WORK/status_root_after.json"
        assert_json "$WORK/status_root_after.json" "not foreign_on(target)"
    else
        echo "SKIP: root-owned foreign test (sudo unavailable)"
    fi
else
    echo "SKIP: CUDA-specific tests (no usable CUDA probe)"
fi

log "claims / break-glass"
rm -f "$STATE_DIR"/leases/*.json 2>/dev/null || true
gq claim "$TARGET_GPU" --for root >/dev/null 2>&1 || fail "claim"
gq run -n 1 --gpu "$TARGET_GPU" --no-idle-check --wait 0 -- true >/dev/null 2>"$WORK/claim.err"
RC=$?
[ "$RC" -eq 2 ] && pass "claim blocks --gpu" || fail "claim did not block --gpu (rc=$RC)"
gq run --ignore-claims --reason "validation" -n 1 --gpu "$TARGET_GPU" --no-idle-check --wait 0 -- true \
    >/dev/null 2>"$WORK/breakglass.err"
RC=$?
[ "$RC" -eq 0 ] && pass "break-glass override" || fail "break-glass failed (rc=$RC)"
gq unclaim "$TARGET_GPU" >/dev/null 2>&1 || fail "unclaim"

log "summary"
if [ "$FAILS" -ne 0 ]; then
    echo "FAILURES: $FAILS"
    exit 1
fi
echo "ALL NVIDIA VALIDATION CHECKS PASSED"
