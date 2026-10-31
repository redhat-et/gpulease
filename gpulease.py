#!/usr/bin/env python3
# Copyright 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0
"""gpulease — GPU checkout manager with adoption watcher. Single-file, stdlib only."""

import argparse
import csv
import fcntl
import json
import os
import pwd
import random
import re
import secrets
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

GIB = 1 << 30
__version__ = "0.3.0"

DEFAULTS = {
    "gpus": None,
    "pool": None,
    "vendor": "auto",
    "occupancy_mode": "auto",
    "reserved": {},
    "claims": {},
    "idle_check": True,
    "idle_threshold_gib": 2.0,
    "ttl_minutes": 240.0,
    "watch_interval_seconds": 5.0,
    "adopt_grace_seconds": 30.0,
    "default_wait_seconds": 3600.0,
    "pending_grace_seconds": 30.0,
}

SELF = Path(__file__).resolve()


def system_state_dir():
    return Path(os.environ.get("GPULEASE_SYSTEM_STATE_DIR", "/var/lib/gpulease"))


def state_dir():
    override = os.environ.get("GPULEASE_STATE_DIR")
    if override:
        return Path(override)
    system = system_state_dir()
    if (system / ".shared").exists():
        return system
    base = os.environ.get("XDG_STATE_HOME")
    if base:
        return Path(base) / "gpulease"
    return Path.home() / ".local" / "state" / "gpulease"


def is_shared():
    return (state_dir() / ".shared").exists()


def file_mode():
    return 0o660 if is_shared() else 0o644


def current_uid():
    return os.getuid()


def user_name(uid=None):
    uid = current_uid() if uid is None else uid
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def uid_for_user(name):
    if isinstance(name, int):
        return name
    try:
        return pwd.getpwnam(str(name)).pw_uid
    except KeyError:
        return None


def claims_map(cfg):
    out = {}
    for spec, who in (cfg.get("claims") or {}).items():
        uid = uid_for_user(who)
        if uid is None:
            uid = -1
        for g in parse_gpu_list(spec):
            out[g] = uid
    return out


def claim_owner_name(cfg, gpu):
    for spec, who in (cfg.get("claims") or {}).items():
        if gpu in parse_gpu_list(spec):
            return str(who)
    return None


def effective_pool(cfg, uid=None, ignore_claims=False):
    uid = current_uid() if uid is None else uid
    if ignore_claims:
        reserved = set()
        for spec in cfg.get("reserved", {}):
            reserved.update(parse_gpu_list(spec))
        return [g for g in cfg["gpus"] if g not in reserved]
    cm = claims_map(cfg)
    return [g for g in cfg["pool"] if cm.get(g) in (None, uid)]


def rocm_smi_cmd():
    return os.environ.get("GPULEASE_ROCM_SMI", "rocm-smi")


def nvidia_smi_cmd():
    return os.environ.get("GPULEASE_NVIDIA_SMI", "nvidia-smi")


def proc_root():
    return Path(os.environ.get("GPULEASE_PROC_ROOT", "/proc"))


def ensure_state():
    dirs = (state_dir(), locks_dir(), leases_dir(), releases_dir())
    if is_shared():
        missing = [d for d in dirs if not d.is_dir()]
        if missing:
            raise SystemExit(
                f"gpulease: shared state incomplete ({missing[0]}); "
                "run deploy/install-shared.sh"
            )
        for d in dirs:
            if not os.access(d, os.W_OK | os.X_OK):
                raise SystemExit(
                    f"gpulease: no write access to shared state {d}; "
                    "check group membership and ACLs"
                )
        return
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def locks_dir():
    return state_dir() / "locks"


def leases_dir():
    return state_dir() / "leases"


def releases_dir():
    return state_dir() / "releases"


def release_file(lease_id):
    return releases_dir() / lease_id


def log_path():
    if is_shared():
        return state_dir() / f"watch-{user_name()}.log"
    return state_dir() / "watch.log"


def audit_log(msg):
    ensure_state()
    path = state_dir() / "audit.log"
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {user_name()}({current_uid()}) {msg}\n"
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, file_mode())
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def load_config():
    cfg = dict(DEFAULTS)
    cfg["reserved"] = dict(DEFAULTS["reserved"])
    cfg["claims"] = dict(DEFAULTS["claims"])
    p = state_dir() / "config.json"
    if p.exists():
        try:
            user = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            if is_shared():
                raise SystemExit(f"gpulease: cannot read shared config {p}: {e}")
            user = {}
        for k, v in user.items():
            cfg[k] = v
    if cfg.get("gpus") is None:
        cfg["gpus"] = detect_gpus(resolve_vendor(cfg))
    else:
        cfg["gpus"] = sorted(set(cfg["gpus"]))
    reserved = set()
    for spec in cfg.get("reserved", {}):
        reserved.update(parse_gpu_list(spec))
    if cfg.get("pool") is None:
        cfg["pool"] = [g for g in cfg["gpus"] if g not in reserved]
    else:
        cfg["pool"] = sorted(set(cfg["pool"]))
    return cfg


def save_config(cfg):
    ensure_state()
    atomic_write_json(state_dir() / "config.json", cfg)


def atomic_write_json(path, obj):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    try:
        os.chmod(tmp, file_mode())
    except OSError:
        pass
    os.replace(tmp, path)


@contextmanager
def alloc_lock():
    ensure_state()
    fd = os.open(state_dir() / "alloc.lock", os.O_CREAT | os.O_RDWR, file_mode())
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


class GpuBusy(Exception):
    pass


def acquire_gpu_locks(gpus):
    fds = []
    try:
        for g in gpus:
            p = locks_dir() / f"gpu{g}.lock"
            fd = os.open(p, os.O_CREAT | os.O_RDWR, file_mode())
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                raise GpuBusy(f"gpu{g}")
            fds.append(fd)
        return fds
    except GpuBusy:
        for fd in fds:
            os.close(fd)
        raise


def read_lease(lease_id):
    p = leases_dir() / f"{lease_id}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def read_leases():
    out = {}
    try:
        files = list(leases_dir().glob("*.json"))
    except OSError:
        return out
    for f in files:
        try:
            rec = json.loads(f.read_text())
            out[rec["id"]] = rec
        except (OSError, ValueError, KeyError):
            continue
    return out


def write_lease(rec):
    ensure_state()
    atomic_write_json(leases_dir() / f"{rec['id']}.json", rec)


def remove_lease(lease_id):
    try:
        (leases_dir() / f"{lease_id}.json").unlink()
        return True
    except OSError:
        return False


def new_lease_id(prefix):
    return f"{prefix}-{int(time.time())}-{secrets.token_hex(3)}"


def pid_alive(pid):
    if not pid or pid <= 1:
        return False
    try:
        data = (proc_root() / str(pid) / "stat").read_text()
    except OSError:
        return False
    rp = data.rfind(")")
    if rp < 0:
        return False
    fields = data[rp + 2:].split()
    return bool(fields) and fields[0] != "Z"


def proc_ppid(pid):
    try:
        data = (proc_root() / str(pid) / "stat").read_text()
    except OSError:
        return None
    rp = data.rfind(")")
    if rp < 0:
        return None
    fields = data[rp + 2:].split()
    try:
        return int(fields[1])
    except (IndexError, ValueError):
        return None


def proc_comm(pid):
    try:
        data = (proc_root() / str(pid) / "stat").read_text()
    except OSError:
        return ""
    lp = data.find("(")
    rp = data.rfind(")")
    if lp < 0 or rp < 0:
        return ""
    return data[lp + 1:rp]


def proc_cmdline(pid):
    try:
        raw = (proc_root() / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()


def ancestor_chain(pid, limit=64):
    chain = []
    seen = set()
    cur = pid
    while cur and cur > 1 and cur not in seen and len(chain) < limit:
        chain.append(cur)
        seen.add(cur)
        cur = proc_ppid(cur)
    return chain


def lease_active(rec, cfg, now):
    t = rec.get("type")
    if t in ("run", "checkout"):
        hp = rec.get("holder_pid")
        if hp:
            return pid_alive(hp)
        return now - rec.get("created", 0) < cfg["pending_grace_seconds"]
    if t == "adopted":
        return any(pid_alive(p) for p in rec.get("pids", []))
    if t == "foreign":
        return now - rec.get("heartbeat", 0) <= cfg["adopt_grace_seconds"]
    return False


def reap_locked(leases, cfg, now):
    removed = []
    for lid, rec in list(leases.items()):
        if not lease_active(rec, cfg, now):
            remove_lease(lid)
            leases.pop(lid, None)
            removed.append(lid)
    return removed


def busy_from_leases(leases, cfg, now):
    busy = {}
    for rec in leases.values():
        if lease_active(rec, cfg, now):
            for g in rec.get("gpus", []):
                busy[g] = rec
    return busy


def rocm_json(args):
    try:
        out = subprocess.run(
            [rocm_smi_cmd()] + args,
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    i = out.find("{")
    if i < 0:
        return {}
    try:
        return json.loads(out[i:])
    except ValueError:
        return {}


def rocm_vram():
    data = rocm_json(["--showmeminfo", "vram", "--json"])
    out = {}
    for key, val in data.items():
        m = re.match(r"card(\d+)$", key)
        if not m:
            continue
        used = val.get("VRAM Total Used Memory (B)")
        try:
            out[int(m.group(1))] = int(used)
        except (TypeError, ValueError):
            continue
    return out


def detect_gpus(vendor="amd"):
    if vendor == "nvidia":
        return sorted(nvidia_gpu_info())
    return sorted(rocm_vram())


def rocm_bdf_map():
    data = rocm_json(["--showbus", "--json"])
    out = {}
    for key, val in data.items():
        m = re.match(r"card(\d+)$", key)
        if not m or not isinstance(val, dict):
            continue
        bus = val.get("PCI Bus")
        if bus:
            out[int(m.group(1))] = bus.strip().lower()
    return out


def _nvidia_num(val):
    val = (val or "").strip()
    if not val or "N/A" in val or "Not Supported" in val:
        return None
    try:
        return int(float(val))
    except ValueError:
        return None


def nvidia_csv(args):
    try:
        proc = subprocess.run(
            [nvidia_smi_cmd()] + args,
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return [line for line in proc.stdout.splitlines() if line.strip()]


def nvidia_gpu_info():
    rows = nvidia_csv([
        "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ])
    info = {}
    if rows is not None:
        for row in rows:
            parts = [f.strip() for f in next(csv.reader([row]))]
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            total = _nvidia_num(parts[3]) if len(parts) > 3 else None
            used = _nvidia_num(parts[4]) if len(parts) > 4 else None
            info[idx] = {
                "uuid": None if "N/A" in parts[1] else parts[1],
                "name": parts[2],
                "memory_total": total * 1024 * 1024 if total is not None else None,
                "memory_used": used * 1024 * 1024 if used is not None else None,
                "mig": None,
            }
    if not info:
        try:
            proc = subprocess.run(
                [nvidia_smi_cmd(), "-L"], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if proc.returncode != 0:
            return {}
        for line in proc.stdout.splitlines():
            m = re.search(r"GPU\s+(\d+):\s+(.*?)\s+\(UUID:\s*([^)]+)\)", line)
            if m:
                info[int(m.group(1))] = {
                    "uuid": m.group(3), "name": m.group(2),
                    "memory_total": None, "memory_used": None, "mig": None,
                }
    return info


def nvidia_vram(info=None):
    info = nvidia_gpu_info() if info is None else info
    return {
        idx: data["memory_used"]
        for idx, data in info.items()
        if data.get("memory_used") is not None
    }


def nvidia_uuid_map(info=None):
    info = nvidia_gpu_info() if info is None else info
    return {idx: data["uuid"] for idx, data in info.items() if data.get("uuid")}


def nvidia_compute_apps(info=None):
    rows = nvidia_csv([
        "--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    if rows is None:
        return None
    uuid_to_idx = {uuid: idx for idx, uuid in nvidia_uuid_map(info).items()}
    apps = {}
    for row in rows:
        parts = [f.strip() for f in next(csv.reader([row]))]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        gpu = uuid_to_idx.get(parts[2])
        if gpu is None:
            continue
        mem = _nvidia_num(parts[3]) if len(parts) > 3 else None
        apps.setdefault(pid, {})[gpu] = mem * 1024 * 1024 if mem is not None else None
    return apps


def nvidia_mig_enabled(info=None):
    rows = nvidia_csv([
        "--query-gpu=index,mig.mode.current",
        "--format=csv,noheader,nounits",
    ])
    if rows:
        for row in rows:
            parts = [f.strip() for f in next(csv.reader([row]))]
            if len(parts) < 2:
                continue
            mig = parts[1].strip().lower()
            if mig and mig not in ("n/a", "[n/a]", "none", "disabled"):
                return True
        return False
    try:
        proc = subprocess.run(
            [nvidia_smi_cmd(), "-L"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "MIG" in proc.stdout


def resolve_vendor(cfg):
    vendor = (cfg.get("vendor") or "auto").lower()
    if vendor in ("amd", "nvidia"):
        return vendor
    if vendor != "auto":
        raise SystemExit(f"gpulease: invalid vendor {vendor!r} (use auto|amd|nvidia)")
    amd = bool(rocm_vram())
    nv = bool(nvidia_gpu_info())
    if amd and nv:
        raise SystemExit("gpulease: both AMD and NVIDIA GPUs detected; set vendor=amd|nvidia")
    if nv:
        return "nvidia"
    return "amd"


def resolve_occupancy_mode(cfg, vendor, info=None):
    mode = (cfg.get("occupancy_mode") or "auto").lower()
    if mode in ("vram", "process"):
        return mode
    if mode != "auto":
        raise SystemExit(
            f"gpulease: invalid occupancy_mode {mode!r} (use auto|vram|process)"
        )
    if vendor != "nvidia":
        return "vram"
    info = nvidia_gpu_info() if info is None else info
    if any(
        d.get("memory_total") is not None or d.get("memory_used") is not None
        for d in info.values()
    ):
        return "vram"
    return "process"


def gpu_snapshot(cfg):
    vendor = resolve_vendor(cfg)
    if vendor == "amd":
        return {
            "vendor": "amd", "mode": "vram", "info": {}, "apps": {},
            "vram": rocm_vram(), "busy": set(), "unknown": False,
        }
    info = nvidia_gpu_info()
    mode = resolve_occupancy_mode(cfg, vendor, info)
    apps = nvidia_compute_apps(info)
    if mode == "process" and any(not d.get("uuid") for d in info.values()):
        apps = None
    if apps is None:
        busy = set(cfg["gpus"])
        unknown = True
    else:
        busy = {g for per in apps.values() for g in per}
        unknown = False
    return {
        "vendor": "nvidia", "mode": mode, "info": info, "apps": apps,
        "vram": nvidia_vram(info) if mode == "vram" else {},
        "busy": busy, "unknown": unknown,
    }


def idle_vram(cfg, threshold):
    snap = gpu_snapshot(cfg)
    vram = dict(snap["vram"])
    if snap["unknown"]:
        for g in cfg["pool"]:
            vram[g] = max(vram.get(g, 0), threshold)
    elif snap["mode"] == "process":
        for g in snap["busy"]:
            vram[g] = max(vram.get(g, 0), threshold)
    return vram


def proc_uid(pid):
    try:
        status = (proc_root() / str(pid) / "status").read_text()
    except OSError:
        return None
    m = re.search(r"^Uid:\s+(\d+)", status, re.M)
    return int(m.group(1)) if m else None


def nvidia_own_usage(apps):
    uid = os.getuid()
    usage = {}
    own_vram = {}
    if not apps:
        return usage, own_vram
    for pid, per_gpu in apps.items():
        if proc_uid(pid) != uid:
            continue
        gpu_map = {}
        for g, mem in per_gpu.items():
            gpu_map[g] = mem or 0
            if mem:
                own_vram[g] = own_vram.get(g, 0) + mem
        if gpu_map:
            usage[pid] = gpu_map
    return usage, own_vram


def parse_gpu_list(spec):
    gpus = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            gpus.update(range(a, b + 1))
        elif re.match(r"^\d+$", part):
            gpus.add(int(part))
        else:
            raise ValueError(f"bad GPU spec: {part!r}")
    return sorted(gpus)


def parse_gpu_seq(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)-(\d+)$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            rng = range(a, b + 1) if a <= b else range(a, b - 1, -1)
            for g in rng:
                if g not in out:
                    out.append(g)
        elif re.match(r"^\d+$", part):
            g = int(part)
            if g not in out:
                out.append(g)
        else:
            raise ValueError(f"bad GPU spec: {part!r}")
    return out


def fmt_gpus(gpus):
    return "[" + ",".join(str(g) for g in gpus) + "]"


def fmt_age(seconds):
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d"


def lease_human(rec, now):
    t = rec["type"]
    cmd = rec.get("cmd", "")
    if len(cmd) > 70:
        cmd = cmd[:67] + "..."
    return f"{t} owner={rec.get('owner', '-')} age={fmt_age(now - rec.get('created', now))} cmd={cmd!r}"


def ordered_free(pool, busy, vram, threshold, idle_check, required, prefer):
    free = []
    for g in pool:
        if g in busy:
            continue
        if idle_check and vram.get(g, 0) >= threshold:
            continue
        free.append(g)
    if required:
        return list(required) if all(g in free for g in required) else []
    if prefer:
        pref = [g for g in prefer if g in free]
        rest = [g for g in free if g not in pref]
        return pref + rest
    return free


def try_allocate(n, cfg, owner, kind, wait, idle_check, ttl_minutes=None,
                 quiet=False, required=None, prefer=None, requested=None,
                 pool=None, extra=None):
    pool = list(cfg["pool"]) if pool is None else list(pool)
    threshold = cfg["idle_threshold_gib"] * GIB
    if idle_check is None:
        idle_check = cfg["idle_check"]
    deadline = None if wait is None or wait < 0 else time.time() + wait
    last_msg = 0.0
    while True:
        now = time.time()
        with alloc_lock():
            leases = read_leases()
            reap_locked(leases, cfg, now)
            busy = busy_from_leases(leases, cfg, now)
        vram = idle_vram(cfg, threshold) if idle_check else {}
        cand = ordered_free(pool, busy, vram, threshold, idle_check, required, prefer)
        if len(cand) >= n:
            with alloc_lock():
                leases = read_leases()
                reap_locked(leases, cfg, time.time())
                busy2 = busy_from_leases(leases, cfg, time.time())
                cand2 = ordered_free(pool, busy2, vram, threshold, idle_check, required, prefer)
                if len(cand2) >= n:
                    gpus = cand2[:n]
                    try:
                        fds = acquire_gpu_locks(gpus)
                    except GpuBusy:
                        fds = None
                    if fds is not None:
                        now2 = time.time()
                        rec = {
                            "id": new_lease_id(kind),
                            "type": kind,
                            "owner": owner,
                            "uid": current_uid(),
                            "gpus": gpus,
                            "pids": [],
                            "holder_pid": os.getpid(),
                            "cmd": "",
                            "created": now2,
                            "expires": now2 + ttl_minutes * 60 if ttl_minutes else None,
                            "heartbeat": now2,
                            "requested": requested,
                        }
                        if extra:
                            rec.update(extra)
                        write_lease(rec)
                        return rec, fds
            continue
        if deadline is not None and time.time() >= deadline:
            return None, None
        if not quiet and time.time() - last_msg > 15:
            holders = []
            for rec in busy.values():
                holders.append(f"{fmt_gpus(rec.get('gpus', []))}:{rec['type']}")
            avail = ordered_free(pool, busy, vram, threshold, idle_check, None, None)
            want = ""
            if required:
                want = f" want={fmt_gpus(required)}"
            elif prefer:
                want = f" prefer={fmt_gpus(prefer)}"
            idlebusy_msg = ""
            if idle_check:
                idlebusy = [g for g in pool if g not in busy and vram.get(g, 0) >= threshold]
                if idlebusy:
                    idlebusy_msg = f" unleased-busy={fmt_gpus(idlebusy)}"
            print(
                f"gpulease: waiting for {n} GPU(s); pool={fmt_gpus(pool)} "
                f"free={fmt_gpus(avail)}{want} busy={holders}{idlebusy_msg}",
                file=sys.stderr, flush=True,
            )
            last_msg = time.time()
        time.sleep(1.0 + random.random() * 2.0)


def build_env(gpus, lease_id, vendor="amd"):
    env = os.environ.copy()
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        env.pop(key, None)
    vis = ",".join(str(g) for g in gpus)
    if vendor == "nvidia":
        env["CUDA_VISIBLE_DEVICES"] = vis
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    else:
        env["HIP_VISIBLE_DEVICES"] = vis
    env["GPULEASE_LEASE_ID"] = lease_id
    env["GPULEASE_GPUS"] = vis
    return env


def export_lines(rec):
    vis = ",".join(str(g) for g in rec["gpus"])
    lines = []
    if rec.get("vendor") == "nvidia":
        lines.append(f"export CUDA_VISIBLE_DEVICES={vis}")
        lines.append("export CUDA_DEVICE_ORDER=PCI_BUS_ID")
    else:
        lines.append(f"export HIP_VISIBLE_DEVICES={vis}")
    lines.append(f"export GPULEASE_LEASE_ID={rec['id']}")
    lines.append(f"export GPULEASE_GPUS={vis}")
    return lines


def watch_log(msg):
    ensure_state()
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n"
    try:
        fd = os.open(log_path(), os.O_CREAT | os.O_APPEND | os.O_WRONLY, file_mode())
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)
    except OSError:
        pass


def scan_own_usage(threshold_bytes=0):
    uid = os.getuid()
    root = proc_root()
    usage = {}
    own_vram = {}
    try:
        entries = list(root.iterdir())
    except OSError:
        return usage, own_vram
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue
        m = re.search(r"^Uid:\s+(\d+)", status, re.M)
        if not m or int(m.group(1)) != uid:
            continue
        try:
            fds = list((entry / "fdinfo").iterdir())
        except OSError:
            continue
        per_bdf = {}
        for fd in fds:
            try:
                txt = fd.read_text()
            except OSError:
                continue
            pdev = None
            vram_kib = 0
            for line in txt.splitlines():
                if line.startswith("drm-pdev:"):
                    pdev = line.split(":", 1)[1].strip().lower()
                elif line.startswith("drm-memory-vram:"):
                    try:
                        v = int(line.split(":", 1)[1].strip().split()[0])
                        vram_kib = max(vram_kib, v)
                    except (IndexError, ValueError):
                        pass
            if pdev:
                per_bdf[pdev] = max(per_bdf.get(pdev, 0), vram_kib * 1024)
        if per_bdf:
            usage[pid] = per_bdf
    bdf_map = rocm_bdf_map()
    bdf_to_gpu = {bdf: g for g, bdf in bdf_map.items()}
    result = {}
    for pid, per_bdf in usage.items():
        per_gpu = {}
        for bdf, vram in per_bdf.items():
            g = bdf_to_gpu.get(bdf)
            if g is None:
                continue
            per_gpu[g] = max(per_gpu.get(g, 0), vram)
        if not per_gpu:
            continue
        for g, vram in per_gpu.items():
            own_vram[g] = own_vram.get(g, 0) + vram
        busy = {g: v for g, v in per_gpu.items() if v >= threshold_bytes}
        if busy:
            result[pid] = busy
    return result, own_vram


def build_clusters(usage):
    pids = sorted(usage)
    chains = {p: ancestor_chain(p) for p in pids}
    parent = {p: p for p in pids}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, p in enumerate(pids):
        for q in pids[i + 1:]:
            common = (set(chains[p]) & set(chains[q])) - {0, 1}
            if common:
                union(p, q)
    groups = {}
    for p in pids:
        groups.setdefault(find(p), []).append(p)
    clusters = []
    for members in groups.values():
        gpus = sorted({g for p in members for g in usage[p]})
        root = min(members, key=lambda p: len(chains[p]))
        best_real = None
        best_any = None
        for p in members:
            for depth, anc in enumerate(chains[p]):
                c = proc_cmdline(anc).strip()
                if not c:
                    c = proc_comm(anc).strip()
                    if not c:
                        continue
                if best_any is None or depth < best_any[0]:
                    best_any = (depth, c)
                if (" " in c or "/" in c):
                    if best_real is None or depth < best_real[0]:
                        best_real = (depth, c)
        cmd = (best_real or best_any or (0, proc_comm(root)))[1]
        clusters.append({"pids": sorted(members), "gpus": gpus, "cmd": cmd.strip()})
    return clusters


def live_holder_pids(leases, cfg, now):
    pids = set()
    for rec in leases.values():
        if rec.get("type") in ("run", "checkout") and lease_active(rec, cfg, now):
            if rec.get("holder_pid"):
                pids.add(rec["holder_pid"])
            pids.update(rec.get("pids", []))
    return pids


def filter_covered(usage, holders):
    if not holders:
        return usage
    out = {}
    for pid, per_gpu in usage.items():
        chain = set(ancestor_chain(pid))
        if chain & holders:
            continue
        out[pid] = per_gpu
    return out


def reconcile_adopted(leases, clusters, now, owner):
    events = []
    adopted = [r for r in leases.values() if r.get("type") == "adopted"]
    matched = set()
    for cluster in clusters:
        cpid = set(cluster["pids"])
        hit = None
        for rec in adopted:
            if rec["id"] in matched:
                continue
            if set(rec.get("pids", [])) & cpid:
                hit = rec
                break
        if hit:
            matched.add(hit["id"])
            changed = (hit.get("gpus") != cluster["gpus"]
                       or hit.get("pids") != cluster["pids"])
            hit["gpus"] = cluster["gpus"]
            hit["pids"] = cluster["pids"]
            hit["cmd"] = cluster["cmd"]
            hit["heartbeat"] = now
            hit["uid"] = current_uid()
            hit["user"] = user_name()
            write_lease(hit)
            if changed:
                events.append(("update", hit))
        else:
            rec = {
                "id": new_lease_id("adopted"),
                "type": "adopted",
                "owner": owner,
                "uid": current_uid(),
                "user": user_name(),
                "gpus": cluster["gpus"],
                "pids": cluster["pids"],
                "holder_pid": None,
                "cmd": cluster["cmd"],
                "created": now,
                "expires": None,
                "heartbeat": now,
            }
            write_lease(rec)
            leases[rec["id"]] = rec
            events.append(("adopt", rec))
    for rec in adopted:
        if rec["id"] in matched:
            continue
        if any(pid_alive(p) for p in rec.get("pids", [])):
            continue
        remove_lease(rec["id"])
        leases.pop(rec["id"], None)
        events.append(("release", rec))
    return events


def reconcile_foreign(leases, cfg, own_vram, card_vram, now,
                      mode="vram", busy_gpus=None, unknown=False):
    threshold = cfg["idle_threshold_gib"] * GIB
    covered = set()
    for rec in leases.values():
        if rec.get("type") in ("run", "checkout", "adopted") and lease_active(rec, cfg, now):
            covered.update(rec.get("gpus", []))
    if mode == "process":
        busy = set(cfg["pool"]) if unknown else set(busy_gpus or ())
        foreign_gpus = [g for g in cfg["pool"] if g in busy and g not in covered]
    elif is_shared():
        foreign_gpus = [
            g for g in cfg["pool"]
            if card_vram.get(g, 0) >= threshold and g not in covered
        ]
    else:
        foreign_gpus = []
        for g in cfg["pool"]:
            used = card_vram.get(g, 0)
            explained = own_vram.get(g, 0)
            if used - explained >= threshold:
                foreign_gpus.append(g)
    reason = "uncovered compute processes" if mode == "process" else "unattributed VRAM usage"
    events = []
    existing = [r for r in leases.values() if r.get("type") == "foreign"]
    if foreign_gpus:
        if existing:
            rec = existing[0]
            for extra in existing[1:]:
                remove_lease(extra["id"])
                leases.pop(extra["id"], None)
            if rec.get("gpus") != foreign_gpus or rec.get("cmd") != reason:
                rec["gpus"] = foreign_gpus
                rec["cmd"] = reason
                events.append(("update", rec))
            rec["heartbeat"] = now
            write_lease(rec)
        else:
            rec = {
                "id": new_lease_id("foreign"),
                "type": "foreign",
                "owner": "external",
                "gpus": foreign_gpus,
                "pids": [],
                "holder_pid": None,
                "cmd": reason,
                "created": now,
                "expires": None,
                "heartbeat": now,
            }
            write_lease(rec)
            leases[rec["id"]] = rec
            events.append(("foreign", rec))
    else:
        for rec in existing:
            if now - rec.get("heartbeat", 0) > cfg["adopt_grace_seconds"]:
                remove_lease(rec["id"])
                leases.pop(rec["id"], None)
                events.append(("release", rec))
    return events


def watch_cycle(cfg, announce=False):
    now = time.time()
    snap = gpu_snapshot(cfg)
    if snap["vendor"] == "nvidia":
        usage, own_vram = nvidia_own_usage(snap["apps"])
    else:
        usage, own_vram = scan_own_usage(cfg["idle_threshold_gib"] * GIB)
    if not usage:
        usage = {}
    with alloc_lock():
        leases = read_leases()
        reap_locked(leases, cfg, now)
        holders = live_holder_pids(leases, cfg, now)
        clusters = build_clusters(filter_covered(usage, holders))
        events = reconcile_adopted(leases, clusters, now, user_name())
        events += reconcile_foreign(
            leases, cfg, own_vram, snap["vram"], now,
            mode=snap["mode"], busy_gpus=snap["busy"], unknown=snap["unknown"],
        )
    for action, rec in events:
        msg = f"{action} {rec['id']} type={rec['type']} gpus={fmt_gpus(rec.get('gpus', []))} pids={rec.get('pids', [])} cmd={rec.get('cmd', '')!r}"
        watch_log(msg)
        if announce:
            print("gpulease: " + msg)
    return events


def resolve_gpu_request(args, cfg):
    if args.ignore_claims and not (args.reason or "").strip():
        print("gpulease: --ignore-claims requires a non-empty --reason", file=sys.stderr)
        return None
    if args.reason and not args.ignore_claims:
        print("gpulease: --reason requires --ignore-claims", file=sys.stderr)
        return None
    uid = current_uid()
    pool = effective_pool(cfg, uid, ignore_claims=args.ignore_claims)
    cm = claims_map(cfg)
    reserved = set()
    for spec in cfg.get("reserved", {}):
        reserved.update(parse_gpu_list(spec))

    def why(g):
        if g in reserved:
            return "reserved"
        owner = cm.get(g)
        if owner is not None and owner != uid:
            return f"claimed by {claim_owner_name(cfg, g)}"
        return "not in pool"

    required = args.gpu_req
    prefer = args.prefer_req
    requested = None
    if required:
        bad = [g for g in required if g not in pool]
        if bad:
            details = "; ".join(f"gpu {g} {why(g)}" for g in bad)
            print(f"gpulease: cannot require gpu(s) {fmt_gpus(bad)}: {details}", file=sys.stderr)
            return None
        requested = "gpu=" + ",".join(str(g) for g in required)
    elif prefer:
        kept = [g for g in prefer if g in pool]
        dropped = [g for g in prefer if g not in pool]
        if dropped:
            print(f"gpulease: ignoring preferred gpu(s) {fmt_gpus(dropped)} "
                  f"({why(dropped[0])})", file=sys.stderr)
        if not kept:
            print("gpulease: no preferred GPU is allocatable", file=sys.stderr)
            return None
        prefer = kept
        requested = "prefer=" + ",".join(str(g) for g in prefer)
    if args.num is None:
        n = len(required) if required else 1
    elif required and args.num != len(required):
        print(f"gpulease: -n {args.num} conflicts with --gpu "
              f"({len(required)} GPU(s) requested)", file=sys.stderr)
        return None
    else:
        n = args.num
    return n, required, prefer, requested, pool


def apply_claim_override(rec, cfg, args):
    if not args.ignore_claims:
        return
    cm = claims_map(cfg)
    ignored = [g for g in rec["gpus"] if cm.get(g) not in (None, current_uid())]
    with alloc_lock():
        latest = read_lease(rec["id"]) or rec
        latest["claims_ignored"] = ignored
        latest["reason"] = args.reason
        write_lease(latest)
        rec.update(latest)
    audit_log(
        f"ignore-claims lease={rec['id']} gpus={fmt_gpus(rec['gpus'])} "
        f"claims_ignored={fmt_gpus(ignored)} reason={args.reason!r}"
    )


def cmd_run(args):
    cfg = load_config()
    req = resolve_gpu_request(args, cfg)
    if req is None:
        return 2
    n, required, prefer, requested, pool = req
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("gpulease: run requires a command after --", file=sys.stderr)
        return 2
    vendor = resolve_vendor(cfg)
    rec, fds = try_allocate(
        n, cfg, args.owner, "run", args.wait, args.idle_check, quiet=args.quiet,
        required=required, prefer=prefer, requested=requested, pool=pool,
        extra={"vendor": vendor},
    )
    if rec is None:
        print(
            f"gpulease: timed out waiting for {n} GPU(s) in pool {fmt_gpus(pool)}",
            file=sys.stderr,
        )
        return 3
    apply_claim_override(rec, cfg, args)
    env = build_env(rec["gpus"], rec["id"], rec.get("vendor", "amd"))
    override = ""
    if rec.get("claims_ignored"):
        override = f" claims-ignored={fmt_gpus(rec['claims_ignored'])}"
    print(
        f"gpulease: lease {rec['id']} gpus={fmt_gpus(rec['gpus'])} "
        f"(pool {fmt_gpus(pool)}, uid {rec.get('uid')}){override}",
        file=sys.stderr, flush=True,
    )
    try:
        proc = subprocess.Popen(cmd, env=env)

        def forward(signum, frame):
            try:
                proc.terminate()
            except OSError:
                pass

        old_term = signal.signal(signal.SIGTERM, forward)
        old_int = signal.signal(signal.SIGINT, forward)
        with alloc_lock():
            latest = read_lease(rec["id"])
            if latest:
                latest["pids"] = [proc.pid]
                latest["cmd"] = " ".join(cmd)[:200]
                write_lease(latest)
        try:
            rc = proc.wait()
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
            with alloc_lock():
                remove_lease(rec["id"])
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
        return rc if rc >= 0 else 128 - rc
    except OSError as e:
        with alloc_lock():
            remove_lease(rec["id"])
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
        print(f"gpulease: failed to exec command: {e}", file=sys.stderr)
        return 127


def cmd_checkout(args):
    cfg = load_config()
    req = resolve_gpu_request(args, cfg)
    if req is None:
        return 2
    n, required, prefer, requested, pool = req
    if args.idle_check is None:
        args.idle_check = cfg["idle_check"]
    ttl = args.ttl if args.ttl is not None else cfg["ttl_minutes"]
    vendor = resolve_vendor(cfg)
    rec, fds = try_allocate(
        n, cfg, args.owner, "checkout", args.wait, args.idle_check,
        ttl_minutes=ttl, quiet=args.quiet,
        required=required, prefer=prefer, requested=requested, pool=pool,
        extra={"vendor": vendor},
    )
    if rec is None:
        print(
            f"gpulease: timed out waiting for {n} GPU(s) in pool {fmt_gpus(pool)}",
            file=sys.stderr,
        )
        return 3
    apply_claim_override(rec, cfg, args)
    try:
        release_file(rec["id"]).unlink()
    except OSError:
        pass
    logf = os.fdopen(
        os.open(log_path(), os.O_CREAT | os.O_APPEND | os.O_WRONLY, file_mode()), "ab"
    )
    try:
        holder = subprocess.Popen(
            [sys.executable, str(SELF), "_hold", rec["id"]],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=logf,
            pass_fds=tuple(fds), start_new_session=True, env=os.environ.copy(),
        )
    except OSError as e:
        with alloc_lock():
            remove_lease(rec["id"])
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
        print(f"gpulease: failed to start lease holder: {e}", file=sys.stderr)
        return 1
    finally:
        logf.close()
    with alloc_lock():
        latest = read_lease(rec["id"])
        if latest:
            latest["holder_pid"] = holder.pid
            latest["cmd"] = "checkout"
            write_lease(latest)
            rec = latest
    print(
        f"gpulease: lease {rec['id']} gpus={fmt_gpus(rec['gpus'])} ttl={ttl:.0f}m "
        f"(release with: gpulease release {rec['id']})",
        file=sys.stderr,
    )
    if args.json:
        print(json.dumps({"lease_id": rec["id"], "gpus": rec["gpus"], "ttl_minutes": ttl}))
    else:
        for line in export_lines(rec):
            print(line)
    return 0


def cmd_hold(args):
    lease_id = args.lease_id
    rec = read_lease(lease_id)
    if not rec:
        return 1
    rel = release_file(lease_id)
    while True:
        if rel.exists():
            break
        rec = read_lease(lease_id)
        if not rec:
            break
        exp = rec.get("expires")
        if exp and time.time() > exp:
            break
        time.sleep(1.0)
    with alloc_lock():
        remove_lease(lease_id)
    try:
        rel.unlink()
    except OSError:
        pass
    watch_log(f"holder-exit {lease_id}")
    return 0


def check_lease_owner(rec, action):
    if not is_shared():
        return True
    uid = rec.get("uid")
    if uid is None or uid == current_uid() or current_uid() == 0:
        return True
    print(
        f"gpulease: cannot {action} lease {rec['id']}: owned by {user_name(uid)}",
        file=sys.stderr,
    )
    return False


def cmd_release(args):
    ensure_state()
    lease_id = args.lease_id
    with alloc_lock():
        rec = read_lease(lease_id)
        if not rec:
            print(f"gpulease: no such lease {lease_id}", file=sys.stderr)
            return 1
        if rec.get("type") in ("adopted", "foreign"):
            remove_lease(lease_id)
            print(f"gpulease: removed {rec['type']} lease {lease_id}")
            return 0
        if not check_lease_owner(rec, "release"):
            return 1
        rf = release_file(lease_id)
        rf.write_text("released\n")
        try:
            os.chmod(rf, file_mode())
        except OSError:
            pass
        hp = rec.get("holder_pid")
    if hp and pid_alive(hp):
        for _ in range(20):
            if not read_lease(lease_id):
                break
            time.sleep(0.25)
    with alloc_lock():
        rec = read_lease(lease_id)
        if rec:
            remove_lease(lease_id)
    print(f"gpulease: released {lease_id}")
    return 0


def cmd_renew(args):
    cfg = load_config()
    ttl = args.ttl if args.ttl is not None else cfg["ttl_minutes"]
    with alloc_lock():
        rec = read_lease(args.lease_id)
        if not rec:
            print(f"gpulease: no such lease {args.lease_id}", file=sys.stderr)
            return 1
        if rec.get("type") not in ("checkout",):
            print("gpulease: renew only applies to checkout leases", file=sys.stderr)
            return 1
        if not check_lease_owner(rec, "renew"):
            return 1
        rec["expires"] = time.time() + ttl * 60
        write_lease(rec)
    print(f"gpulease: renewed {rec['id']} ttl={ttl:.0f}m")
    return 0


def cmd_env(args):
    rec = read_lease(args.lease_id)
    if not rec:
        print(f"gpulease: no such lease {args.lease_id}", file=sys.stderr)
        return 1
    for line in export_lines(rec):
        print(line)
    return 0


def cmd_status(args):
    cfg = load_config()
    now = time.time()
    uid = current_uid()
    with alloc_lock():
        leases = read_leases()
        removed = reap_locked(leases, cfg, now)
        active = [r for r in leases.values() if lease_active(r, cfg, now)]
    snap = gpu_snapshot(cfg)
    vram = snap["vram"]
    mode = snap["mode"]
    busy = snap["busy"]
    unknown = snap["unknown"]
    threshold = cfg["idle_threshold_gib"] * GIB
    cm = claims_map(cfg)
    pool = effective_pool(cfg, uid)
    by_gpu = {}
    for rec in active:
        for g in rec.get("gpus", []):
            by_gpu[g] = rec
    reserved_gpus = set()
    for spec in cfg["reserved"]:
        reserved_gpus.update(parse_gpu_list(spec))
    rows = []
    for g in cfg["gpus"]:
        rec = by_gpu.get(g)
        used = vram.get(g)
        reserved = g in reserved_gpus
        claim = claim_owner_name(cfg, g)
        claim_uid = cm.get(g)
        if rec:
            row = {
                "gpu": g, "reserved": reserved, "claim": claim, "vram_bytes": used,
                "state": "leased", "lease_type": rec["type"],
                "owner": rec.get("owner", ""), "uid": rec.get("uid"),
                "age_seconds": int(now - rec.get("created", now)),
                "cmd": rec.get("cmd", ""), "lease_id": rec["id"],
            }
        else:
            if mode == "process":
                is_busy = unknown or g in busy
            else:
                is_busy = used is not None and used >= threshold
            if reserved:
                state = "reserved-busy" if is_busy else "reserved"
            elif unknown:
                state = "unknown"
            elif is_busy:
                state = "busy"
            elif claim_uid is not None:
                state = "claimed"
            else:
                state = "free"
            row = {
                "gpu": g, "reserved": reserved, "claim": claim, "vram_bytes": used,
                "state": state, "lease_type": None, "owner": None, "uid": None,
                "age_seconds": None, "cmd": None, "lease_id": None,
            }
        rows.append(row)
    if args.json:
        print(json.dumps({
            "pool": cfg["pool"], "reserved": cfg["reserved"], "claims": cfg["claims"],
            "effective_pool": pool, "uid": uid, "user": user_name(uid),
            "vendor": snap["vendor"], "occupancy_mode": mode,
            "gpu_info": snap["info"], "unknown": unknown,
            "shared": is_shared(), "state_dir": str(state_dir()),
            "leases": active, "gpus": rows, "reaped": removed,
        }, indent=2))
        return 0
    print(f"pool={fmt_gpus(cfg['pool'])} effective={fmt_gpus(pool)} "
          f"vendor={snap['vendor']} occupancy={mode} "
          f"claims={cfg['claims']} reserved={cfg['reserved']} "
          f"idle_threshold={cfg['idle_threshold_gib']:.1f}GiB "
          f"mode={'shared' if is_shared() else 'user'}")
    print(f"{'GPU':<4} {'RES':<4} {'CLAIM':<12} {'VRAM':>9} {'STATE':<14} {'TYPE':<8} {'OWNER':<10} {'AGE':<6} CMD")
    for row in rows:
        vram_s = "n/a" if row["vram_bytes"] is None else f"{row['vram_bytes'] / GIB:8.1f}G"
        print(
            f"{row['gpu']:<4} {'yes' if row['reserved'] else '-':<4} "
            f"{(row['claim'] or '-'):<12} {vram_s:>9} "
            f"{row['state']:<14} {(row['lease_type'] or '-'):<8} {(row['owner'] or '-'):<10} "
            f"{fmt_age(row['age_seconds']) if row['age_seconds'] is not None else '-':<6} "
            f"{row['cmd'] or ''}"
        )
    for rec in active:
        holders = ",".join(str(p) for p in rec.get("pids", [])[:6])
        want = f" want={rec['requested']}" if rec.get("requested") else ""
        over = ""
        if rec.get("claims_ignored"):
            over = f" claims-ignored={fmt_gpus(rec['claims_ignored'])} reason={rec.get('reason')!r}"
        print(f"lease {rec['id']} {lease_human(rec, now)} "
              f"gpus={fmt_gpus(rec.get('gpus', []))} pids=[{holders}]{want}{over}")
    return 0


def cmd_pool(args):
    cfg = load_config()
    uid = current_uid()
    vendor = resolve_vendor(cfg)
    mode = resolve_occupancy_mode(cfg, vendor)
    print(json.dumps({
        "gpus": cfg["gpus"], "pool": cfg["pool"],
        "effective_pool": effective_pool(cfg, uid),
        "vendor": vendor, "occupancy_mode": mode,
        "claims": cfg["claims"], "reserved": cfg["reserved"],
        "shared": is_shared(), "state_dir": str(state_dir()),
        "user": user_name(uid), "uid": uid,
        "idle_check": cfg["idle_check"],
        "idle_threshold_gib": cfg["idle_threshold_gib"],
        "ttl_minutes": cfg["ttl_minutes"],
        "default_wait_seconds": cfg["default_wait_seconds"],
        "watch_interval_seconds": cfg["watch_interval_seconds"],
    }, indent=2))
    return 0


def cmd_reserve(args):
    cfg = load_config()
    gpus = parse_gpu_list(args.spec)
    spec = f"{min(gpus)}-{max(gpus)}" if len(gpus) > 1 else str(gpus[0])
    note = args.for_who or "reserved"
    with alloc_lock():
        cfg = load_config()
        cfg["reserved"][spec] = note
        cfg["pool"] = sorted(set(cfg["pool"]) - set(gpus))
        save_config(cfg)
    audit_log(f"reserve {fmt_gpus(gpus)} note={note!r}")
    print(f"gpulease: reserved {fmt_gpus(gpus)} ({note}); pool now {fmt_gpus(cfg['pool'])}")
    return 0


def cmd_unreserve(args):
    gpus = parse_gpu_list(args.spec)
    with alloc_lock():
        cfg = load_config()
        changed = []
        for spec in list(cfg["reserved"]):
            r = set(parse_gpu_list(spec))
            if r & set(gpus):
                changed.append(spec)
                del cfg["reserved"][spec]
        cfg["pool"] = sorted(set(cfg["pool"]) | set(gpus))
        save_config(cfg)
    audit_log(f"unreserve {fmt_gpus(gpus)}")
    print(f"gpulease: unreserved {fmt_gpus(gpus)} (removed entries: {changed}); "
          f"pool now {fmt_gpus(cfg['pool'])}")
    return 0


def claim_name_map(cfg):
    out = {}
    for spec, who in (cfg.get("claims") or {}).items():
        for g in parse_gpu_list(spec):
            out[g] = str(who)
    return out


def rebuild_claims(cfg, per_gpu):
    by_owner = {}
    for g, who in sorted(per_gpu.items()):
        by_owner.setdefault(who, []).append(g)
    claims = {}
    for who, gs in by_owner.items():
        start = prev = gs[0]
        for g in gs[1:]:
            if g == prev + 1:
                prev = g
                continue
            key = f"{start}-{prev}" if start != prev else str(start)
            claims[key] = who
            start = prev = g
        key = f"{start}-{prev}" if start != prev else str(start)
        claims[key] = who
    cfg["claims"] = claims


def cmd_claim(args):
    gpus = parse_gpu_list(args.spec)
    who = args.for_who or user_name()
    if uid_for_user(who) is None:
        print(f"gpulease: unknown user {who!r}", file=sys.stderr)
        return 1
    with alloc_lock():
        cfg = load_config()
        unknown = [g for g in gpus if g not in cfg["gpus"]]
        if unknown:
            print(f"gpulease: gpu(s) {fmt_gpus(unknown)} do not exist", file=sys.stderr)
            return 1
        per = claim_name_map(cfg)
        for g in gpus:
            per[g] = who
        rebuild_claims(cfg, per)
        save_config(cfg)
        claims = cfg["claims"]
    audit_log(f"claim {fmt_gpus(gpus)} for {who}")
    print(f"gpulease: claimed {fmt_gpus(gpus)} for {who}; claims now {claims}")
    return 0


def cmd_unclaim(args):
    gpus = parse_gpu_list(args.spec)
    with alloc_lock():
        cfg = load_config()
        per = claim_name_map(cfg)
        for g in gpus:
            per.pop(g, None)
        rebuild_claims(cfg, per)
        save_config(cfg)
        claims = cfg["claims"]
    audit_log(f"unclaim {fmt_gpus(gpus)}")
    print(f"gpulease: unclaimed {fmt_gpus(gpus)}; claims now {claims}")
    return 0


def cmd_reap(args):
    cfg = load_config()
    with alloc_lock():
        leases = read_leases()
        removed = reap_locked(leases, cfg, time.time())
    if removed:
        print("gpulease: reaped " + ", ".join(removed))
    elif not args.quiet:
        print("gpulease: nothing to reap")
    return 0


def cmd_watch(args):
    cfg = load_config()
    interval = args.interval if args.interval is not None else cfg["watch_interval_seconds"]
    if args.once:
        events = watch_cycle(cfg, announce=True)
        if not events:
            print("gpulease: watch cycle: no changes")
        return 0
    watch_log("watch-start")
    while True:
        try:
            watch_cycle(cfg, announce=False)
        except Exception as e:
            watch_log(f"watch-error {e!r}")
        time.sleep(interval)


def build_parser():
    p = argparse.ArgumentParser(prog="gpulease", description="GPU leases with adoption of unmanaged jobs")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, with_wait=True):
        sp.add_argument("-n", "--num", type=int, default=None,
                        help="number of GPUs (default 1; inferred from --gpu)")
        sp.add_argument("--owner", default=os.environ.get("GPULEASE_OWNER") or os.environ.get("USER") or "user")
        grp = sp.add_mutually_exclusive_group()
        grp.add_argument("-g", "--gpu", dest="gpu_req", type=parse_gpu_seq, default=None,
                         metavar="SPEC",
                         help="require exact GPU(s), e.g. 6 or 6,4; order preserved; waits until free")
        grp.add_argument("-p", "--prefer", dest="prefer_req", type=parse_gpu_seq, default=None,
                         metavar="SPEC",
                         help="prefer GPU(s) when free, e.g. 7,6; silently falls back to other free pool GPUs")
        if with_wait:
            sp.add_argument("--wait", type=float, default=None,
                            help="seconds to wait for free GPUs; 0=fail fast, -1=forever")
        sp.add_argument("--idle-check", dest="idle_check", action="store_true", default=None)
        sp.add_argument("--no-idle-check", dest="idle_check", action="store_false")
        sp.add_argument("--ignore-claims", action="store_true",
                        help="break glass: bypass claims (requires --reason); still respects leases and VRAM")
        sp.add_argument("--reason", default=None,
                        help="audit reason for --ignore-claims")
        sp.add_argument("-q", "--quiet", action="store_true")

    sp = sub.add_parser("run", help="wait for N GPUs, run command with them, release on exit")
    add_common(sp)
    sp.add_argument("command", nargs=argparse.REMAINDER)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("checkout", help="acquire GPUs for a long-lived job; prints env exports")
    add_common(sp)
    sp.add_argument("--ttl", type=float, default=None, help="lease TTL in minutes")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_checkout)

    sp = sub.add_parser("release", help="release a checkout lease")
    sp.add_argument("lease_id")
    sp.set_defaults(func=cmd_release)

    sp = sub.add_parser("renew", help="extend a checkout lease TTL")
    sp.add_argument("lease_id")
    sp.add_argument("--ttl", type=float, default=None)
    sp.set_defaults(func=cmd_renew)

    sp = sub.add_parser("env", help="print env exports for a lease")
    sp.add_argument("lease_id")
    sp.set_defaults(func=cmd_env)

    sp = sub.add_parser("status", help="show GPU state: leases, adoption, rocm-smi VRAM")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("pool", help="show pool/reserved/threshold config")
    sp.set_defaults(func=cmd_pool)

    sp = sub.add_parser("reserve", help="exclude GPUs from the pool (e.g. reserve 0-3 --for alice)")
    sp.add_argument("spec")
    sp.add_argument("--for", dest="for_who", default=None)
    sp.set_defaults(func=cmd_reserve)

    sp = sub.add_parser("unreserve", help="return reserved GPUs to the pool")
    sp.add_argument("spec")
    sp.set_defaults(func=cmd_unreserve)

    sp = sub.add_parser("claim", help="claim GPUs for a user (e.g. claim 0-3 --for alice)")
    sp.add_argument("spec")
    sp.add_argument("--for", dest="for_who", default=None)
    sp.set_defaults(func=cmd_claim)

    sp = sub.add_parser("unclaim", help="remove a claim")
    sp.add_argument("spec")
    sp.set_defaults(func=cmd_unclaim)

    sp = sub.add_parser("reap", help="drop stale lease records")
    sp.add_argument("-q", "--quiet", action="store_true")
    sp.set_defaults(func=cmd_reap)

    sp = sub.add_parser("watch", help="adoption watcher loop (usually under systemd --user)")
    sp.add_argument("--once", action="store_true")
    sp.add_argument("--interval", type=float, default=None)
    sp.set_defaults(func=cmd_watch)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_hold":
        if len(argv) != 2:
            print("gpulease: internal _hold requires a lease id", file=sys.stderr)
            return 2
        ensure_state()
        return cmd_hold(argparse.Namespace(lease_id=argv[1]))
    args = build_parser().parse_args(argv)
    cfg = load_config()
    if not cfg["gpus"]:
        print(
            "gpulease: no GPUs detected (need rocm-smi or nvidia-smi; set "
            "GPULEASE_ROCM_SMI/GPULEASE_NVIDIA_SMI, vendor, or configure gpus in "
            f"{state_dir() / 'config.json'})",
            file=sys.stderr,
        )
        return 2
    vendor = resolve_vendor(cfg)
    if vendor == "nvidia" and nvidia_mig_enabled():
        print(
            "gpulease: MIG is enabled; whole-GPU leases are not supported on MIG "
            "instances (disable MIG or use vendor-specific tooling)",
            file=sys.stderr,
        )
        return 2
    if args.cmd in ("run", "checkout") and args.wait is None:
        args.wait = cfg["default_wait_seconds"]
    ensure_state()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
