# Copyright 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gpulease  # noqa: E402

MODULE = ROOT / "gpulease.py"

FAKE_ROCM_SRC = '''#!/usr/bin/env python3
import json, os, sys
state = json.load(open(os.environ["FAKE_ROCM_STATE"]))
if "--showmeminfo" in sys.argv:
    print(json.dumps({f"card{g}": {"VRAM Total Used Memory (B)": str(b)} for g, b in state.get("vram", {}).items()}))
elif "--showbus" in sys.argv:
    print(json.dumps({f"card{g}": {"PCI Bus": bdf} for g, bdf in state.get("bus", {}).items()}))
'''


def write_fake_proc(root, pid, comm, ppid, fdinfo, uid=None, cmdline=""):
    d = root / str(pid)
    (d / "fdinfo").mkdir(parents=True, exist_ok=True)
    (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12345 0\n")
    (d / "status").write_text(f"Name:\t{comm}\nUid:\t{uid or os.getuid()}\t{uid or os.getuid()}\n")
    (d / "cmdline").write_text(cmdline.replace(" ", "\x00") + ("\x00" if cmdline else ""))
    n = 3
    for text in fdinfo:
        (d / "fdinfo" / str(n)).write_text(text)
        n += 1


def drm_fd(bdf, vram_kib):
    return f"pos:\t0\ndrm-pdev:\t{bdf}\ndrm-memory-vram:\t{vram_kib} KiB\n"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gpulease-test-"))
        self.state = self.tmp / "state"
        self.state.mkdir()
        (self.state / "config.json").write_text(json.dumps({
            "gpus": list(range(8)),
            "pool": [4, 5, 6, 7],
            "reserved": {"0-3": "test"},
        }))
        self.proc = self.tmp / "proc"
        self.proc.mkdir()
        self.rocm_state = self.tmp / "rocm.json"
        self.rocm = self.tmp / "fake-rocm-smi"
        self.rocm.write_text(FAKE_ROCM_SRC)
        self.rocm.chmod(self.rocm.stat().st_mode | stat.S_IEXEC)
        self.env = os.environ.copy()
        self.env["GPULEASE_STATE_DIR"] = str(self.state)
        self.env["GPULEASE_ROCM_SMI"] = str(self.rocm)
        self.env["FAKE_ROCM_STATE"] = str(self.rocm_state)
        self.env.pop("GPULEASE_PROC_ROOT", None)
        self.set_rocm(vram={}, bus={})
        for k in ("GPULEASE_STATE_DIR", "GPULEASE_ROCM_SMI", "FAKE_ROCM_STATE", "GPULEASE_PROC_ROOT"):
            os.environ[k] = self.env.get(k, "")
        os.environ.pop("GPULEASE_PROC_ROOT", None)

    def tearDown(self):
        for k in ("GPULEASE_STATE_DIR", "GPULEASE_ROCM_SMI", "FAKE_ROCM_STATE", "GPULEASE_PROC_ROOT"):
            os.environ.pop(k, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def set_rocm(self, vram=None, bus=None):
        self.rocm_state.write_text(json.dumps({
            "vram": {str(k): v for k, v in (vram or {}).items()},
            "bus": {str(k): v for k, v in (bus or {}).items()},
        }))

    def cli(self, *args, **kw):
        return subprocess.run(
            [sys.executable, str(MODULE)] + list(args),
            env=self.env, capture_output=True, text=True, timeout=60, **kw,
        )

    def leases(self):
        out = {}
        for f in (self.state / "leases").glob("*.json"):
            rec = json.loads(f.read_text())
            out[rec["id"]] = rec
        return out


class TestParsing(Base):
    def test_parse_gpu_list(self):
        self.assertEqual(gpulease.parse_gpu_list("0-3,5"), [0, 1, 2, 3, 5])
        self.assertEqual(gpulease.parse_gpu_list("7"), [7])
        with self.assertRaises(ValueError):
            gpulease.parse_gpu_list("abc")


class TestAllocation(Base):
    def setUp(self):
        super().setUp()
        self.cfg = gpulease.load_config()

    def test_lowest_first_and_disjoint(self):
        rec1, fds1 = gpulease.try_allocate(2, self.cfg, "a", "run", 0, True)
        self.assertIsNotNone(rec1)
        self.assertEqual(rec1["gpus"], [4, 5])
        rec2, fds2 = gpulease.try_allocate(2, self.cfg, "b", "run", 0, True)
        self.assertIsNotNone(rec2)
        self.assertEqual(rec2["gpus"], [6, 7])
        rec3, fds3 = gpulease.try_allocate(1, self.cfg, "c", "run", 0, True, quiet=True)
        self.assertIsNone(rec3)

    def test_reserved_never_allocated(self):
        rec, fds = gpulease.try_allocate(4, self.cfg, "a", "run", 0, True)
        self.assertEqual(rec["gpus"], [4, 5, 6, 7])
        self.assertTrue(set(rec["gpus"]).isdisjoint({0, 1, 2, 3}))

    def test_idle_check_skips_busy_vram(self):
        self.set_rocm(vram={4: 80 * gpulease.GIB}, bus={})
        rec, fds = gpulease.try_allocate(1, self.cfg, "a", "run", 0, True)
        self.assertEqual(rec["gpus"], [5])

    def test_idle_check_disabled_uses_leases_only(self):
        self.set_rocm(vram={4: 80 * gpulease.GIB}, bus={})
        rec, fds = gpulease.try_allocate(1, self.cfg, "a", "run", 0, False)
        self.assertEqual(rec["gpus"], [4])

    def test_reap_removes_dead_holder(self):
        rec, fds = gpulease.try_allocate(1, self.cfg, "a", "run", 0, True)
        dead = dict(rec, id="dead-1", holder_pid=999999, created=time.time() - 300)
        gpulease.write_lease(dead)
        self.assertTrue(gpulease.read_lease("dead-1"))
        rc = self.cli("reap")
        self.assertIn("reaped", rc.stdout)
        self.assertIsNone(gpulease.read_lease("dead-1"))

    def test_pending_lease_grace(self):
        rec = {
            "id": "pending-1", "type": "run", "owner": "a", "gpus": [4],
            "pids": [], "holder_pid": None, "cmd": "", "created": time.time(),
            "expires": None, "heartbeat": time.time(),
        }
        self.assertTrue(gpulease.lease_active(rec, self.cfg, time.time()))
        rec["created"] = time.time() - 300
        self.assertFalse(gpulease.lease_active(rec, self.cfg, time.time()))

    def test_build_env(self):
        env = gpulease.build_env([4, 6], "lease-x")
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "4,6")
        self.assertNotIn("ROCR_VISIBLE_DEVICES", env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
        self.assertEqual(env["GPULEASE_LEASE_ID"], "lease-x")


class TestCLI(Base):
    def test_run_passthrough_and_cleanup(self):
        self.set_rocm(vram={}, bus={})
        r = self.cli("run", "-n", "1", "--", "sh", "-c", "echo hello; exit 7")
        self.assertEqual(r.returncode, 7)
        self.assertIn("hello", r.stdout)
        self.assertIn("gpus=[4]", r.stderr)
        self.assertEqual(self.leases(), {})

    def test_run_concurrent_disjoint(self):
        self.set_rocm(vram={}, bus={})
        p1 = subprocess.Popen(
            [sys.executable, str(MODULE), "run", "-n", "2", "--", "sleep", "3"],
            env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        p2 = subprocess.Popen(
            [sys.executable, str(MODULE), "run", "-n", "2", "--", "sleep", "3"],
            env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        deadline = time.time() + 20
        seen = []
        while time.time() < deadline:
            ls = [r for r in self.leases().values() if r["type"] == "run"]
            if len(ls) == 2:
                seen = [set(r["gpus"]) for r in ls]
                break
            time.sleep(0.2)
        p1.wait(timeout=30)
        p2.wait(timeout=30)
        self.assertEqual(len(seen), 2, "expected two concurrent run leases")
        self.assertTrue(seen[0].isdisjoint(seen[1]), f"overlap: {seen}")
        self.assertEqual(seen[0] | seen[1], {4, 5, 6, 7})

    def test_checkout_release(self):
        self.set_rocm(vram={}, bus={})
        r = self.cli("checkout", "-n", "1", "--ttl", "10")
        self.assertEqual(r.returncode, 0)
        exports = dict(
            line.split("=", 1) for line in r.stdout.strip().splitlines()
            if line.startswith("export ")
        )
        lease_id = exports["export GPULEASE_LEASE_ID"]
        self.assertEqual(exports["export HIP_VISIBLE_DEVICES"], "4")
        rec = gpulease.read_lease(lease_id)
        self.assertEqual(rec["gpus"], [4])
        self.assertTrue(gpulease.pid_alive(rec["holder_pid"]))
        r2 = self.cli("release", lease_id)
        self.assertEqual(r2.returncode, 0)
        self.assertIsNone(gpulease.read_lease(lease_id))
        self.assertFalse(gpulease.pid_alive(rec["holder_pid"]))

    def test_checkout_ttl_expiry(self):
        self.set_rocm(vram={}, bus={})
        r = self.cli("checkout", "-n", "1", "--ttl", "0.02")
        lease_id = [line for line in r.stdout.splitlines() if "GPULEASE_LEASE_ID" in line][0].split("=")[1]
        deadline = time.time() + 15
        while time.time() < deadline and gpulease.read_lease(lease_id):
            time.sleep(0.5)
        self.assertIsNone(gpulease.read_lease(lease_id))

    def test_status_json(self):
        self.set_rocm(vram={0: 280 * gpulease.GIB, 4: 50 * gpulease.GIB}, bus={})
        rec, fds = gpulease.try_allocate(1, gpulease.load_config(), "a", "run", 0, False)
        r = self.cli("status", "--json")
        data = json.loads(r.stdout)
        states = {row["gpu"]: row["state"] for row in data["gpus"]}
        self.assertEqual(states[0], "reserved-busy")
        self.assertEqual(states[4], "leased")
        g4 = [row for row in data["gpus"] if row["gpu"] == 4][0]
        self.assertEqual(g4["lease_type"], "run")

    def test_reserve_unreserve(self):
        r = self.cli("reserve", "2-3", "--for", "someone")
        self.assertEqual(r.returncode, 0)
        cfg = json.loads((self.state / "config.json").read_text())
        self.assertEqual(cfg["pool"], [4, 5, 6, 7])
        r = self.cli("unreserve", "0-3")
        cfg = json.loads((self.state / "config.json").read_text())
        self.assertEqual(cfg["pool"], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(cfg["reserved"], {})


class TestDefaults(Base):
    def test_auto_detects_pool(self):
        self.set_rocm(vram={i: 300 * 1024 * 1024 for i in range(8)}, bus={})
        self.env["GPULEASE_STATE_DIR"] = str(self.tmp / "fresh")
        r = self.cli("pool")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(data["gpus"], list(range(8)))
        self.assertEqual(data["pool"], list(range(8)))
        self.assertEqual(data["reserved"], {})

    def test_no_gpus_detected(self):
        self.set_rocm(vram={}, bus={})
        self.env["GPULEASE_STATE_DIR"] = str(self.tmp / "empty")
        r = self.cli("status")
        self.assertEqual(r.returncode, 2)
        self.assertIn("no GPUs detected", r.stderr)


class TestGpuSelection(Base):
    def setUp(self):
        super().setUp()
        self.cfg = gpulease.load_config()

    def test_prefer_free_granted(self):
        rec, fds = gpulease.try_allocate(1, self.cfg, "a", "run", 0, True, prefer=[6])
        self.assertEqual(rec["gpus"], [6])

    def test_prefer_falls_back_when_busy(self):
        gpulease.try_allocate(1, self.cfg, "a", "run", 0, True, required=[6])
        rec, fds = gpulease.try_allocate(1, self.cfg, "b", "run", 0, True, prefer=[6])
        self.assertEqual(rec["gpus"], [4])

    def test_prefer_partial_then_fill(self):
        rec, fds = gpulease.try_allocate(2, self.cfg, "a", "run", 0, True, prefer=[6])
        self.assertEqual(rec["gpus"], [6, 4])

    def test_prefer_order_preserved(self):
        rec, fds = gpulease.try_allocate(2, self.cfg, "a", "run", 0, True, prefer=[7, 6])
        self.assertEqual(rec["gpus"], [7, 6])

    def test_required_exact_order_preserved(self):
        rec, fds = gpulease.try_allocate(2, self.cfg, "a", "run", 0, True,
                                     required=[6, 4], requested="gpu=6,4")
        self.assertEqual(rec["gpus"], [6, 4])
        self.assertEqual(rec["requested"], "gpu=6,4")

    def test_required_all_or_nothing(self):
        gpulease.try_allocate(1, self.cfg, "a", "run", 0, True, required=[6])
        rec, fds = gpulease.try_allocate(2, self.cfg, "b", "run", 0, True, required=[5, 6])
        self.assertIsNone(rec)

    def test_cli_gpu_reserved_error(self):
        self.set_rocm(vram={}, bus={})
        r = self.cli("run", "-g", "2", "--wait", "0", "--", "true")
        self.assertEqual(r.returncode, 2)
        self.assertIn("not in pool", r.stderr)
        self.assertIn("reserved", r.stderr)

    def test_cli_n_conflict(self):
        r = self.cli("run", "-n", "1", "-g", "5,6", "--wait", "0", "--", "true")
        self.assertEqual(r.returncode, 2)
        self.assertIn("conflicts", r.stderr)

    def test_cli_mutual_exclusion(self):
        r = self.cli("run", "-g", "5", "-p", "6", "--wait", "0", "--", "true")
        self.assertEqual(r.returncode, 2)
        self.assertIn("not allowed with", r.stderr)

    def test_cli_gpu_infers_n_and_order(self):
        self.set_rocm(vram={}, bus={})
        r = self.cli("run", "-g", "6,5", "--no-idle-check", "--", "sleep", "0.1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("gpus=[6,5]", r.stderr)

    def test_cli_prefer_fallback_and_record(self):
        self.set_rocm(vram={}, bus={})
        gpulease.try_allocate(1, self.cfg, "a", "run", 0, True, required=[6])
        r = self.cli("run", "-p", "6", "--no-idle-check", "--", "sleep", "0.1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("gpus=[4]", r.stderr)

    def test_cli_prefer_ignores_reserved_with_warning(self):
        self.set_rocm(vram={}, bus={})
        r = self.cli("run", "-p", "2,4", "--wait", "0", "--", "true")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ignoring preferred", r.stderr)
        self.assertIn("gpus=[4]", r.stderr)


class TestAdoption(Base):
    def setUp(self):
        super().setUp()
        self.set_rocm(vram={}, bus={4: "0000:01:00.0", 5: "0000:02:00.0"})
        os.environ["GPULEASE_PROC_ROOT"] = str(self.proc)

    def test_scan_cluster_and_adopt(self):
        write_fake_proc(self.proc, 100, "vllm", 1, [], cmdline="vllm serve model")
        write_fake_proc(self.proc, 101, "VLLM::Worker_TP0", 100,
                        [drm_fd("0000:01:00.0", 88634240)])
        write_fake_proc(self.proc, 102, "VLLM::Worker_TP1", 100,
                        [drm_fd("0000:02:00.0", 88634240)])
        usage, own_vram = gpulease.scan_own_usage(2 * gpulease.GIB)
        self.assertEqual(set(usage), {101, 102})
        clusters = gpulease.build_clusters(usage)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["gpus"], [4, 5])
        self.assertEqual(clusters[0]["pids"], [101, 102])
        self.assertIn("vllm serve model", clusters[0]["cmd"])
        leases = {}
        events = gpulease.reconcile_adopted(leases, clusters, time.time(), "adopted")
        self.assertEqual([e[0] for e in events], ["adopt"])
        adopted = list(leases.values())[0]
        self.assertEqual(adopted["gpus"], [4, 5])
        self.assertTrue(gpulease.lease_active(adopted, gpulease.load_config(), time.time()))

    def test_adopted_below_threshold_ignored(self):
        write_fake_proc(self.proc, 101, "tiny", 1, [drm_fd("0000:01:00.0", 97)])
        usage, own_vram = gpulease.scan_own_usage(2 * gpulease.GIB)
        self.assertEqual(usage, {})

    def test_release_when_pids_die(self):
        write_fake_proc(self.proc, 101, "Worker", 1, [drm_fd("0000:01:00.0", 88634240)])
        usage, own_vram = gpulease.scan_own_usage(2 * gpulease.GIB)
        leases = {}
        gpulease.reconcile_adopted(leases, gpulease.build_clusters(usage), time.time(), "adopted")
        self.assertEqual(len(leases), 1)
        shutil.rmtree(self.proc / "101")
        usage, own_vram = gpulease.scan_own_usage(2 * gpulease.GIB)
        events = gpulease.reconcile_adopted(leases, gpulease.build_clusters(usage), time.time(), "adopted")
        self.assertEqual([e[0] for e in events], ["release"])
        self.assertEqual(leases, {})

    def test_no_double_adopt_for_run_lease(self):
        now = time.time()
        holder = {
            "id": "run-1", "type": "run", "owner": "a", "gpus": [4], "pids": [],
            "holder_pid": 100, "cmd": "", "created": now, "expires": None,
            "heartbeat": now,
        }
        write_fake_proc(self.proc, 100, "gpulease-child", 1, [], cmdline="vllm serve model")
        write_fake_proc(self.proc, 101, "Worker", 100, [drm_fd("0000:01:00.0", 88634240)])
        cfg = gpulease.load_config()
        leases = {holder["id"]: holder}
        usage, own_vram = gpulease.scan_own_usage(2 * gpulease.GIB)
        holders = gpulease.live_holder_pids(leases, cfg, now)
        self.assertEqual(holders, {100})
        clusters = gpulease.build_clusters(gpulease.filter_covered(usage, holders))
        self.assertEqual(clusters, [])
        events = gpulease.reconcile_adopted(leases, clusters, now, "adopted")
        self.assertEqual(events, [])
        self.assertEqual(len(leases), 1)

    def test_foreign_detection(self):
        cfg = gpulease.load_config()
        leases = {}
        own_vram = {}
        card_vram = {4: 100 * gpulease.GIB, 5: 300 * 1024 * 1024}
        events = gpulease.reconcile_foreign(leases, cfg, own_vram, card_vram, time.time())
        self.assertEqual([e[0] for e in events], ["foreign"])
        rec = list(leases.values())[0]
        self.assertEqual(rec["gpus"], [4])
        card_vram = {4: 300 * 1024 * 1024, 5: 300 * 1024 * 1024}
        rec["heartbeat"] = time.time() - 999
        events = gpulease.reconcile_foreign(leases, cfg, own_vram, card_vram, time.time())
        self.assertEqual([e[0] for e in events], ["release"])
        self.assertEqual(leases, {})

    def test_explained_vram_not_foreign(self):
        write_fake_proc(self.proc, 101, "Worker", 1, [drm_fd("0000:01:00.0", 88634240)])
        usage, own_vram = gpulease.scan_own_usage(2 * gpulease.GIB)
        card_vram = {4: own_vram[4] + 300 * 1024 * 1024}
        leases = {}
        events = gpulease.reconcile_foreign(leases, gpulease.load_config(), own_vram, card_vram, time.time())
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
