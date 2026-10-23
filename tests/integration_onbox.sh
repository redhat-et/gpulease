#!/usr/bin/env bash
# Copyright 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
GPULEASE="${GPULEASE:-$here/../gpulease.py}"
export GPULEASE_STATE_DIR="$(mktemp -d /tmp/gpulease-it.XXXXXX)"
trap 'rm -rf "$GPULEASE_STATE_DIR"' EXIT

mkdir -p "$GPULEASE_STATE_DIR"
cat > "$GPULEASE_STATE_DIR/config.json" <<'EOF'
{
  "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
  "pool": [4, 5, 6, 7],
  "reserved": {"0-3": "test"}
}
EOF

say() { echo "== $*"; }
fail() { echo "FAIL: $*"; exit 1; }

say "pool"
"$GPULEASE" pool >/dev/null || fail "pool"

say "status --json shape"
"$GPULEASE" status --json > "$GPULEASE_STATE_DIR/status.json" || fail "status"
python3 - "$GPULEASE_STATE_DIR/status.json" <<'EOF' || fail "status shape"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["pool"] == [4, 5, 6, 7], d["pool"]
assert "0-3" in d["reserved"], d["reserved"]
gpus = {r["gpu"]: r for r in d["gpus"]}
assert set(gpus) == set(range(8)), gpus.keys()
reserved = [r for r in d["gpus"] if r["reserved"]]
assert sorted(r["gpu"] for r in reserved) == [0, 1, 2, 3]
busy = [r for r in d["gpus"] if r["state"] in ("busy", "reserved-busy")]
print("  8 gpus; reserved 0-3; busy:", sorted(r["gpu"] for r in busy))
EOF

say "concurrent run leases are disjoint (bookkeeping only, --no-idle-check)"
"$GPULEASE" run -q --no-idle-check -n 2 -- sleep 5 & p1=$!
"$GPULEASE" run -q --no-idle-check -n 2 -- sleep 5 & p2=$!
sleep 3
python3 - "$GPULEASE_STATE_DIR" <<'EOF' || fail "concurrent disjoint"
import glob, json, os, sys
st = sys.argv[1]
leases = [json.load(open(f)) for f in glob.glob(os.path.join(st, "leases", "*.json"))]
runs = [l for l in leases if l["type"] == "run"]
assert len(runs) == 2, f"expected 2 run leases, got {len(runs)}: {leases}"
a, b = set(runs[0]["gpus"]), set(runs[1]["gpus"])
assert not (a & b), f"overlap: {a} {b}"
assert a | b == {4, 5, 6, 7}, (a, b)
print(f"  disjoint: {sorted(a)} vs {sorted(b)}")
EOF
wait "$p1"; wait "$p2"

say "kill -9 releases lease"
"$GPULEASE" run -q --no-idle-check -n 1 -- sleep 300 &
pid=$!
sleep 3
child=$(pgrep -P "$pid" || true)
kill -9 "$pid" 2>/dev/null
sleep 1
[ -n "$child" ] && kill -9 "$child" 2>/dev/null
nrun=$("$GPULEASE" status --json | python3 -c 'import json,sys; print(len([l for l in json.load(sys.stdin)["leases"] if l["type"]=="run"]))')
[ "$nrun" = "0" ] || fail "stale run lease after kill -9 (count=$nrun)"
echo "  lease reaped after SIGKILL"

say "checkout / env / release"
out=$("$GPULEASE" checkout -q --no-idle-check -n 1 --ttl 5) || fail "checkout"
lid=$(echo "$out" | sed -n 's/^export GPULEASE_LEASE_ID=//p')
[ -n "$lid" ] || fail "no lease id in output"
"$GPULEASE" env "$lid" | grep -q "^export HIP_VISIBLE_DEVICES=4$" || fail "env"
"$GPULEASE" release "$lid" || fail "release"
left=$(ls "$GPULEASE_STATE_DIR/leases" 2>/dev/null | wc -l)
[ "$left" = "0" ] || fail "leases dir not empty: $left"
echo "  checkout+release OK ($lid)"

say "adoption of live unmanaged jobs (read-only)"
"$GPULEASE" watch --once || fail "watch --once"
python3 - "$GPULEASE_STATE_DIR" <<'EOF' || fail "adoption"
import glob, json, os, sys
st = sys.argv[1]
leases = [json.load(open(f)) for f in glob.glob(os.path.join(st, "leases", "*.json"))]
ad = [l for l in leases if l["type"] == "adopted"]
assert ad, f"no adopted lease found: {leases}"
gpus = sorted({g for l in ad for g in l["gpus"]})
print(f"  adopted gpus={gpus} pids={[p for l in ad for p in l['pids']][:8]}")
print(f"  cmd={ad[0]['cmd'][:80]!r}")
assert 4 in gpus or 5 in gpus or 6 in gpus or 7 in gpus, gpus
EOF

say "ALL INTEGRATION CHECKS PASSED"
