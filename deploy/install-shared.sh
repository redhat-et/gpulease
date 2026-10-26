#!/usr/bin/env bash
# Copyright 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

STATE_DIR="${GPULEASE_SYSTEM_STATE_DIR:-/var/lib/gpulease}"
GROUP="gpulease"
USERS=()
CLAIMS=()

usage() {
    echo "usage: $0 [--claim SPEC:USER] USER [USER...]" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --claim) CLAIMS+=("${2:-}"); shift 2 ;;
        --user) USERS+=("${2:-}"); shift 2 ;;
        -h|--help) usage ;;
        *) USERS+=("$1"); shift ;;
    esac
done

[ "$(id -u)" -eq 0 ] || { echo "run as root" >&2; exit 1; }
[ "${#USERS[@]}" -gt 0 ] || usage

if ! command -v setfacl >/dev/null 2>&1; then
    if command -v dnf >/dev/null 2>&1; then
        dnf install -y acl
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update && apt-get install -y acl
    elif command -v zypper >/dev/null 2>&1; then
        zypper --non-interactive install acl
    else
        echo "install the 'acl' package (setfacl) first" >&2
        exit 1
    fi
fi

getent group "$GROUP" >/dev/null || groupadd -r "$GROUP"
for user in "${USERS[@]}"; do
    id -u "$user" >/dev/null 2>&1 || { echo "no such user: $user" >&2; exit 1; }
    usermod -aG "$GROUP" "$user"
done

install -d -o root -g "$GROUP" -m 2770 \
    "$STATE_DIR" "$STATE_DIR/locks" "$STATE_DIR/leases" "$STATE_DIR/releases"
touch "$STATE_DIR/.shared"
chown root:"$GROUP" "$STATE_DIR/.shared"
chmod 0640 "$STATE_DIR/.shared"

python3 - "$STATE_DIR" "${CLAIMS[@]}" <<'PY'
import json
import os
import sys

state = sys.argv[1]
claims = {}
for item in sys.argv[2:]:
    spec, sep, who = item.partition(":")
    if not sep or not spec or not who:
        raise SystemExit(f"bad --claim {item!r}; expected SPEC:USER")
    claims[spec] = who

path = os.path.join(state, "config.json")
if os.path.exists(path):
    print(f"note: {path} exists; claims were not modified")
else:
    with open(path, "w") as f:
        json.dump({"claims": claims}, f, indent=2, sort_keys=True)
        f.write("\n")
PY
chown root:"$GROUP" "$STATE_DIR/config.json"
chmod 0660 "$STATE_DIR/config.json"

for user in "${USERS[@]}"; do
    for dir in "$STATE_DIR" "$STATE_DIR/locks" "$STATE_DIR/leases" "$STATE_DIR/releases"; do
        setfacl -m "u:$user:rwx" "$dir"
        setfacl -m "d:u:$user:rwx" "$dir"
    done
    setfacl -m "u:$user:rw" "$STATE_DIR/config.json"
done

if command -v restorecon >/dev/null 2>&1; then
    restorecon -R "$STATE_DIR" >/dev/null 2>&1 || true
fi

echo "shared state: $STATE_DIR (root:$GROUP, 2770)"
echo "users: ${USERS[*]}"
echo "claims: ${CLAIMS[*]:-none}"
echo
echo "next:"
echo "  * per user: bash deploy/install.sh"
echo "  * always-on watchers: sudo loginctl enable-linger <user>"
echo "  * new logins pick up the $GROUP group; ACLs already grant access"
