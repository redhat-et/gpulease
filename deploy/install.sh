#!/usr/bin/env bash
# Copyright 2026 Red Hat, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
bin="$HOME/.local/bin"
unit_dir="$HOME/.config/systemd/user"

mkdir -p "$bin" "$unit_dir"
install -m 0755 "$here/../gpulease.py" "$bin/gpulease"
install -m 0644 "$here/gpulease-watch.service" "$unit_dir/gpulease-watch.service"

if systemctl --user daemon-reload 2>/dev/null && systemctl --user enable --now gpulease-watch 2>/dev/null; then
    echo "gpulease installed to $bin/gpulease; gpulease-watch enabled via systemd --user"
else
    echo "gpulease installed to $bin/gpulease"
    echo "systemd --user unavailable; start watcher manually:"
    echo "  nohup $bin/gpulease watch >>$HOME/.local/state/gpulease/watch.log 2>&1 &"
fi

"$bin/gpulease" pool
