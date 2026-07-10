#!/bin/sh
# v4.0.81 — MVE container entrypoint.
#
# Runs the equivalent of `python3 main.py`, but FIRST checks whether any
# offline-required wheels are installed and pip-installs them from the
# bind-mounted /wheels dir if missing. Currently that means pypylon
# (Basler USB3 Vision SDK), which the target site's container can't
# download because files.pythonhosted.org is unreachable from the
# operator's network — see docker-compose.yml `wheels` bind-mount.
#
# The install goes into the container's writable overlay, so:
#   - Survives `docker restart monitait_vision_engine` (no re-install).
#   - LOST on `docker compose up --force-recreate` and image rebuild —
#     which is exactly when this script re-runs and re-installs from
#     the bind-mounted wheel. Fully self-healing across every recreate.
#
# Wheels dir layout (relative to compose file root):
#   wheels/pypylon-*.whl        — required for Basler pro cameras
#   wheels/README                — how to download / update wheels
#
# If /wheels is empty or the wheel is missing, we log a warning and
# continue booting — pypylon is opt-in per site (only pro-camera sites
# need it), so its absence must not stop MVE from serving USB / IP
# cameras. Basler enumerate paths handle ImportError and return an
# empty device list.

set -e

log() { echo "[startup] $*"; }

# Only try to install if the /wheels bind-mount actually landed.
if [ -d /wheels ]; then
    for wheel in /wheels/pypylon-*.whl; do
        [ -f "$wheel" ] || { log "no pypylon wheel in /wheels — skipping (Basler features will be disabled)"; break; }
        if python3 -c 'import pypylon' 2>/dev/null; then
            log "pypylon already installed — skipping"
            break
        fi
        log "installing $wheel offline"
        # --no-index refuses pypi (we can't reach it anyway); --no-deps
        # skips dependency resolution (the wheel is self-contained).
        # Failure here is non-fatal — MVE boots without Basler support.
        pip install --no-index --no-deps "$wheel" 2>&1 | tail -3 || {
            log "pypylon install FAILED — MVE will boot without Basler support"
        }
        break  # first pypylon-*.whl wins
    done
else
    log "/wheels not mounted — Basler features will be disabled if pypylon is not in the image"
fi

log "exec python3 main.py"
exec python3 main.py
