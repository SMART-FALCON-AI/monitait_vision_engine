#!/usr/bin/env bash
# ============================================================
# MonitaQC Setup (interactive installer)
#
# Two ways to run:
#
#   1. Terminal:
#        sudo bash setup.sh
#        sudo bash setup.sh --yes           # accept all defaults
#        sudo bash setup.sh [install_dir]   # positional override
#
#   2. Double-click (Ubuntu file manager):
#        chmod +x setup.sh                  (one time — see note below)
#        then double-click -> "Run as a Program"
#        A terminal opens, sudo prompts for your password, the
#        interactive installer runs inside it.
#
# NOTE: On modern Ubuntu, right-click -> "Run as a Program" is
# required (Nautilus no longer launches .sh on plain double-click).
# To make plain double-click work project-wide:
#     gsettings set org.gnome.nautilus.preferences \
#       executable-text-activation 'launch'
#
# What this script does:
#   1. Prerequisites (Docker Engine + Docker Compose v2)
#   2. Image acquisition — for each image, tries in order:
#        a) private registry (registry.monitait.com, no auth)
#        b) Docker Hub (public upstream)
#        c) local build from source (only for images in BUILDABLE)
#   3. Project install + first-time start
#
# Env overrides (pre-fill prompts; still confirmable unless --yes):
#   INSTALL_DIR                          default: ~<user>/projects/mve
#   REGISTRY=host[:port]                 default: registry.monitait.com
#   REGISTRY_INSECURE=0|1                allow plain-HTTP registry
# ============================================================
set -euo pipefail

# ------------------------------------------------------------
# Double-click relaunch: if invoked without a controlling TTY
# but a graphical session is present, re-launch inside a
# terminal so the interactive prompts are visible. Uses sudo
# (not pkexec) so the user sees the familiar password prompt.
# ------------------------------------------------------------
if [ ! -t 0 ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] \
        && [ -z "${MONITAQC_RELAUNCHED:-}" ]; then
    SELF="$(readlink -f "$0")"
    INNER="MONITAQC_RELAUNCHED=1 sudo bash \"$SELF\"; echo; read -r -p 'Press Enter to close...' _"
    for T in gnome-terminal tilix mate-terminal konsole xfce4-terminal x-terminal-emulator; do
        if command -v "$T" &>/dev/null; then
            case "$T" in
                gnome-terminal|tilix|mate-terminal)
                    exec "$T" -- bash -c "$INNER" ;;
                konsole)
                    exec "$T" -e bash -c "$INNER" ;;
                *)
                    exec "$T" -e bash -c "$INNER" ;;
            esac
        fi
    done
    # No terminal found — surface an error via zenity/notify if possible.
    MSG="No terminal emulator found. Open a terminal and run:  sudo bash $SELF"
    if command -v zenity &>/dev/null; then zenity --error --text="$MSG"
    elif command -v notify-send &>/dev/null; then notify-send "MonitaQC Setup" "$MSG"
    fi
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve the *invoking* user's home — when run under sudo, $HOME is /root.
# Install dir defaults into that user's ~/projects/mve so files stay user-owned.
REAL_USER="${SUDO_USER:-${USER:-root}}"
REAL_HOME="$(getent passwd "$REAL_USER" 2>/dev/null | cut -d: -f6)"
REAL_HOME="${REAL_HOME:-/root}"
DEFAULT_INSTALL_DIR="$REAL_HOME/projects/mve"

# ---------- arg parsing ----------
ASSUME_YES=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        -y|--yes) ASSUME_YES=1 ;;
        -h|--help)
            sed -n '2,37p' "$0"; exit 0 ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

INSTALL_DIR="${INSTALL_DIR:-${POSITIONAL[0]:-$DEFAULT_INSTALL_DIR}}"
REGISTRY="${REGISTRY:-registry.monitait.com}"
REGISTRY_INSECURE="${REGISTRY_INSECURE:-0}"

# ---------- images ----------
IMAGES=(
    "monitait/mve:latest"
    "redis:7-alpine"
    "timescale/timescaledb:latest-pg15"
    "grafana/grafana:latest"
    "bpatrik/pigallery2:latest"
)

# image_tag -> build_context_dir (relative to project root = $SCRIPT_DIR)
declare -A BUILDABLE=(
    ["monitait/mve:latest"]="vision_engine"
)

# ---------- tty helpers ----------
log()  { echo -e "\n\033[1;36m[setup]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*" >&2; }
die()  { echo -e "\033[1;31m[error]\033[0m $*" >&2; exit 1; }
hr()   { echo -e "\033[2m------------------------------------------------------------\033[0m"; }

is_interactive() { [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ]; }

prompt() {
    local __var=$1 __q=$2 __def=$3 __ans
    if ! is_interactive; then
        printf -v "$__var" '%s' "$__def"
        echo "  $__q [$__def]  (auto)"
        return
    fi
    read -r -p "  $__q [$__def]: " __ans </dev/tty || __ans=""
    printf -v "$__var" '%s' "${__ans:-$__def}"
}

confirm() {
    local q=$1 def=${2:-N} ans hint
    if [ "${def^^}" = "Y" ]; then hint="[Y/n]"; else hint="[y/N]"; fi
    if ! is_interactive; then
        echo "  $q $hint  (auto: $def)"
        [ "${def^^}" = "Y" ]
        return
    fi
    read -r -p "  $q $hint: " ans </dev/tty || ans=""
    ans=${ans:-$def}
    [[ "${ans,,}" =~ ^y ]]
}

require_root() {
    if [ "$(id -u)" -ne 0 ]; then
        cat >&2 <<EOF
[error] This installer must be run as root.

  Please re-run with sudo:
      sudo bash $0

  (Docker install, /etc/docker config, and system service setup all
   require root privileges.)
EOF
        exit 1
    fi
}

# ---------- 0. Interactive config ----------
gather_config() {
    hr
    echo " MonitaQC Setup"
    if is_interactive; then
        echo " Interactive mode — press Enter to accept [defaults]."
    else
        echo " Non-interactive mode — using defaults/env overrides."
    fi
    hr

    prompt INSTALL_DIR       "Install directory"                    "$INSTALL_DIR"
    prompt REGISTRY          "Private Docker registry"              "$REGISTRY"
    local insec_default="N"; [ "$REGISTRY_INSECURE" = "1" ] && insec_default="Y"
    if confirm "Registry uses plain HTTP (insecure)?" "$insec_default"; then
        REGISTRY_INSECURE=1
    else
        REGISTRY_INSECURE=0
    fi

    echo
    echo " Summary:"
    echo "   Install dir:       $INSTALL_DIR"
    echo "   Registry:          $REGISTRY"
    echo "   Insecure registry: $([ "$REGISTRY_INSECURE" = "1" ] && echo yes || echo no)"
    echo "   Images to acquire: ${#IMAGES[@]} (registry -> Docker Hub -> local build)"
    hr
    if ! confirm "Proceed with install?" "Y"; then
        die "Aborted by user."
    fi
}

# ---------- 1. Prerequisites ----------
install_docker() {
    log "Installing Docker Engine (via get.docker.com convenience script)..."
    if is_interactive && ! confirm "Install Docker now?" "Y"; then
        die "Docker is required. Install manually and re-run."
    fi
    if command -v apt-get &>/dev/null; then
        apt-get update -y
        apt-get install -y ca-certificates curl gnupg
    elif command -v dnf &>/dev/null; then
        dnf install -y ca-certificates curl gnupg
    fi
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sh /tmp/get-docker.sh
    systemctl enable --now docker
}

ensure_prereqs() {
    log "Checking prerequisites..."
    if ! command -v docker &>/dev/null; then
        install_docker
    else
        echo "  docker: $(docker --version)"
    fi
    if ! docker compose version &>/dev/null; then
        if command -v apt-get &>/dev/null; then
            apt-get install -y docker-compose-plugin || die "Install docker-compose-plugin manually"
        elif command -v dnf &>/dev/null; then
            dnf install -y docker-compose-plugin || die "Install docker-compose-plugin manually"
        else
            die "docker compose v2 missing and no supported package manager found"
        fi
    fi
    echo "  compose: $(docker compose version)"

    if [ "$REGISTRY_INSECURE" = "1" ]; then
        local daemon=/etc/docker/daemon.json
        mkdir -p /etc/docker
        if ! grep -q "\"$REGISTRY\"" "$daemon" 2>/dev/null; then
            log "Adding $REGISTRY to insecure-registries"
            [ -f "$daemon" ] && cp "$daemon" "$daemon.bak.$(date +%s)"
            cat >"$daemon" <<EOF
{
  "insecure-registries": ["$REGISTRY"]
}
EOF
            systemctl restart docker
        fi
    fi
}

# ---------- 2. Image acquisition ----------
have_image() { docker image inspect "$1" &>/dev/null; }

try_private_registry() {
    local img="$1"
    local remote="$REGISTRY/$img"
    if docker pull "$remote" 2>/dev/null; then
        docker tag "$remote" "$img"
        docker rmi "$remote" &>/dev/null || true
        return 0
    fi
    return 1
}

try_dockerhub() {
    docker pull "$1" 2>/dev/null
}

try_local_build() {
    local img="$1"
    local ctx="${BUILDABLE[$img]:-}"
    [ -n "$ctx" ] || return 1
    [ -d "$SCRIPT_DIR/$ctx" ] || return 1
    log "Building $img from $SCRIPT_DIR/$ctx ..."
    docker build -t "$img" "$SCRIPT_DIR/$ctx"
}

acquire_images() {
    log "Acquiring images ($REGISTRY -> Docker Hub -> local build)..."
    local failed=()

    for img in "${IMAGES[@]}"; do
        if have_image "$img"; then
            echo "  [ok]        $img  (already present)"
            continue
        fi

        if try_private_registry "$img"; then
            echo "  [registry]  $img  <- $REGISTRY"
            continue
        fi

        warn "  registry miss for $img — trying Docker Hub..."
        if try_dockerhub "$img"; then
            echo "  [dockerhub] $img"
            continue
        fi

        if [ -n "${BUILDABLE[$img]:-}" ]; then
            warn "  Docker Hub miss for $img — falling back to local build..."
            if try_local_build "$img"; then
                echo "  [built]     $img"
                continue
            fi
        fi

        failed+=("$img")
    done

    if [ ${#failed[@]} -gt 0 ]; then
        die "Could not acquire images: ${failed[*]}"
    fi
}

# ---------- 3. Install project ----------
install_project() {
    # If the user picked the same dir as the source, skip the copy.
    if [ "$(readlink -f "$INSTALL_DIR" 2>/dev/null)" = "$SCRIPT_DIR" ]; then
        log "Install dir is the source dir — skipping copy."
    else
        log "Installing project to $INSTALL_DIR ..."
        if [ -d "$INSTALL_DIR" ] && [ -n "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]; then
            warn "$INSTALL_DIR already exists and is not empty."
            if ! confirm "Overwrite project files there (data volumes preserved)?" "N"; then
                die "Aborted — pick a different install dir."
            fi
        fi
        mkdir -p "$INSTALL_DIR"
        # Copy everything except git/caches/volume data/the installer itself.
        rsync -a \
            --exclude='.git/' \
            --exclude='__pycache__/' \
            --exclude='*.pyc' \
            --exclude='.env' \
            --exclude='volumes/redis/' \
            --exclude='volumes/timescaledb/' \
            --exclude='volumes/grafana/' \
            --exclude='volumes/pigallery2_config/' \
            --exclude='volumes/pigallery2_db/' \
            "$SCRIPT_DIR/" "$INSTALL_DIR/"
    fi

    for d in volumes/redis volumes/timescaledb volumes/grafana \
             volumes/pigallery2_config volumes/pigallery2_db \
             volumes/weights raw_images; do
        mkdir -p "$INSTALL_DIR/$d"
    done

    # Installer runs as root; hand ownership back to the invoking user.
    if [ "$REAL_USER" != "root" ] && id "$REAL_USER" &>/dev/null; then
        chown -R "$REAL_USER:$REAL_USER" "$INSTALL_DIR"
        echo "  Ownership set to $REAL_USER"
    fi
}

start_stack() {
    log "Starting MonitaQC ..."
    cd "$INSTALL_DIR"
    if ! confirm "Start the stack now (docker compose up -d)?" "Y"; then
        echo "  Skipped. Start later with:  cd $INSTALL_DIR && python3 deploy/start.py"
        return
    fi
    python3 deploy/start.py up -d
}

# ---------- main ----------
require_root
gather_config
ensure_prereqs
acquire_images
install_project
start_stack

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
cat <<EOF

===========================================
 MonitaQC installed at: ${INSTALL_DIR}
 Dashboard: http://${IP:-localhost}:80
===========================================

 Useful commands:
   cd ${INSTALL_DIR}
   python3 deploy/start.py          # Start (auto-detects hardware)
   docker compose logs -f    # View logs
   docker compose down       # Stop
   docker compose restart    # Restart

EOF
