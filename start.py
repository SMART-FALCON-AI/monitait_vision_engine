#!/usr/bin/env python3
"""
MonitaQC Startup Script
Auto-detects OS and hardware, configures docker-compose accordingly.
  - Linux (production): DATA_ROOT=/mnt/SSD-RESERVE, PRIVILEGED=true
  - Windows/macOS (dev): DATA_ROOT=. (project dir), PRIVILEGED=false
  - Auto-tunes YOLO replicas/workers, SHM, Redis memory from CPU/RAM/GPU
"""
import os
import sys
import platform
import subprocess
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(PROJECT_DIR, ".env")


def _detect_hardware():
    """Detect CPU cores, RAM, and GPU count on the host machine."""
    # CPU
    cpu_count = os.cpu_count() or 4

    # RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # Fallback without psutil
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            ram_gb = int(line.split()[1]) / (1024 ** 2)
                            break
                    else:
                        ram_gb = 8.0
            except Exception:
                ram_gb = 8.0
        else:
            ram_gb = 8.0

    # GPU (nvidia-smi) — detect count and per-GPU VRAM
    gpu_count = 0
    gpu_vram_mb = 0  # VRAM of first GPU in MB
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            lines = r.stdout.strip().split("\n")
            gpu_count = len(lines)
            # Parse VRAM from first GPU: "NVIDIA GeForce RTX 3050, 8192"
            try:
                gpu_vram_mb = int(lines[0].split(",")[-1].strip())
            except (ValueError, IndexError):
                gpu_vram_mb = 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return cpu_count, ram_gb, gpu_count, gpu_vram_mb


def _compute_resources(cpu, ram_gb, gpu_count, gpu_vram_mb=0):
    """Compute optimal docker-compose resource settings from hardware.

    GPU strategy: each YOLO worker loads the model into ~500MB VRAM.
    Use 80% of VRAM, split across replicas (1 per GPU).
    More workers = more concurrent inference = higher throughput.
    """
    has_gpu = gpu_count > 0

    if has_gpu:
        # Replicas = 1 per GPU (Docker assigns GPUs to containers)
        # Workers = fill VRAM within each replica (~500MB per worker, use 80%)
        yolo_replicas = max(1, gpu_count)
        if gpu_vram_mb > 0:
            usable_vram = int(gpu_vram_mb * 0.80)
            yolo_workers = max(2, usable_vram // 500)
        else:
            yolo_workers = 2  # fallback if VRAM unknown
        shm_gb = min(4, max(1, int(ram_gb // 4)))
    else:
        # CPU-only: spread YOLO processes across cores, leave ~40% for vision engine
        yolo_replicas = max(1, cpu // 6)
        yolo_workers = max(1, cpu // (yolo_replicas * 3))
        shm_gb = min(2, max(1, int(ram_gb // 8)))

    # Redis: ~5% of RAM, minimum 256MB
    redis_mb = max(256, int(ram_gb * 50))

    return {
        "YOLO_REPLICAS": str(yolo_replicas),
        "YOLO_WORKERS": str(yolo_workers),
        "SHM_SIZE": f"{shm_gb}g",
        "REDIS_MAXMEMORY": str(redis_mb),
    }


def detect_and_write_env():
    system = platform.system()

    if system == "Linux":
        data_root = "/mnt/SSD-RESERVE"
        privileged = "true"
        mode = "PRODUCTION (Linux)"
    else:
        data_root = "."
        privileged = "false"
        mode = f"DEVELOPMENT ({system})"

    # Detect hardware
    cpu, ram_gb, gpu_count, gpu_vram_mb = _detect_hardware()
    resources = _compute_resources(cpu, ram_gb, gpu_count, gpu_vram_mb)

    vram_str = f", {gpu_vram_mb}MB VRAM" if gpu_vram_mb else ""
    print(f"[MonitaQC] Hardware: {cpu} CPU cores, {ram_gb:.1f} GB RAM, {gpu_count} GPU(s){vram_str}")
    print(f"[MonitaQC] Auto-tuned: {resources['YOLO_REPLICAS']} YOLO replicas x "
          f"{resources['YOLO_WORKERS']} workers, SHM={resources['SHM_SIZE']}, "
          f"Redis={resources['REDIS_MAXMEMORY']}MB")

    # Read existing .env to preserve user-defined vars
    existing = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    existing[key.strip()] = val.strip()

    # Only set if not already defined by user
    if "DATA_ROOT" not in existing:
        existing["DATA_ROOT"] = data_root
    if "PRIVILEGED" not in existing:
        existing["PRIVILEGED"] = privileged

    # Always update hardware-detected values (re-detect on every start)
    for key, val in resources.items():
        existing[key] = val

    with open(ENV_FILE, "w") as f:
        f.write(f"# Auto-generated by start.py - detected: {mode}\n")
        f.write(f"# Hardware: {cpu} cores, {ram_gb:.1f}GB RAM, {gpu_count} GPU(s)\n")
        f.write(f"# Override DATA_ROOT/PRIVILEGED manually if needed\n")
        f.write(f"# YOLO_REPLICAS, YOLO_WORKERS, SHM_SIZE, REDIS_MAXMEMORY are auto-detected\n")
        for key, val in existing.items():
            f.write(f"{key}={val}\n")

    print(f"[MonitaQC] Mode: {mode}")
    print(f"[MonitaQC] DATA_ROOT={existing['DATA_ROOT']}")
    print(f"[MonitaQC] .env written to {ENV_FILE}")

    # Create local volume dirs if using dev mode
    if existing["DATA_ROOT"] == ".":
        for d in ["volumes/redis", "volumes/timescaledb", "volumes/grafana",
                   "volumes/pigallery2_config", "volumes/pigallery2_db", "raw_images"]:
            path = os.path.join(PROJECT_DIR, d)
            os.makedirs(path, exist_ok=True)
        print("[MonitaQC] Local volume directories created")


def main():
    detect_and_write_env()

    # Pass any extra args to docker compose (e.g. "start.py up -d", "start.py down")
    compose_args = sys.argv[1:] if len(sys.argv) > 1 else ["up", "-d"]

    cmd = ["docker", "compose", "-f", os.path.join(PROJECT_DIR, "docker-compose.yml")] + compose_args
    print(f"[MonitaQC] Running: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd, cwd=PROJECT_DIR))


if __name__ == "__main__":
    main()
