#!/usr/bin/env python3
"""
NVR-Style Disk Cleanup Service for MonitaQC

Instead of scanning individual files, this service deletes entire hourly
time-chunk directories (oldest first). This mirrors how DVR/NVR systems
manage storage — O(1) deletion per chunk, no file-by-file scanning.

Directory structure expected:
    /data/{shipment_id}/{YYYY-MM-DD_HH}/  ← hourly chunks
        image1.jpg
        image2.jpg
        ...
"""
import os
import shutil
import time
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MONITOR_DIR = os.getenv("MONITOR_DIR", "/data")
MAX_USAGE_PERCENT = int(os.getenv("MAX_USAGE_PERCENT", 90))
MIN_USAGE_PERCENT = int(os.getenv("MIN_USAGE_PERCENT", 80))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 10))
MIN_CHUNK_AGE_HOURS = int(os.getenv("MIN_CHUNK_AGE_HOURS", 1))

# Metrics
cleanup_runs = 0
total_chunks_deleted = 0
total_space_freed_gb = 0.0


def get_disk_usage() -> Tuple[int, int, int]:
    """Returns (usage_percent, used_gb, total_gb)."""
    try:
        usage = shutil.disk_usage(MONITOR_DIR)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        pct = int((usage.used / usage.total) * 100) if usage.total > 0 else 100
        return pct, round(used_gb, 1), round(total_gb, 1)
    except Exception as e:
        logger.error(f"Disk usage error: {e}")
        return 100, 0, 0


def get_time_chunks() -> List[Tuple[str, Path]]:
    """
    Find all hourly time-chunk directories, sorted oldest first.

    Scans: /data/{shipment}/{YYYY-MM-DD_HH}/
    Returns list of (sort_key, chunk_path) tuples.
    The sort key is the chunk directory name (YYYY-MM-DD_HH) which sorts chronologically.
    """
    chunks = []
    monitor = Path(MONITOR_DIR)

    try:
        # Walk one level: shipment directories
        for shipment_dir in monitor.iterdir():
            if not shipment_dir.is_dir():
                continue
            # Walk second level: time-chunk directories
            for chunk_dir in shipment_dir.iterdir():
                if not chunk_dir.is_dir():
                    continue
                # Skip non-time-chunk dirs (like "mismatch")
                name = chunk_dir.name
                if len(name) < 10 or '_' not in name:
                    continue
                # Validate it looks like YYYY-MM-DD_HH
                try:
                    parts = name.split('_')
                    if len(parts) >= 2 and len(parts[0]) == 10 and parts[0][4] == '-':
                        chunks.append((name, chunk_dir))
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        logger.error(f"Error scanning for time chunks: {e}")

    # Sort by chunk name (chronological since format is YYYY-MM-DD_HH)
    chunks.sort(key=lambda x: x[0])
    return chunks


def get_chunk_age_hours(chunk_name: str) -> float:
    """Estimate chunk age from its directory name (YYYY-MM-DD_HH)."""
    try:
        from datetime import datetime
        # Parse YYYY-MM-DD_HH
        chunk_time = datetime.strptime(chunk_name[:13], "%Y-%m-%d_%H")
        age = datetime.now() - chunk_time
        return age.total_seconds() / 3600
    except Exception:
        return float('inf')  # If we can't parse, treat as very old (safe to delete)


def delete_chunk(chunk_path: Path) -> float:
    """Delete an entire time-chunk directory. Returns GB freed."""
    try:
        # Calculate size before deletion
        chunk_size = sum(f.stat().st_size for f in chunk_path.rglob('*') if f.is_file())
        chunk_gb = chunk_size / (1024 ** 3)

        shutil.rmtree(chunk_path, ignore_errors=True)
        logger.info(f"Deleted chunk: {chunk_path} ({chunk_gb:.2f} GB)")
        return chunk_gb
    except Exception as e:
        logger.error(f"Error deleting chunk {chunk_path}: {e}")
        return 0.0


def cleanup_empty_shipment_dirs():
    """Remove empty shipment directories after chunk deletion."""
    monitor = Path(MONITOR_DIR)
    try:
        for shipment_dir in monitor.iterdir():
            if shipment_dir.is_dir():
                try:
                    if not any(shipment_dir.iterdir()):
                        shipment_dir.rmdir()
                        logger.info(f"Removed empty shipment dir: {shipment_dir.name}")
                except Exception:
                    pass
    except Exception:
        pass


def run_cleanup_cycle():
    """Delete oldest time chunks until disk usage drops below target."""
    global cleanup_runs, total_chunks_deleted, total_space_freed_gb

    cleanup_runs += 1
    cycle_start = time.time()

    usage_pct, used_gb, total_gb = get_disk_usage()
    logger.info(f"=== Cleanup Cycle {cleanup_runs} ===")
    logger.info(f"Disk: {usage_pct}% ({used_gb}GB / {total_gb}GB) — target: {MIN_USAGE_PERCENT}%")

    chunks = get_time_chunks()
    if not chunks:
        logger.warning("No time-chunk directories found")
        return

    logger.info(f"Found {len(chunks)} time chunks (oldest: {chunks[0][0]}, newest: {chunks[-1][0]})")

    chunks_deleted = 0
    gb_freed = 0.0

    for chunk_name, chunk_path in chunks:
        if usage_pct <= MIN_USAGE_PERCENT:
            break

        # Protect recent chunks
        age_hours = get_chunk_age_hours(chunk_name)
        if age_hours < MIN_CHUNK_AGE_HOURS:
            logger.info(f"Skipping {chunk_name} (age: {age_hours:.1f}h < {MIN_CHUNK_AGE_HOURS}h)")
            continue

        freed = delete_chunk(chunk_path)
        gb_freed += freed
        chunks_deleted += 1

        # Re-check disk usage after each chunk deletion
        usage_pct, used_gb, total_gb = get_disk_usage()
        logger.info(f"Progress: {chunks_deleted} chunks deleted, {gb_freed:.2f} GB freed, disk now {usage_pct}%")

    cleanup_empty_shipment_dirs()

    total_chunks_deleted += chunks_deleted
    total_space_freed_gb += gb_freed

    duration = time.time() - cycle_start
    logger.info(f"Cycle done in {duration:.1f}s — deleted {chunks_deleted} chunks ({gb_freed:.2f} GB)")
    logger.info(f"Lifetime: {total_chunks_deleted} chunks, {total_space_freed_gb:.2f} GB freed")


def verify_mount():
    """Verify MONITOR_DIR is accessible and writable."""
    monitor = Path(MONITOR_DIR)
    if not monitor.exists() or not monitor.is_dir():
        logger.error(f"{MONITOR_DIR} does not exist or is not a directory")
        return False

    test_file = monitor / ".cleanup_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
    except Exception as e:
        logger.error(f"{MONITOR_DIR} is not writable: {e}")
        return False

    usage = shutil.disk_usage(MONITOR_DIR)
    logger.info(f"Mount OK: {usage.total // (1024**3)}GB total, {usage.free // (1024**3)}GB free")
    return True


def main():
    logger.info("=== MonitaQC NVR Cleanup Service ===")
    logger.info(f"Monitor: {MONITOR_DIR}")
    logger.info(f"Threshold: {MAX_USAGE_PERCENT}% → {MIN_USAGE_PERCENT}%")
    logger.info(f"Check interval: {CHECK_INTERVAL}s")
    logger.info(f"Protect chunks newer than: {MIN_CHUNK_AGE_HOURS}h")

    if not verify_mount():
        time.sleep(60)
        if not verify_mount():
            logger.critical("Mount check failed twice. Exiting.")
            return

    usage_pct, used_gb, total_gb = get_disk_usage()
    logger.info(f"Current: {usage_pct}% ({used_gb}GB / {total_gb}GB)")

    while True:
        try:
            usage_pct, used_gb, total_gb = get_disk_usage()

            if usage_pct > MAX_USAGE_PERCENT:
                logger.warning(f"Disk {usage_pct}% > {MAX_USAGE_PERCENT}% — starting cleanup")
                run_cleanup_cycle()
            elif int(time.time()) % 300 < CHECK_INTERVAL:
                logger.info(f"OK: {usage_pct}% ({used_gb}GB / {total_gb}GB)")

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
