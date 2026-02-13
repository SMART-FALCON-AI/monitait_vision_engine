#!/usr/bin/env python3
"""
Disk Cleanup Service for MonitaQC
Monitors disk usage and removes oldest files to maintain storage health.
Optimized for high-volume image storage on SSD.
"""
import os
import shutil
import time
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MONITOR_DIR = os.getenv("MONITOR_DIR", "/data")
MAX_USAGE_PERCENT = int(os.getenv("MAX_USAGE_PERCENT", 90))  # Start cleanup
MIN_USAGE_PERCENT = int(os.getenv("MIN_USAGE_PERCENT", 80))  # Stop cleanup
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 60))  # Check every 60 seconds
DELETION_BATCH_SIZE = int(os.getenv("DELETION_BATCH_SIZE", 1000))  # Files per batch
MIN_FILE_AGE_HOURS = int(os.getenv("MIN_FILE_AGE_HOURS", 1))  # Protect recent files
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"

# Metrics tracking
cleanup_runs = 0
total_files_deleted = 0
total_space_freed_mb = 0


def get_disk_usage() -> Tuple[int, int, int]:
    """
    Returns disk usage information for the filesystem containing MONITOR_DIR.
    Uses shutil.disk_usage() which directly queries the OS for the actual
    mounted filesystem - more reliable than parsing df output in containers.

    Returns:
        Tuple[int, int, int]: (usage_percent, used_gb, total_gb)
    """
    try:
        usage = shutil.disk_usage(MONITOR_DIR)
        total_gb = usage.total // (1024 ** 3)
        used_gb = usage.used // (1024 ** 3)
        usage_percent = int((usage.used / usage.total) * 100) if usage.total > 0 else 100
        return usage_percent, used_gb, total_gb
    except Exception as e:
        logger.error(f"Error fetching disk usage: {e}")
        return 100, 0, 0


def get_files_by_age(min_age_hours: int = 0) -> List[Tuple[float, Path, int]]:
    """
    Get all files sorted by modification time (oldest first).

    Args:
        min_age_hours: Only include files older than this many hours

    Returns:
        List of tuples: (mtime, filepath, size_bytes)
    """
    files = []
    min_age_seconds = min_age_hours * 3600
    current_time = time.time()

    logger.info(f"Scanning {MONITOR_DIR} for files older than {min_age_hours}h...")

    try:
        for entry in Path(MONITOR_DIR).rglob('*'):
            if entry.is_file():
                try:
                    stat = entry.stat()
                    age_seconds = current_time - stat.st_mtime

                    # Only include files older than minimum age
                    if age_seconds >= min_age_seconds:
                        files.append((stat.st_mtime, entry, stat.st_size))
                except (OSError, PermissionError) as e:
                    logger.debug(f"Skipping {entry}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error scanning directory: {e}")
        return []

    # Sort by modification time (oldest first)
    files.sort(key=lambda x: x[0])

    logger.info(f"Found {len(files)} eligible files for cleanup")
    return files


def delete_files_batch(files: List[Tuple[float, Path, int]], batch_size: int) -> Tuple[int, int]:
    """
    Delete a batch of files.

    Args:
        files: List of (mtime, filepath, size) tuples
        batch_size: Maximum number of files to delete

    Returns:
        Tuple[int, int]: (num_deleted, bytes_freed)
    """
    deleted_count = 0
    bytes_freed = 0

    for i, (mtime, filepath, size) in enumerate(files):
        if i >= batch_size:
            break

        try:
            filepath.unlink()
            deleted_count += 1
            bytes_freed += size

            # Log every 100 files for visibility
            if deleted_count % 100 == 0:
                logger.info(f"Deleted {deleted_count} files, freed {bytes_freed / (1024**2):.1f} MB")

        except PermissionError:
            # External SSD files may have restrictive permissions - try chmod first
            try:
                filepath.chmod(0o666)
                filepath.unlink()
                deleted_count += 1
                bytes_freed += size
            except Exception as e2:
                logger.warning(f"Permission denied even after chmod for {filepath}: {e2}")
        except OSError as e:
            if e.errno == 16:  # Device or resource busy
                logger.warning(f"File busy, skipping {filepath}")
            elif e.errno == 30:  # Read-only filesystem
                logger.error(f"Read-only filesystem for {filepath} - check mount options!")
            else:
                logger.warning(f"OS error deleting {filepath}: {e}")
        except Exception as e:
            logger.warning(f"Error deleting {filepath}: {e}")

    return deleted_count, bytes_freed


def cleanup_empty_directories():
    """Recursively remove empty directories."""
    removed_count = 0

    try:
        for dirpath, dirnames, filenames in os.walk(MONITOR_DIR, topdown=False):
            for dirname in dirnames:
                dir_path = Path(dirpath) / dirname
                try:
                    if not any(dir_path.iterdir()):  # Check if empty
                        dir_path.rmdir()
                        removed_count += 1
                        logger.debug(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    logger.debug(f"Could not remove {dir_path}: {e}")
    except Exception as e:
        logger.error(f"Error during directory cleanup: {e}")

    if removed_count > 0:
        logger.info(f"Removed {removed_count} empty directories")


def run_cleanup_cycle():
    """Execute one cleanup cycle: delete oldest files until target usage reached."""
    global cleanup_runs, total_files_deleted, total_space_freed_mb

    cleanup_runs += 1
    cycle_start = time.time()

    usage_percent, used_gb, total_gb = get_disk_usage()
    logger.info(f"=== Cleanup Cycle {cleanup_runs} Started ===")
    logger.info(f"Disk usage: {usage_percent}% ({used_gb}GB / {total_gb}GB)")
    logger.info(f"Target: reduce to {MIN_USAGE_PERCENT}%")

    # Get eligible files (protecting recent files)
    eligible_files = get_files_by_age(MIN_FILE_AGE_HOURS)

    if not eligible_files:
        logger.warning("No eligible files found for deletion (all files may be too recent)")
        return

    cycle_deleted = 0
    cycle_freed_bytes = 0

    # Delete files in batches until target reached
    while usage_percent > MIN_USAGE_PERCENT:
        # Calculate how many files to delete this batch
        remaining_files = len(eligible_files) - cycle_deleted
        batch_size = min(DELETION_BATCH_SIZE, remaining_files)

        if batch_size <= 0:
            logger.warning(f"No more files to delete, but usage still at {usage_percent}%")
            break

        # Delete batch
        files_to_delete = eligible_files[cycle_deleted:cycle_deleted + batch_size]
        deleted, freed = delete_files_batch(files_to_delete, batch_size)

        cycle_deleted += deleted
        cycle_freed_bytes += freed

        # Check progress
        usage_percent, used_gb, total_gb = get_disk_usage()
        logger.info(f"Progress: {cycle_deleted} files deleted, {cycle_freed_bytes / (1024**3):.2f} GB freed, usage now {usage_percent}%")

        if usage_percent <= MIN_USAGE_PERCENT:
            break

    # Cleanup empty directories
    cleanup_empty_directories()

    # Update global metrics
    total_files_deleted += cycle_deleted
    total_space_freed_mb += cycle_freed_bytes / (1024**2)

    cycle_duration = time.time() - cycle_start

    logger.info(f"=== Cleanup Cycle {cleanup_runs} Completed ===")
    logger.info(f"Duration: {cycle_duration:.1f}s")
    logger.info(f"Deleted: {cycle_deleted} files ({cycle_freed_bytes / (1024**3):.2f} GB)")
    logger.info(f"Final usage: {usage_percent}% ({used_gb}GB / {total_gb}GB)")

    if ENABLE_METRICS:
        logger.info(f"=== Lifetime Metrics ===")
        logger.info(f"Total cleanup runs: {cleanup_runs}")
        logger.info(f"Total files deleted: {total_files_deleted}")
        logger.info(f"Total space freed: {total_space_freed_mb / 1024:.2f} GB")


def verify_mount_health():
    """Verify that MONITOR_DIR is accessible, writable, and on the expected filesystem."""
    monitor_path = Path(MONITOR_DIR)

    if not monitor_path.exists():
        logger.error(f"MONITOR_DIR {MONITOR_DIR} does not exist!")
        return False

    if not monitor_path.is_dir():
        logger.error(f"MONITOR_DIR {MONITOR_DIR} is not a directory!")
        return False

    # Check if writable by creating and removing a test file
    test_file = monitor_path / ".cleanup_write_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        logger.info(f"Mount is writable: {MONITOR_DIR}")
    except PermissionError:
        logger.error(f"MONITOR_DIR {MONITOR_DIR} is NOT writable! Check mount permissions.")
        return False
    except OSError as e:
        logger.error(f"MONITOR_DIR {MONITOR_DIR} write test failed: {e}")
        return False

    # Log filesystem info
    usage = shutil.disk_usage(MONITOR_DIR)
    logger.info(f"Filesystem total: {usage.total // (1024**3)}GB, "
                f"used: {usage.used // (1024**3)}GB, "
                f"free: {usage.free // (1024**3)}GB")

    # Check that this is actually a separate mount (not root /)
    try:
        root_usage = shutil.disk_usage("/")
        if usage.total == root_usage.total:
            logger.warning(f"WARNING: {MONITOR_DIR} appears to be on the same filesystem as /. "
                           f"Expected a separate mount (external SSD).")
        else:
            logger.info(f"Confirmed: {MONITOR_DIR} is on a separate filesystem from / "
                        f"(root={root_usage.total // (1024**3)}GB, monitor={usage.total // (1024**3)}GB)")
    except Exception:
        pass

    return True


def main():
    """Main monitoring loop."""
    logger.info("=== MonitaQC Disk Cleanup Service Started ===")
    logger.info(f"Monitor directory: {MONITOR_DIR}")
    logger.info(f"Cleanup threshold: {MAX_USAGE_PERCENT}% (target: {MIN_USAGE_PERCENT}%)")
    logger.info(f"Check interval: {CHECK_INTERVAL}s")
    logger.info(f"Batch size: {DELETION_BATCH_SIZE} files")
    logger.info(f"Min file age: {MIN_FILE_AGE_HOURS}h")

    # Verify mount health before starting
    if not verify_mount_health():
        logger.error("Mount health check failed! Waiting 60s before retry...")
        time.sleep(60)
        if not verify_mount_health():
            logger.critical("Mount health check failed twice. Exiting.")
            return

    # Initial disk check
    usage_percent, used_gb, total_gb = get_disk_usage()
    logger.info(f"Current disk usage: {usage_percent}% ({used_gb}GB / {total_gb}GB)")

    while True:
        try:
            usage_percent, used_gb, total_gb = get_disk_usage()

            if usage_percent > MAX_USAGE_PERCENT:
                logger.warning(f"Disk usage threshold exceeded: {usage_percent}% > {MAX_USAGE_PERCENT}%")
                run_cleanup_cycle()
            else:
                # Periodic status update (every 10 checks)
                if int(time.time()) % (CHECK_INTERVAL * 10) == 0:
                    logger.info(f"Status: {usage_percent}% disk usage ({used_gb}GB / {total_gb}GB) - OK")

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Cleanup service stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
