import os
import time
import subprocess
import heapq


# Configuration
MONITOR_DIR = os.getenv("MONITOR_DIR", "/data")
MAX_USAGE_PERCENT = int(os.getenv("MAX_USAGE_PERCENT", 85))  # Start deleting
MIN_USAGE_PERCENT = int(os.getenv("MIN_USAGE_PERCENT", 75))  # Stop deleting
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 10))  # Check every 10 sec
DELETION_BATCH_SIZE = int(os.getenv("DELETION_BATCH_SIZE", 20))  # Number of files to delete per cycle

def get_disk_usage():
    """Returns the disk usage percentage of MONITOR_DIR using 'df'."""
    try:
        result = subprocess.check_output(["df", "--output=pcent", MONITOR_DIR]).decode("utf-8")
        usage = result.splitlines()[1].strip().strip('%')  # Extracts the percentage value
        return int(usage) if usage.isdigit() else 100
    except Exception as e:
        print(f"Error fetching disk usage: {e}")
        return 100


def get_oldest_files(batch_size):
    """Finds the oldest files in MONITOR_DIR and its subdirectories."""
    file_heap = []

    for root, _, files in os.walk(MONITOR_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                heapq.heappush(file_heap, (os.path.getmtime(file_path), file_path))
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    return [heapq.heappop(file_heap)[1] for _ in range(min(batch_size, len(file_heap)))]

def delete_files(files):
    """Deletes files one by one."""
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

def delete_empty_dirs():
    """Recursively removes empty directories."""
    for root, dirs, _ in os.walk(MONITOR_DIR, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Check if directory is empty
                try:
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
                except Exception as e:
                    print(f"Error removing directory {dir_path}: {e}")

def cleanup_oldest_files():
    """Deletes oldest files until disk usage reaches MIN_USAGE_PERCENT."""
    while True:
        current_usage = get_disk_usage()
        if current_usage <= MIN_USAGE_PERCENT:
            print(f"Disk usage at {current_usage}%, stopping cleanup.")
            break

        oldest_files = get_oldest_files(DELETION_BATCH_SIZE)
        if not oldest_files:
            print("No files found for deletion.")
            break

        delete_files(oldest_files)
        delete_empty_dirs()  # Clean up empty directories

        current_usage = get_disk_usage()
        if current_usage <= MIN_USAGE_PERCENT:
            print(f"Disk usage at {current_usage}%, stopping cleanup.")
            break

        time.sleep(0.1)

if __name__ == "__main__":
    while True:
        percent_used = get_disk_usage()

        if percent_used > MAX_USAGE_PERCENT:
            print(f"Disk usage at {percent_used}%, starting cleanup...")
            cleanup_oldest_files()

        time.sleep(CHECK_INTERVAL)
