import logging
import threading
import queue
import psycopg2
from psycopg2.extras import Json
from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = logging.getLogger(__name__)

# Database connection pool
db_connection_pool = None

# Background write queue — all DB writes go through this to avoid blocking callers
_db_queue = queue.Queue(maxsize=1000)


def get_db_connection():
    """Get PostgreSQL connection from pool."""
    global db_connection_pool
    if db_connection_pool is None:
        try:
            from psycopg2 import pool
            db_connection_pool = pool.SimpleConnectionPool(
                1, 10,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            return None
    try:
        return db_connection_pool.getconn()
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
        return None

def release_db_connection(conn):
    """Release connection back to pool."""
    if db_connection_pool and conn:
        db_connection_pool.putconn(conn)


def _db_writer_loop():
    """Background thread that drains the write queue and executes DB inserts."""
    while True:
        try:
            task = _db_queue.get()
            if task is None:
                break
            sql, params = task
            conn = get_db_connection()
            if not conn:
                continue
            try:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                cursor.close()
            except Exception as e:
                logger.error(f"DB write error: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                release_db_connection(conn)
        except Exception as e:
            logger.error(f"DB writer loop error: {e}")


# Start the background writer thread
_db_writer_thread = threading.Thread(target=_db_writer_loop, daemon=True, name="db-writer")
_db_writer_thread.start()


def write_inference_to_db(shipment, image_path, detections, inference_time_ms, model_used="yolov8"):
    """Queue inference result for async write to TimescaleDB."""
    try:
        _db_queue.put_nowait((
            """INSERT INTO inference_results
               (time, shipment, image_path, detections, detection_count, inference_time_ms, model_used)
               VALUES (NOW(), %s, %s, %s, %s, %s, %s)""",
            (shipment, image_path, Json(detections), len(detections), inference_time_ms, model_used)
        ))
    except queue.Full:
        logger.warning("DB write queue full — dropping inference result")

def write_production_metrics_to_db(encoder_value, ok_counter, ng_counter, shipment, is_moving, downtime_seconds):
    """Queue production metrics for async write to TimescaleDB."""
    try:
        _db_queue.put_nowait((
            """INSERT INTO production_metrics
               (time, encoder_value, ok_counter, ng_counter, shipment, is_moving, downtime_seconds)
               VALUES (NOW(), %s, %s, %s, %s, %s, %s)""",
            (encoder_value, ok_counter, ng_counter, shipment, is_moving, downtime_seconds)
        ))
    except queue.Full:
        logger.warning("DB write queue full — dropping production metrics")
