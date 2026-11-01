import logging

from redis import Redis
from config import REDIS_DB

logger = logging.getLogger(__name__)


class RedisConnection:
    def __init__(self, redis_hostname, redis_port):
        self.redis_hostname = redis_hostname
        self.redis_port = redis_port
        self.redis_connection = self.connect_to_redis()

    # Connecting to Radis database
    def connect_to_redis(self):
        # 4.0.425 — socket timeouts: this client is shared by the capture loop, the ejector
        # and /health (which pings it ON the event loop). With none, a hung Redis connection
        # blocked forever — the web server included. 10 s is far above the longest blocking
        # call made on it (BLPOP with ≤ 1 s timeout).
        return Redis(self.redis_hostname, self.redis_port, db=REDIS_DB,
                     socket_timeout=10, socket_connect_timeout=5)

    # 4.0.422 — lists that only an EXTERNAL consumer may read (fabriqc / the speaker
    # service). Where nothing reads them they grew forever (vteam12: 82,776 `dms` +
    # 82,880 `speaker` items, Redis at 730 MB) until allkeys-lru evicted live keys such
    # as `shipment`. Keep the newest N — a consumer that keeps up never sees the cap.
    # MVE's own queues (stream_queue, ejector_queue) are never trimmed.
    _LIST_CAPS = {"dms": 2000, "speaker": 200}

    def update_queue_messages_redis(self, queue_messages , stream_name="dms"):
        with self.redis_connection.pipeline() as pipe2:
            pipe2.rpush(stream_name, queue_messages)
            cap = self._LIST_CAPS.get(stream_name)
            if cap:
                pipe2.ltrim(stream_name, -cap, -1)
            pipe2.execute()
        # self.redis_connection.set('dms', queue_messages)

    def pop_queue_messages_redis(self, stream_name="frames_queue"):
            return self.redis_connection.lpop(stream_name)

    def pop_queue_blocking(self, stream_name="frames_queue", timeout=1.0):
        """v4.0.101 — wake-on-data BLPOP variant of pop_queue_messages_redis.

        Returns (key, value) tuple on data, or None on timeout. Used by the
        ejector loop (services/watcher.py) to eliminate its 200 Hz polling
        against Redis: `EJECTOR_POLL_INTERVAL=0.005` was waking the thread
        200 times per second even when idle. With BLPOP the thread parks
        inside Redis until either new data lands or `timeout` seconds pass,
        which caps idle wake-ups at ~1/sec instead of ~200/sec (also cuts
        end-to-end ejection latency from ~2.5 ms average to sub-millisecond).

        `timeout=0` would block forever; we pass 1.0 by default so the outer
        state-machine loop still gets a chance to check the encoder value
        even during a lull with no new ejection requests.
        """
        return self.redis_connection.blpop(stream_name, timeout=timeout)
