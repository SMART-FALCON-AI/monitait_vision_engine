import json
import logging
import time

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
        return Redis(self.redis_hostname, self.redis_port, db=REDIS_DB)

    def set_flag(self, list_lenght):
        with self.redis_connection.pipeline() as pipe:
            pipe.delete("camera_list")
            for i in range(list_lenght):
                pipe.rpush("camera_list", 0)
            pipe.execute()

    def update_encoder_redis(self, encoder):
        self.redis_connection.set("encoder_values", json.dumps(encoder))

    def set_captuting_flag(self, key):
        self.redis_connection.set(key, 1)

    def set_redis(self, key, value):
        self.redis_connection.set(key, value)

    def get_redis(self, key):
        return self.redis_connection.get(key)

    def set_light_mode(self, mode):
        self.redis_connection.set("light_mode", mode)

    def update_queue_messages_redis(self, queue_messages , stream_name="dms"):
        with self.redis_connection.pipeline() as pipe2:
            # pipe2.delete("queue_messages")
            # for i in range(queue_messages):
            pipe2.rpush(stream_name, queue_messages)
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
