import json
import logging
import time

from redis import Redis

logger = logging.getLogger(__name__)


class RedisConnection:
    def __init__(self, redis_hostname, redis_port):
        self.redis_hostname = redis_hostname
        self.redis_port = redis_port
        self.redis_connection = self.connect_to_redis()

    # Connecting to Radis database
    def connect_to_redis(self):
        return Redis(self.redis_hostname, self.redis_port, db=3)

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
