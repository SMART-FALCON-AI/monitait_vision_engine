from datetime import datetime
from redis import Redis
import time
import pyttsx3

default_text_ejector = "eject"

class RedisConnection:
    def __init__(self, redis_hostname, redis_port):
        self.redis_hostname = redis_hostname
        self.redis_port = redis_port
        self.redis_connection = self.connect_to_redis()

    # Connecting to Redis database
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

    def update_queue_messages_redis(self,  queue_messages, stream_name="dms"):
        with self.redis_connection.pipeline() as pipe2:
            # pipe2.delete("queue_messages")
            # for i in range(queue_messages):
            pipe2.rpush(stream_name, queue_messages)
            pipe2.execute()
        # self.redis_connection.set('dms', queue_messages)

redis_connection = RedisConnection('0.0.0.0', 6379)
run_scanning = True

engine = pyttsx3.init()
def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

while run_scanning:
    try:
        # Continuously attempt to pop messages from the "speaker" list
        message = redis_connection.redis_connection.rpop("speaker")
        if message:
            text = message.decode('utf-8')
            text_to_speech(text)  # Narrate the text
            print(f"Narrated: {text}")
        else:
            # No message in the list, pause briefly before checking again
            time.sleep(0.1)

    except Exception as e:
        print("Narration failed: {}".format(str(e)))

print("bye for 15 seconds!")
time.sleep(15)