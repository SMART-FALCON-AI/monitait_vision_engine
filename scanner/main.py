import evdev
from datetime import datetime
from redis import Redis
import time
import threading
import os

default_path_hc = "/dev/input/event3"

SCANNER_FILE = ".env.scanner"

def load_default_path(file_path=SCANNER_FILE):
    try:
        # Open and read the file
        with open(file_path, "r") as file:
            for line in file:
                # Skip empty lines and comments
                line = line.strip()
                if line and not line.startswith("#"):
                    # Parse the key-value pair
                    key, value = line.split("=", 1)
                    if key.strip() == "default_path":
                        return value.strip()
        # If the key is not found, return None
        return default_path
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return default_path_hc

default_path = load_default_path()

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

    def update_queue_messages_redis(self, queue_messages):
        with self.redis_connection.pipeline() as pipe2:
            # pipe2.delete("queue_messages")
            # for i in range(queue_messages):
            pipe2.rpush("dms", queue_messages)
            pipe2.execute()
        # self.redis_connection.set('dms', queue_messages)

class Scanner:
    def __init__(self, device_path_id, vendor_product= [[0x1eab, 0x1e03], [0x0581, 0x011c], [0xac90, 0x3002]]) -> None:
        self.VENDOR_PRODUCT = vendor_product
        self.device_path_id = device_path_id
        self.start_init = False
        self.CHARMAP = {
        evdev.ecodes.KEY_1: ['1', '!'],
        evdev.ecodes.KEY_2: ['2', '@'],
        evdev.ecodes.KEY_3: ['3', '#'],
        evdev.ecodes.KEY_4: ['4', '$'],
        evdev.ecodes.KEY_5: ['5', '%'],
        evdev.ecodes.KEY_6: ['6', '^'],
        evdev.ecodes.KEY_7: ['7', '&'],
        evdev.ecodes.KEY_8: ['8', '*'],
        evdev.ecodes.KEY_9: ['9', '('],
        evdev.ecodes.KEY_0: ['0', ')'],
        evdev.ecodes.KEY_MINUS: ['-', '_'],
        evdev.ecodes.KEY_EQUAL: ['=', '+'],
        evdev.ecodes.KEY_TAB: ['\t', '\t'],
        evdev.ecodes.KEY_Q: ['q', 'Q'],
        evdev.ecodes.KEY_W: ['w', 'W'],
        evdev.ecodes.KEY_E: ['e', 'E'],
        evdev.ecodes.KEY_R: ['r', 'R'],
        evdev.ecodes.KEY_T: ['t', 'T'],
        evdev.ecodes.KEY_Y: ['y', 'Y'],
        evdev.ecodes.KEY_U: ['u', 'U'],
        evdev.ecodes.KEY_I: ['i', 'I'],
        evdev.ecodes.KEY_O: ['o', 'O'],
        evdev.ecodes.KEY_P: ['p', 'P'],
        evdev.ecodes.KEY_LEFTBRACE: ['[', '{'],
        evdev.ecodes.KEY_RIGHTBRACE: [']', '}'],
        evdev.ecodes.KEY_A: ['a', 'A'],
        evdev.ecodes.KEY_S: ['s', 'S'],
        evdev.ecodes.KEY_D: ['d', 'D'],
        evdev.ecodes.KEY_F: ['f', 'F'],
        evdev.ecodes.KEY_G: ['g', 'G'],
        evdev.ecodes.KEY_H: ['h', 'H'],
        evdev.ecodes.KEY_J: ['j', 'J'],
        evdev.ecodes.KEY_K: ['k', 'K'],
        evdev.ecodes.KEY_L: ['l', 'L'],
        evdev.ecodes.KEY_SEMICOLON: [';', ':'],
        evdev.ecodes.KEY_APOSTROPHE: ['\'', '"'],
        evdev.ecodes.KEY_BACKSLASH: ['\\', '|'],
        evdev.ecodes.KEY_Z: ['z', 'Z'],
        evdev.ecodes.KEY_X: ['x', 'X'],
        evdev.ecodes.KEY_C: ['c', 'C'],
        evdev.ecodes.KEY_V: ['v', 'V'],
        evdev.ecodes.KEY_B: ['b', 'B'],
        evdev.ecodes.KEY_N: ['n', 'N'],
        evdev.ecodes.KEY_M: ['m', 'M'],
        evdev.ecodes.KEY_COMMA: [',', '<'],
        evdev.ecodes.KEY_DOT: ['.', '>'],
        evdev.ecodes.KEY_SLASH: ['/', '?'],
        evdev.ecodes.KEY_SPACE: [' ', ' '],
        }
        self.ERROR_CHARACTER = '?'
        self.VALUE_UP = 0
        self.VALUE_DOWN = 1
        self.barcode_string_output = ''
        if self.device_path_id:
            print(self.device_path_id)
            self.dev = evdev.InputDevice(self.device_path_id)
            try:
                print("try to connect: {}".format(self.device_path_id))
                self.dev.grab()   
                self.start_init = True
                print("done")
            except:
                self.dev.ungrab()
                print("couldn't grab device: {}".format(self.dev.path))
                pass
        
        if not(self.start_init):    
            for self.dev in [evdev.InputDevice(os.path.join("/dev/input", path)) for path in os.listdir("/dev/input")]:
                for vp in self.VENDOR_PRODUCT:
                    if self.dev.info.vendor == vp[0] and self.dev.info.product == vp[1]:
                        try:
                            print("let's try: {}, info: {}".format(self.dev.path, self.dev.info))
                            self.dev.grab()             
                            print('selected device:', self.dev)
                            self.start_init = True 
                            break

                        except:
                            self.dev.ungrab()
                            print("fail to connect : {}".format(self.dev.path))
                            self.start_init = False  
                            pass



        if self.start_init:
            threading.Thread(target=self.read_barcode).start()
        else:
            return False

    def barcode_reader_evdev(self):    
        self.barcode_string_output = ''
        # barcode can have a 'shift' character; this switches the character set
        # from the lower to upper case variant for the next character only.
        self.shift_active = False
        for self.event in self.dev.read_loop():

            if self.event.code == evdev.ecodes.KEY_ENTER and self.event.value == self.VALUE_DOWN:
                #print('KEY_ENTER -> return')
                # all barcodes end with a carriage return
                return self.barcode_string_output
            elif self.event.code == evdev.ecodes.KEY_LEFTSHIFT or self.event.code == evdev.ecodes.KEY_RIGHTSHIFT:
                #print('SHIFT')
                self.shift_active = self.event.value == self.VALUE_DOWN
            elif self.event.value == self.VALUE_DOWN:
                ch = self.CHARMAP.get(self.event.code, self.ERROR_CHARACTER)[1 if self.shift_active else 0]
                #print('ch:', ch)
                # if the charcode isn't recognized, use ?
                self.barcode_string_output += ch
        
        return self.barcode_string_output
    
    def read_barcode(self):
        try:
            self.upcnumber = self.barcode_reader_evdev()
            # print(self.upcnumber)
            return self.upcnumber
        except KeyboardInterrupt:
            print('Keyboard interrupt')
        except Exception as err:
            print(err)
#        self.dev.ungrab()
        
        

redis = RedisConnection('0.0.0.0', 6379)

print("redis configed for notify-keyspace-events KEA: {}".format(redis.redis_connection.config_set('notify-keyspace-events', 'KEA')))

for device_path in ["/dev/input/by-id/usb-TC_Electroinc_2D_Barcode_Scanner-hid-event-kbd",
                    "/dev/input/by-id/usb-SM_SM-2D_PRODUCT_HID_KBW_APP-000000000-event-kbd",
                    "/dev/input/by-id/usb-TC_Electroinc_2D_Barcode_Scanner-hid-event-kbd",
                    "/dev/event100",
                    "/dev/input/event3",
                    "/dev/input/event15",
                    default_path_hc,
                    default_path]: # the last is the default path
    if os.path.exists(device_path):
        print(device_path)
        device = device_path
        try:
            scanner=Scanner(device)
            run_scanning = True
            break
        except Exception as e:
            print("no init: {}".format(str(e)))
            run_scanning = False
    else:
        print("no path")
        device = None

print("run: {}".format(run_scanning))

while run_scanning:
    try:
        barcode = scanner.read_barcode()
        if barcode != '':
            print("{} : {}".format(time.time(), barcode))
            if "batch_uuid" in barcode: 
                redis.set_redis("batch_uuid", barcode)
            elif "operator_id" in barcode:
                redis.set_redis("operator_id", barcode)
            else:
                redis.set_redis("shipment", barcode)

        time.sleep(0.1)
    except Exception as e:
        print(str(e))
        # run_scanning = False
        time.sleep(3)
        pass

print("bye for 15 seconds!")
time.sleep(15)
