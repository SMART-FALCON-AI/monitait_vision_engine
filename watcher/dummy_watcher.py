#!/usr/bin/env python3
"""Dummy Watcher Jet: drives MVE's watcher API exactly like the real device, without
the Raspberry Pi hardware. Use it to prove an MVE install accepts watcher data before
connecting a real Watcher Jet.

Built from monitait_watcher_jet/main.py (commit 8113aa0).
  KEPT from the firmware (same logic):
    - watcher_update(): payload shape, 2-step image upload, what counts as success
    - DB: the SQLite offline buffer
    - Counter.run() / Counter.db_checker(): when to send (a + b > dps, or the
      heartbeat), subtract-on-success (each report carries the counts SINCE THE LAST
      REPORT, not a running total), store-and-replay with the original timestamp
      when MVE is unreachable
  SIMULATED:
    - FakeArduino: OK/NG pulses at --ppm with --ng ratio, analog level (c),
      battery level (d) and the serial channels d0..dN
    - FakeCamera: writes a small JPEG instead of grabbing /dev/video0

The ONLY change a real Watcher Jet needs to report to MVE instead of the cloud is the
base URL: in main.py replace "https://app.monitait.com" (3 places) with
"http://<mve-ip>" -> e.g. http://192.168.0.143/api/factory/update-watcher/

Usage:
  python3 dummy_watcher.py --url http://<mve-ip> --duration 300
  python3 dummy_watcher.py --url http://<mve-ip> --duration 240 --offline-first 40 --image-every 60
  python3 dummy_watcher.py --url http://<mve-ip> --verify-only   # re-check the last run

Requires: python3 + requests (sqlite3 is stdlib). Exit code 0 = every check passed.
"""
import argparse
import datetime
import json
import os
import random
import socket
import sqlite3
import sys
import threading
import time
import warnings
from threading import Thread

import requests

warnings.filterwarnings("ignore", category=DeprecationWarning)  # utcnow(), kept as in the firmware

hostname = str(socket.gethostname())

# Base URL for the 3 watcher endpoints. The firmware hard-codes the cloud
# (https://app.monitait.com); MVE serves the same paths.
MVE_BASE = "http://127.0.0.1"

# Ground truth for the end-of-run check: what was produced vs what MVE acknowledged.
LEDGER = {"produced_ok": 0, "produced_ng": 0, "delivered_ok": 0, "delivered_ng": 0,
          "reports_ok": 0, "reports_failed": 0, "buffered": 0, "replayed": 0,
          "images_sent": 0}
_ledger_lock = threading.Lock()


def _ledger_add(**kw):
    with _ledger_lock:
        for k, v in kw.items():
            LEDGER[k] += v


def watcher_update(register_id, quantity, defect_quantity, send_img, image_path="scene_image.jpg", product_id=0, lot_info=0, extra_info=None, timestamp=None, *args, **kwargs):
    """Firmware watcher_update(), unchanged except: base URL = MVE_BASE, and the
    device timestamp default is evaluated per call (the firmware's default arg is
    frozen at import; every firmware caller passes it explicitly anyway)."""
    timestamp = (timestamp or datetime.datetime.utcnow()).strftime('%Y-%m-%dT%H:%M:%S.%f')
    product_info = kwargs.pop("product_info", None)

    DATA = {
        "register_id" : register_id,
        "quantity" : quantity,
        "defect_quantity": defect_quantity,
        "product_id": product_id,
        "extra_info": extra_info if extra_info else {},
        "lot_info": lot_info,
        "timestamp":timestamp,
        "product_info":product_info
        }
    session = requests.Session()

    URL = MVE_BASE + "/api/factory/update-watcher/" # send data without waiting for elastic id
    URL_DATA = MVE_BASE + "/api/factory/image-update-watcher-data/" # send data and get elastic id
    URL_IMAGE = MVE_BASE + "/api/factory/image-update-watcher/" # send image based on elastic id

    try:
        if send_img:
            try:
                response = session.post(URL_DATA, data=json.dumps(DATA), headers={"content-type": "application/json"}, timeout=150)
                result = response.json()
                _id = result.get('_id', None)
                time.sleep(1)
                if _id:
                    DATA = {
                        'register_id':result['register_id'],
                        'elastic_id':_id
                        }
                    try:
                        with open(image_path, "rb") as f:
                            response = session.post(URL_IMAGE, files={"image": f}, data=DATA, timeout=250)
                        session.close()
                        if (response.status_code == requests.codes.ok):
                            os.remove(image_path)
                            _ledger_add(images_sent=1)
                            return True
                        else:
                            return False
                    except Exception as e:
                        return False
                else:
                    session.close()
                    return False
            except Exception as e:
                return False
        else:
            try:
                response = requests.post(URL, data=json.dumps(DATA), headers={"content-type": "application/json"}, timeout=30)
                if (response.status_code == requests.codes.ok):
                    return True
                else:
                    print(f"  MVE answered HTTP {response.status_code}: {response.text[:200]}")
                    return False
            except Exception as e:
                return False
    except Exception as e:
        session.close()
        return False


class DB:
    """Firmware SQLite offline buffer (path configurable instead of /home/pi/...)."""
    def __init__(self, path) -> None:
        try:
            self.dbconnect = sqlite3.connect(path, check_same_thread=False)
            self.cursor = self.dbconnect.cursor()
            self.db_connection = True
            self.lock = threading.Lock()
            self.cursor.execute('''create table monitait_table (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, register_id TEXT, temp_a INTEGER NULL, temp_b INTEGER NULL, image_name TEXT NULL, extra_info JSON, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL)''')
            self.dbconnect.commit()
        except Exception as e:
            pass

    def write(self, register_id=hostname, a=0, b=0, extra_info={}, image_name="", timestamp=None):
        try:
            with self.lock:
                self.cursor.execute('''insert into monitait_table (register_id, temp_a, temp_b, image_name, extra_info, ts) values (?,?,?,?,?,?)''', (register_id, a, b, image_name, json.dumps(extra_info), timestamp))
                self.dbconnect.commit()
            return True
        except Exception as e:
            return False

    def read(self):
        try:
            with self.lock:
                self.cursor.execute('SELECT * FROM monitait_table LIMIT 1')
                rows = self.cursor.fetchall()
            return rows[0]
        except Exception as e:
            return []

    def delete(self, id):
        try:
            with self.lock:
                self.cursor.execute("""DELETE from monitait_table where id = ?""", (id,))
                self.dbconnect.commit()
            return True
        except Exception as e:
            return False

    def pending(self):
        with self.lock:
            self.cursor.execute('SELECT COUNT(*) FROM monitait_table')
            return self.cursor.fetchone()[0]


class FakeArduino:
    """Stands in for the firmware's Ardiuno class (GPIO pulse reader + serial).
    Same interface: read_GPIO(), minus(), read_serial(), serial_connection, and the
    same smoothed-rate formula for get_ts (the send threshold)."""
    def __init__(self, ppm=120, ng_ratio=0.05, serial_channels=2) -> None:
        self.stop_thread = False
        self.ppm = ppm
        self.ng_ratio = ng_ratio
        self.last_a = 0
        self.last_b = 0
        self.c = 3   # analog input, 8 levels [0..7]
        self.d = 7   # battery, 8 levels [0..7]
        self.get_ts = 1
        self.lock = threading.Lock()
        self.serial_connection = serial_channels > 0
        self.serial_channels = serial_channels
        self.serial_data = {}
        Thread(target=self.run_GPIO, daemon=True).start()
        if self.serial_connection:
            Thread(target=self.run_serial, daemon=True).start()

    def run_serial(self):
        # The real Arduino prints "v0,v1,...\r\n" at 9600 baud; the firmware turns it
        # into {"d0": v0, "d1": v1, ...}. Simulate slowly drifting values (e.g. temps).
        base = [random.randint(150, 250) for _ in range(self.serial_channels)]
        while not self.stop_thread:
            self.serial_data = {"d{}".format(z): base[z] + random.randint(-5, 5)
                                for z in range(self.serial_channels)}
            time.sleep(1)

    def run_GPIO(self):
        self.old_start_ts = time.time()
        while not self.stop_thread:
            if self.ppm <= 0:
                time.sleep(0.2)
                continue
            # regular spacing with ±20% jitter, like a conveyor; Poisson spacing would
            # spike the firmware's 1/dt rate estimate so only the heartbeat would send
            time.sleep(60.0 / self.ppm * random.uniform(0.8, 1.2))
            if self.stop_thread:
                break
            ng = random.random() < self.ng_ratio
            with self.lock:
                if ng:
                    self.last_b += 1
                else:
                    self.last_a += 1
            _ledger_add(produced_ng=int(ng), produced_ok=int(not ng))
            start_ts = time.time()
            self.get_ts = 1/(start_ts - self.old_start_ts)+0.9*self.get_ts
            self.old_start_ts = start_ts
            if random.random() < 0.02:
                self.c = random.randint(0, 7)
            if random.random() < 0.005:
                self.d = max(0, self.d - 1)

    def read_GPIO(self):
        with self.lock:
            return self.last_a, self.last_b, self.c, self.d, self.get_ts

    def minus(self, a, b):
        with self.lock:
            self.last_a -= a
            self.last_b -= b

    def read_serial(self):
        return self.serial_data


# Smallest valid JPEG (1x1 grey) — the image leg only needs a real file to upload.
_TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c"
    "140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27"
    "393d38323c2e333432ffc0000b080001000101011100ffc4001f0000010501010101010100000000"
    "000000000102030405060708090a0bffc400b5100002010303020403050504040000017d01020300"
    "041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a"
    "25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475"
    "767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9ba"
    "c2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda"
    "0008010100003f00fbd3ffd9")


class FakeCamera:
    def __init__(self, folder):
        self.folder = folder
        self.success = True

    def capture_and_save(self):
        date = datetime.datetime.utcnow()
        saving_date = f"{date.year}-{date.month}-{date.day}-{date.hour}-{date.minute}-{date.second}-{date.microsecond}"
        os.makedirs(self.folder, exist_ok=True)
        image_name = os.path.join(self.folder, f"{saving_date}.jpg")
        with open(image_name, "wb") as f:
            f.write(_TINY_JPEG)
        return True, image_name


class Counter:
    """Firmware Counter: run() and db_checker() logic unchanged; intervals are
    parameters (firmware: heartbeat 300 s, picture 300 s) and run() stops at a
    deadline so the test ends by itself."""
    def __init__(self, arduino, db, camera=None, heartbeat=300, picture_interval=300) -> None:
        self.arduino = arduino
        self.stop_thread = False
        self.db = db
        self.old_local_ip = None
        self.old_err_msg = ""
        self.err_msg = ""
        self.old_c = 0 # analog input of watcher
        self.old_d = 0 # battery value
        self.camera = camera
        self.scanner = None
        self.watcher_live_signal = heartbeat
        self.take_picture_interval = picture_interval

    def get_ip_address(self):
        ip_address = None
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
        except Exception as e:
            if not("-IP-address-" in self.err_msg):
                self.err_msg = self.err_msg + "-IP-address-" + str(e)
        finally:
            s.close()
        return ip_address

    def db_checker(self):
        while not self.stop_thread:
            try:
                data = self.db.read()
                if len(data):
                    if watcher_update(register_id=data[1], quantity=data[2], defect_quantity=data[3], send_img=len(data[4]) > 0, image_path=data[4], extra_info=json.loads(data[5]), timestamp=datetime.datetime.strptime(data[6], '%Y-%m-%d %H:%M:%S.%f')):
                        self.db.delete(data[0])
                        _ledger_add(delivered_ok=data[2], delivered_ng=data[3], replayed=1)
                time.sleep(1)
            except Exception as e:
                if not("db_checker" in self.err_msg):
                    self.err_msg = self.err_msg + "db_checker" + str(e)
                time.sleep(1)

    def run(self, register_id, deadline, on_tick=None):
        self.last_server_signal = time.time()
        if self.camera:
            self.last_image = time.time() - self.take_picture_interval  # first report carries an image
        while not self.stop_thread and time.time() < deadline:
            try:
                if on_tick:
                    on_tick()
                data_saved = False
                send_image = False
                image_name = ""
                extra_info = {}
                ts = time.time()
                a ,b ,c, d ,dps = self.arduino.read_GPIO()
                if a + b > dps or ts - self.last_server_signal > self.watcher_live_signal:
                    self.last_server_signal = ts
                    if self.camera:
                        if ts - self.last_image > self.take_picture_interval:
                            captured, image_name = self.camera.capture_and_save()
                            if captured:
                                send_image = True
                                self.last_image = ts
                            else:
                                send_image = False
                    if self.arduino.serial_connection:
                        extra_info = dict(self.arduino.read_serial())

                    if (c != self.old_c):
                        self.old_c  = c
                        extra_info.update({"analog" : self.old_c })

                    if (d != self.old_d):
                        self.old_d  = d
                        extra_info.update({"battery" : self.old_d })

                    self.local_ip = self.get_ip_address()
                    if (self.local_ip != self.old_local_ip ):
                        self.old_local_ip  = self.local_ip
                        extra_info.update({"local_ip" : self.old_local_ip })

                    if (self.err_msg != self.old_err_msg):
                        self.old_err_msg = self.err_msg
                        extra_info.update({"err_msg" : self.old_err_msg })

                    timestamp = datetime.datetime.utcnow()

                    if watcher_update(register_id, quantity=a, defect_quantity=b, send_img=send_image, image_path=image_name, extra_info=extra_info, timestamp=timestamp):
                        data_saved = True
                        _ledger_add(delivered_ok=a, delivered_ng=b, reports_ok=1)
                    else:
                        _ledger_add(reports_failed=1)
                        try:
                            if self.db.write(register_id=register_id, a=a, b=b, extra_info=extra_info, timestamp=timestamp, image_name=image_name):
                                data_saved = True
                                _ledger_add(buffered=1)
                        except Exception as e:
                            if not("db_write" in self.err_msg):
                                self.err_msg = self.err_msg + "db_write" + str(e)
                    if data_saved:
                        self.arduino.minus(a=a, b=b)

                time.sleep(1)
            except Exception as e:
                time.sleep(1)
                print(e)
                if not("counter_run" in self.err_msg):
                    self.err_msg = self.err_msg + "counter_run" + str(e)


# --------------------------------------------------------------------------- #
# verification against MVE's read API
# --------------------------------------------------------------------------- #
def verify(base, reg, since_ms, ledger, ppm):
    fails = []

    def check(name, ok, detail=""):
        print(("  PASS  " if ok else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))
        if not ok:
            fails.append(name)

    print(f"\nVerifying against {base} (register_id={reg})")
    lst = requests.get(base + "/api/watchers", timeout=30).json().get("watchers", [])
    me = next((w for w in lst if w.get("register_id") == reg), None)
    check("machine listed in /api/watchers", me is not None, f"{len(lst)} machine(s) listed")
    if me:
        check("machine shown ONLINE (reported < 10 min ago)", bool(me.get("online")),
              f"age_sec={me.get('age_sec')} last_seen={me.get('last_seen')}")

    mt = requests.get(base + "/api/watchers/metrics",
                      params={"register_id": reg, "since_ms": since_ms, "limit": 100000}, timeout=60).json()
    pts = mt.get("points", [])
    if not pts:
        # nothing inside the run window: were the rows stored with a shifted clock?
        allp = requests.get(base + "/api/watchers/metrics", params={"register_id": reg, "limit": 100000},
                            timeout=60).json().get("points", [])
        if allp:
            print(f"  info  {len(allp)} row(s) exist for this machine but none inside the run window; "
                  f"latest stored ts={allp[-1].get('ts')} (timezone shift?)")
    sum_q = sum(p.get("quantity") or 0 for p in pts)
    sum_d = sum(p.get("defect_quantity") or 0 for p in pts)
    n_expected = ledger["reports_ok"] + ledger["replayed"]
    check("every delivered report stored (row count)", len(pts) == n_expected,
          f"MVE rows={len(pts)} delivered reports={n_expected} (live {ledger['reports_ok']} + replayed {ledger['replayed']})")
    check("OK total matches what the watcher sent", sum_q == ledger["delivered_ok"],
          f"MVE sum={sum_q:g} sent={ledger['delivered_ok']}")
    check("NG total matches what the watcher sent", sum_d == ledger["delivered_ng"],
          f"MVE sum={sum_d:g} sent={ledger['delivered_ng']}")

    # device sends naive UTC; MVE must store it as UTC (no timezone shift)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    lo = datetime.datetime.fromtimestamp(since_ms / 1000.0, datetime.timezone.utc) - datetime.timedelta(seconds=5)
    tss = []
    for p in pts:
        try:
            t = datetime.datetime.fromisoformat(p["ts"].replace("Z", "+00:00"))
            tss.append(t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc))
        except Exception:
            pass
    if tss:
        ok_ts = min(tss) >= lo and max(tss) <= now_utc + datetime.timedelta(seconds=5)
        check("timestamps stored as UTC (no timezone shift)", ok_ts,
              f"stored {min(tss).isoformat()} .. {max(tss).isoformat()}, run started {lo.isoformat()}")
        if ledger["replayed"]:
            check("offline backlog replayed with ORIGINAL timestamps (in order)",
                  tss == sorted(tss), "rows sorted by device time")
    if ledger.get("images_sent"):
        check("image reports accepted (2-step data + image upload)", True,
              f"{ledger['images_sent']} image(s) acknowledged by MVE")
    keys = set(mt.get("metric_keys", []))
    check("analog channels visible as chartable metrics", {"d0", "analog"} & keys != set(),
          f"metric_keys={sorted(keys)}")

    oee = requests.get(base + "/api/watchers/oee", params={"register_id": reg, "since_ms": since_ms}, timeout=60).json()
    check("OEE good count = OK total sent", oee.get("good") == ledger["delivered_ok"],
          f"oee.good={oee.get('good')} sent={ledger['delivered_ok']}")
    check("OEE defect count = NG total sent", oee.get("defect") == ledger["delivered_ng"],
          f"oee.defect={oee.get('defect')} sent={ledger['delivered_ng']}")
    tot = ledger["delivered_ok"] + ledger["delivered_ng"]
    if tot and oee.get("quality") is not None:
        want_q = ledger["delivered_ok"] / tot
        check("OEE quality = OK / (OK+NG)", abs(oee["quality"] - want_q) < 0.01,
              f"oee.quality={oee['quality']} expected={want_q:.4f}")
    print(f"  info  OEE response: {json.dumps(oee)}")
    return fails


def main():
    global MVE_BASE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="MVE base URL, e.g. http://192.168.0.143")
    ap.add_argument("--register-id", default="DUMMY-watcher-test",
                    help="machine id (the real device uses its hostname)")
    ap.add_argument("--duration", type=int, default=180, help="seconds of simulated production")
    ap.add_argument("--ppm", type=float, default=120, help="pieces per minute")
    ap.add_argument("--ng", type=float, default=0.05, help="NG ratio 0..1")
    ap.add_argument("--serial-channels", type=int, default=2, help="analog serial channels d0..dN (0 = none)")
    ap.add_argument("--heartbeat", type=int, default=300, help="idle report interval, s (firmware: 300)")
    ap.add_argument("--image-every", type=int, default=0, help="attach an image every N s (0 = off; firmware camera mode: 300)")
    ap.add_argument("--offline-first", type=int, default=0,
                    help="first N s MVE is 'unreachable': reports buffer in SQLite, then replay")
    ap.add_argument("--idle-last", type=int, default=0, help="last N s produce nothing (exercises the heartbeat)")
    ap.add_argument("--register", action="store_true", help="also set name/line/ideal_rate via /api/watchers/register")
    ap.add_argument("--name", default="Dummy watcher (test)", help="display name sent with --register")
    ap.add_argument("--line", default="TEST", help="line/group sent with --register")
    ap.add_argument("--workdir", default="./dummy_watcher_run", help="SQLite buffer + images + ledger")
    ap.add_argument("--verify-only", action="store_true", help="skip sending; re-check the last run's ledger")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    os.makedirs(args.workdir, exist_ok=True)
    ledger_path = os.path.join(args.workdir, "ledger.json")

    if args.verify_only:
        with open(ledger_path) as f:
            saved = json.load(f)
        fails = verify(base, saved["register_id"], saved["since_ms"], saved["ledger"], saved["ppm"])
        print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED")
        return 1 if fails else 0

    db_path = os.path.join(args.workdir, "monitait.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    since_ms = int(time.time() * 1000)
    t_end = time.time() + args.duration

    if args.register:
        r = requests.post(base + "/api/watchers/register", timeout=30, json={
            "register_id": args.register_id, "name": args.name, "line": args.line,
            "sensor": "simulated", "type": "dummy", "ideal_rate": args.ppm * 60})
        print(f"register -> HTTP {r.status_code} {r.text[:120]}")

    unreachable = "http://127.0.0.1:9"   # discard port: connection refused, like a dead link
    MVE_BASE = unreachable if args.offline_first > 0 else base
    arduino = FakeArduino(ppm=args.ppm, ng_ratio=args.ng, serial_channels=args.serial_channels)
    db = DB(db_path)
    camera = FakeCamera(os.path.join(args.workdir, "images")) if args.image_every > 0 else None
    counter = Counter(arduino, db, camera=camera, heartbeat=args.heartbeat,
                      picture_interval=args.image_every or 300)

    t0 = time.time()
    state = {"online": args.offline_first <= 0, "idle": False, "last_print": 0}

    def on_tick():
        global MVE_BASE
        el = time.time() - t0
        if not state["online"] and el >= args.offline_first:
            MVE_BASE = base
            state["online"] = True
            print(f"[{el:5.0f}s] link UP -> {base} ({db.pending()} buffered reports to replay)")
        if args.idle_last and not state["idle"] and time.time() >= t_end - args.idle_last:
            arduino.ppm = 0
            state["idle"] = True
            print(f"[{el:5.0f}s] production stopped (heartbeat-only from now)")
        if el - state["last_print"] >= 15:
            state["last_print"] = el
            print(f"[{el:5.0f}s] produced ok/ng={LEDGER['produced_ok']}/{LEDGER['produced_ng']} "
                  f"delivered={LEDGER['delivered_ok']}/{LEDGER['delivered_ng']} reports={LEDGER['reports_ok']} "
                  f"failed={LEDGER['reports_failed']} buffered={db.pending()} replayed={LEDGER['replayed']} "
                  f"images={LEDGER['images_sent']}")

    print(f"dummy watcher '{args.register_id}' -> {base}  duration={args.duration}s ppm={args.ppm} ng={args.ng} "
          f"offline_first={args.offline_first}s image_every={args.image_every}s heartbeat={args.heartbeat}s")
    if args.offline_first > 0:
        print(f"[    0s] link DOWN (MVE unreachable for {args.offline_first}s) -> reports go to the SQLite buffer")
    Thread(target=counter.db_checker, daemon=True).start()
    counter.run(args.register_id, t_end, on_tick=on_tick)

    # end of production: final flush of the counters + drain the offline buffer
    arduino.stop_thread = True
    time.sleep(0.5)
    a, b, _, _, _ = arduino.read_GPIO()
    if a or b:
        if watcher_update(args.register_id, quantity=a, defect_quantity=b, send_img=False,
                          extra_info={}, timestamp=datetime.datetime.utcnow()):
            _ledger_add(delivered_ok=a, delivered_ng=b, reports_ok=1)
            arduino.minus(a, b)
    drain_until = time.time() + 120
    while db.pending() and time.time() < drain_until:
        time.sleep(1)
    counter.stop_thread = True
    print(f"done: produced ok/ng={LEDGER['produced_ok']}/{LEDGER['produced_ng']} "
          f"delivered={LEDGER['delivered_ok']}/{LEDGER['delivered_ng']} still buffered={db.pending()} "
          f"| {json.dumps(LEDGER)}")

    with open(ledger_path, "w") as f:
        json.dump({"register_id": args.register_id, "since_ms": since_ms, "ppm": args.ppm,
                   "ledger": LEDGER}, f, indent=1)

    fails = []
    if LEDGER["delivered_ok"] != LEDGER["produced_ok"] or LEDGER["delivered_ng"] != LEDGER["produced_ng"] or db.pending():
        print("  FAIL  not everything produced was delivered (buffer not drained?)")
        fails.append("delivery")
    time.sleep(2)
    fails += verify(base, args.register_id, since_ms, LEDGER, args.ppm)
    print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
