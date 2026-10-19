"""Local UDP receiver and browser dashboard. No motor or navigation output."""

import argparse
from collections import deque
import csv
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import signal
import socket
import threading
import time
from urllib.parse import parse_qs, urlsplit

from .metrics import FEET

WEB = Path(__file__).parent / "web"
ARRAY_FIELDS = ("force_n", "peak_n", "landing_impulse_ns", "landing_peak_n", "landing_time_s")


def validate_packet(packet):
    if not isinstance(packet, dict) or packet.get("version") != 1 or packet.get("source") != "mujoco":
        raise ValueError("Unknown telemetry format")
    if packet.get("feet") != list(FEET):
        raise ValueError("Invalid foot order")
    for key in ("session", "model", "checkpoint"):
        if not isinstance(packet.get(key), str) or not 0 < len(packet[key]) <= 200:
            raise ValueError(key)
    for key in ("sequence", "reset_counter"):
        if type(packet.get(key)) is not int or packet[key] < 0:
            raise ValueError(key)
    for key in ("sim_time", "physics_dt"):
        value = packet.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(key)
    if packet["physics_dt"] <= 0:
        raise ValueError("physics_dt")
    q = packet.get("quietness")
    if q is not None and (type(q) not in (float, int) or not math.isfinite(q) or not 0 <= q <= 1):
        raise ValueError("quietness")
    for key in (*ARRAY_FIELDS, "command", "linear_velocity", "angular_velocity"):
        values = packet.get(key)
        length = 4 if key in ARRAY_FIELDS else 3
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(key)
        for value in values:
            if value is None and key.startswith("landing_"):
                continue
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(key)
            if key in ARRAY_FIELDS and value < 0:
                raise ValueError(key)
    if not isinstance(packet.get("contact"), list) or len(packet["contact"]) != 4 or any(type(x) is not bool for x in packet["contact"]):
        raise ValueError("contact")
    counts = packet.get("landing_count")
    if not isinstance(counts, list) or len(counts) != 4 or any(type(x) is not int or x < 0 for x in counts):
        raise ValueError("landing_count")
    return packet


class TelemetryStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = deque(maxlen=12000)  # About ten minutes at 20 Hz.
        self.generation = None
        self.latest = None
        self.received_at = None
        self.counter = 0
        self.invalid = 0
        self.dropped = 0

    def accept(self, packet, now=None):
        packet = validate_packet(packet)
        now = time.monotonic() if now is None else now
        generation = (packet["session"], packet["reset_counter"])
        with self.lock:
            # Do not mix two live simulators. A disconnected source may be replaced.
            if self.latest and packet["session"] != self.latest["session"] and now - self.received_at < 1.0:
                return False
            if self.latest and packet["session"] == self.latest["session"]:
                if packet["sequence"] <= self.latest["sequence"]:
                    return False
            if generation != self.generation:
                self.history.clear()
                self.latest = None
                self.dropped = 0
                self.generation = generation
            sample = dict(packet)
            sample["rtf"] = None
            if self.latest:
                delta = now - self.received_at
                sample["rtf"] = (packet["sim_time"] - self.latest["sim_time"]) / delta if delta > 0 else None
                self.dropped += max(0, packet["sequence"] - self.latest["sequence"] - 1)
            self.counter += 1
            sample["id"] = self.counter
            self.history.append(sample)
            self.latest = sample
            self.received_at = now
        return True

    def snapshot(self, after=0):
        with self.lock:
            age = time.monotonic() - self.received_at if self.received_at is not None else None
            samples = [item for item in self.history if item["id"] > after][-600:]
            return dict(latest=self.latest, samples=samples, age_s=age,
                        connected=age is not None and age < 1.0,
                        invalid=self.invalid, dropped=self.dropped,
                        generation=list(self.generation) if self.generation else None)

    def csv(self):
        with self.lock:
            rows = list(self.history)
        output = io.StringIO()
        writer = csv.writer(output)
        fields = ("force_n", "peak_n", "landing_impulse_ns", "landing_peak_n", "landing_time_s", "landing_count", "contact")
        writer.writerow(["model", "checkpoint", "session", "reset_counter", "sequence", "sim_time_s", "quietness",
                         "command_vx_mps", "command_vy_mps", "command_wz_radps", "vx_mps", "vy_mps", "vz_mps",
                         "wx_radps", "wy_radps", "wz_radps", "physics_dt_s"] +
                        [f"{foot}_{field}" for field in fields for foot in FEET])
        for row in rows:
            writer.writerow([row[k] for k in ("model", "checkpoint", "session", "reset_counter", "sequence", "sim_time", "quietness")] +
                            row["command"] + row["linear_velocity"] + row["angular_velocity"] + [row["physics_dt"]] +
                            [value for field in fields for value in row[field]])
        return output.getvalue().encode("utf-8-sig")


def receive(sock, store, stop):
    while not stop.is_set():
        try:
            data, _ = sock.recvfrom(65535)
            store.accept(json.loads(data))
        except socket.timeout:
            continue
        except (ValueError, TypeError, KeyError, UnicodeError):
            with store.lock:
                store.invalid += 1
        except OSError:
            break


class DashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, store, **kwargs):
        self.store = store
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass

    def do_GET(self):
        url = urlsplit(self.path)
        attachment = False
        if url.path == "/api/state":
            try:
                after = max(0, int(parse_qs(url.query).get("after", ["0"])[0]))
            except ValueError:
                self.send_error(400)
                return
            content = json.dumps(self.store.snapshot(after), allow_nan=False).encode()
            mime = "application/json"
        elif url.path == "/api/export.csv":
            content = self.store.csv()
            mime, attachment = "text/csv; charset=utf-8", True
        elif url.path in ("/", "/app.js", "/style.css"):
            name = "index.html" if url.path == "/" else url.path[1:]
            content = (WEB / name).read_bytes()
            mime = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8",
                    "style.css": "text/css; charset=utf-8"}[name]
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        if attachment:
            self.send_header("Content-Disposition", 'attachment; filename="go2-gait.csv"')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-port", type=int, default=8081)
    parser.add_argument("--udp-port", type=int, default=9910)
    args = parser.parse_args(argv)
    if any(not 1 <= port <= 65535 for port in (args.http_port, args.udp_port)):
        parser.error("Ports must be in [1, 65535]")
    store, stop = TelemetryStore(), threading.Event()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server = None
    try:
        udp.bind(("127.0.0.1", args.udp_port))
        udp.settimeout(.25)
        server = ThreadingHTTPServer(("127.0.0.1", args.http_port), partial(DashboardHandler, store=store))
        worker = threading.Thread(target=receive, args=(udp, store, stop), daemon=True)
        worker.start()
        def interrupt(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        print(f"Go2 gait monitor: http://localhost:{args.http_port} | UDP {args.udp_port}", flush=True)
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    except OSError as error:
        parser.exit(1, f"Cannot start monitor: {error}\n")
    finally:
        stop.set()
        udp.close()
        if server is not None:
            server.server_close()


if __name__ == "__main__":
    main()
