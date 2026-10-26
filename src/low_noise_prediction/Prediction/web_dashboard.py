"""Local, read-only HTTP dashboard with bounded telemetry storage (stdlib only)."""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit


class Dashboard:
    def __init__(self, host='127.0.0.1', port=8080):
        self.lock = threading.Lock()
        self.points = deque(maxlen=8000)
        self.sequence = 0
        self.status = dict(audio_enabled=False, audio_alive=False, low_received_time=None,
                           audio_received_time=None, simulated=False, predictions=0,
                           model='Model A · LowState only')
        dashboard = self
        assets = {'/': ('dashboard.html', 'text/html; charset=utf-8'),
                  '/dashboard.js': ('dashboard.js', 'text/javascript; charset=utf-8'),
                  '/dashboard.css': ('dashboard.css', 'text/css; charset=utf-8')}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urlsplit(self.path)
                if url.path == '/api/state':
                    try:
                        since = int(parse_qs(url.query).get('since', ['0'])[0])
                        if since < 0:
                            raise ValueError()
                    except ValueError:
                        self.send_error(400, 'Invalid telemetry cursor')
                        return
                    body = json.dumps(dashboard.snapshot(since), allow_nan=False).encode()
                    content_type = 'application/json; charset=utf-8'
                elif url.path in assets:
                    name, content_type = assets[url.path]
                    body = (Path(__file__).parent/name).read_bytes()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self'; object-src 'none'; frame-ancestors 'none'")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://{host}:{self.server.server_port}/'

    def update_status(self, **values):
        with self.lock:
            self.status.update(values)

    def append(self, kind, timestamp, value, **extra):
        if kind not in ('measured', 'predicted') or not math.isfinite(timestamp) or not math.isfinite(value):
            raise ValueError('Invalid dashboard point')
        with self.lock:
            self.sequence += 1
            self.points.append(dict(sequence=self.sequence, kind=kind, time=float(timestamp),
                                    value=float(value), **extra))
            cutoff = time.monotonic()-65
            while self.points and self.points[0]['time'] < cutoff:
                self.points.popleft()

    def snapshot(self, since=0):
        now = time.monotonic()
        with self.lock:
            status = self.status.copy()
            for source in ('low', 'audio'):
                received = status.pop(source+'_received_time', None)
                status[source+'_age_ms'] = None if received is None else max(0., (now-received)*1000)
            # A server restart invalidates a browser's old cursor.
            reset = since > self.sequence
            points = [p for p in self.points if p['sequence'] > (0 if reset else since) and p['time'] >= now-60]
            return dict(now=now, cursor=self.sequence, reset=reset, points=points, status=status)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description='Preview the HTML dashboard using clearly labelled simulated data.')
    parser.add_argument('--demo', action='store_true', required=True)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    dashboard = Dashboard(port=args.port)
    print('SIMULATED DATA ONLY: '+dashboard.url, flush=True)
    started = time.monotonic()
    def add_point(t):
        elapsed = t-started
        measured = -33 + 6*math.sin(elapsed*1.4) + 2*math.sin(elapsed*4.1)
        predicted = -33 + 5.3*math.sin(elapsed*1.4+.06) + 1.3*math.sin(elapsed*4.1)
        dashboard.append('measured', t-.010, measured)
        dashboard.append('predicted', t+.020, predicted)
    for i in range(1500):
        add_point(started-30+i*.020)
    try:
        count = 0
        while True:
            now = time.monotonic()
            add_point(now)
            count += 1
            dashboard.update_status(simulated=True, audio_enabled=True, audio_alive=True,
                                    low_received_time=now, audio_received_time=now,
                                    predictions=count, remaining_horizon_ms=15.,
                                    model='Simulated signals · No robot or HGB inference')
            time.sleep(.020)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()


if __name__ == '__main__':
    main()
