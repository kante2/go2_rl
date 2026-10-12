"""Bounded console reporting and timing diagnostics; no robot I/O."""
from collections import deque
import gc
import queue
import threading
import time


class GCPauseMonitor:
    """Observe collection pauses without disabling GC or logging in callbacks."""
    def __init__(self):
        self.events = deque(maxlen=32)
        self.started = None
        gc.callbacks.append(self.callback)

    def callback(self, phase, info):
        if phase == 'start':
            self.started = time.monotonic()
        elif phase == 'stop' and self.started is not None:
            ended = time.monotonic()
            self.events.append(dict(monotonic=self.started, duration_ms=(ended - self.started) * 1000,
                                    generation=info['generation'], collected=info['collected'],
                                    thread_id=threading.get_ident()))
            self.started = None

    def snapshot(self):
        return list(self.events)

    def close(self):
        if self.callback in gc.callbacks:
            gc.callbacks.remove(self.callback)


class PublishTiming:
    """Separate message preparation/CRC from DDS, with wall and thread CPU time."""
    def __init__(self):
        self.previous = self.started = time.monotonic()
        self.previous_cpu = self.started_cpu = time.thread_time()
        self.values = dict(monotonic=self.started)

    def mark(self, stage):
        now, cpu = time.monotonic(), time.thread_time()
        self.values[stage + '_ms'] = (now - self.previous) * 1000
        self.values[stage + '_cpu_ms'] = (cpu - self.previous_cpu) * 1000
        self.previous, self.previous_cpu = now, cpu

    def finish(self):
        self.values['total_ms'] = (time.monotonic() - self.started) * 1000
        self.values['total_cpu_ms'] = (time.thread_time() - self.started_cpu) * 1000
        return self.values


class AsyncStatusPrinter:
    """A blocked terminal must not block motor updates or entry into damping."""
    def __init__(self, sink=None):
        self.sink = sink or (lambda line: print(line, flush=True))
        self.pending = queue.Queue(maxsize=8)
        self.stop = threading.Event()
        self.dropped = 0
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, line):
        try:
            self.pending.put_nowait(line)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while not self.stop.is_set() or not self.pending.empty():
            try:
                line = self.pending.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                self.sink(line)
            except Exception as error:
                self.error = str(error)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=.2)


class MotorLoopTiming:
    def __init__(self, started, deadline):
        self.started = self.previous = started
        self.values = dict(monotonic=started,
                           wake_lateness_ms=max(0., started - deadline) * 1000)

    def mark(self, stage):
        now = time.monotonic()
        self.values[stage + '_ms'] = (now - self.previous) * 1000
        self.previous = now

    def finish(self, next_deadline):
        now = time.monotonic()
        self.values.update(loop_ms=(now - self.started) * 1000,
                           deadline_lateness_ms=max(0., now - next_deadline) * 1000)
        return now, self.values
