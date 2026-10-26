"""Optional bounded-memory SSH PCM monitor matching sync_record_rl.py format."""
from collections import deque
import os
import shlex
import subprocess
import tempfile
import threading
import time

import numpy as np

FS = 48000


def pcm_dbfs(samples):
    values = np.asarray(samples, dtype=float)
    if not len(values):
        return None
    rms = np.sqrt(max(float(np.mean(values * values) - np.mean(values)**2), 0.))
    return float(20 * np.log10(max(rms, 1e-12) / 32768.))


class RemotePCM:
    def __init__(self, host, device):
        if not host or host.startswith('-') or any(c.isspace() for c in host):
            raise ValueError('Invalid SSH host')
        command = shlex.join(['arecord', '-q', '-D', device, '-f', 'S16_LE', '-r', str(FS),
                              '-c', '1', '-t', 'raw', '--period-time=10000', '--buffer-time=100000'])
        # Same scoped child cleanup as sync_record_rl.py; never kills unrelated arecord processes.
        remote = (
            'exec 3<&0; ' + command + ' </dev/null 3<&- & recorder=$!; '
            'trap \'kill "$recorder" 2>/dev/null || :\' HUP INT TERM EXIT; '
            '(cat >/dev/null; kill "$recorder" 2>/dev/null || :) <&3 3<&- >&2 & watcher=$!; '
            'exec 3<&-; wait "$recorder"; status=$?; '
            'kill "$watcher" 2>/dev/null || :; wait "$watcher" 2>/dev/null; '
            'trap - HUP INT TERM EXIT; exit "$status"'
        )
        self.stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                                      '-o', 'Compression=no', '-o', 'ServerAliveInterval=2',
                                      '-o', 'ServerAliveCountMax=3', host, remote],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0)
        self.lock = threading.Lock()
        self.chunks = deque()
        self.count = 0
        self.t0 = float('inf')
        self.last_received = None
        self.error = ''
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        pending = b''
        try:
            while True:
                data = os.read(self.proc.stdout.fileno(), 65536)
                if not data:
                    break
                received = time.monotonic()
                pending += data
                size = len(pending) // 2 * 2
                samples = np.frombuffer(pending[:size], dtype='<i2').copy()
                pending = pending[size:]
                if not len(samples):
                    continue
                with self.lock:
                    self.chunks.append((self.count, samples))
                    self.count += len(samples)
                    self.t0 = min(self.t0, received - self.count / FS)
                    self.last_received = received
                    while self.chunks and self.chunks[0][0] + len(self.chunks[0][1]) < self.count - FS*5:
                        self.chunks.popleft()
        except Exception as error:
            self.error = str(error)

    def latest(self):
        """Latest 20 ms DC-removed RMS; intentionally NOT paired to a prediction."""
        with self.lock:
            if self.count < 960 or not self.chunks:
                return None
            selected = []
            needed = 960
            for _, samples in reversed(self.chunks):
                take = min(needed, len(samples)); selected.append(samples[-take:]); needed -= take
                if not needed:
                    break
            if needed:
                return None
            return dict(latest_audio_dbfs=pcm_dbfs(np.concatenate(selected[::-1])),
                        audio_receive_age_ms=(time.monotonic()-self.last_received)*1000,
                        audio_time_estimate=self.t0+self.count/FS,
                        audio_alive=self.thread.is_alive() and self.proc.poll() is None,
                        audio_clipping=any(np.any((x == -32768) | (x == 32767)) for x in selected))

    def level_windows(self, after_sample=0):
        """New nonoverlapping 20 ms PCM windows for plotting, using an estimated clock.

        Window IDs use sample counts, so repeated polls never duplicate a window.
        As in the recorder, nominal-rate timing includes unknown SSH delay.
        """
        with self.lock:
            if not self.chunks or self.count < 960 or self.count//960*960 <= after_sample:
                return []
            first = self.chunks[0][0]
            first_end = max((after_sample//960+1)*960, ((first+959)//960)*960+960)
            needed = [(start, chunk) for start, chunk in self.chunks if start+len(chunk) > first_end-960]
            if not needed:
                return []
            first = needed[0][0]
            samples = np.concatenate([chunk for _, chunk in needed])
            windows = []
            for end in range(first_end, self.count+1, 960):
                values = samples[end-960-first:end-first]
                windows.append(dict(end_sample=end, time=self.t0+(end-480)/FS,
                                    value=pcm_dbfs(values),
                                    clipping=bool(np.any((values == -32768) | (values == 32767)))))
            return windows

    def close(self):
        try:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill(); self.proc.wait()
            self.thread.join(timeout=2)
        finally:
            self.proc.stdout.close()
            self.stderr.close()
