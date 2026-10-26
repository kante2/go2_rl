"""ROS-independent, causal state resampling and frozen model-A inference."""
from collections import deque
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'contact_force_velocity'))
from lowstate_kinematics import Go2Kinematics, LEGS, FEATURE_SCHEMA, vertical_motion

HZ = 500
DT = 1 / HZ
FAMILIES = ('force', 'velocity', 'force_change', 'velocity_change')
LAGS = (0, 10, 20, 40)
FEATURE_NAMES = [f'{group}_{leg}__lag{lag}ms' for lag in LAGS for group in FAMILIES for leg in LEGS]


def default_model_path():
    folder = ROOT / 'contact_force_velocity/models/causal_forecast'
    selection = json.loads((folder / 'selection.json').read_text())
    return folder / (selection['ablation'] + '.joblib')


class Predictor:
    def __init__(self, path=None):
        self.path = Path(path) if path else default_model_path()
        self.bundle = joblib.load(self.path)
        b = self.bundle
        if b['task'] != 'causal_forecast' or not b['causal']:
            raise ValueError('Only causal model A is supported')
        if b.get('feature_schema') != FEATURE_SCHEMA:
            raise ValueError('Model must be retrained for the LowState-only feature schema')
        self.columns = np.asarray(b['columns'], dtype=int)
        if [FEATURE_NAMES[i] for i in self.columns] != b['feature_names']:
            raise ValueError('Frozen model feature schema does not match the streaming schema')
        self.kinematics = Go2Kinematics()
        self.history = deque(maxlen=24)  # 40 ms lag + 6 ms derivative = 23 steps.
        self.segment_count = 0
        self.reset()

    def reset(self):
        self.history.clear()
        self.last_tick = None
        self.segment_index = 0
        self.segment_count += 1

    def physical(self, low):
        return self.physical_batch(np.asarray(low)[None, :])[0]

    def physical_batch(self, low):
        low = np.asarray(low, dtype=float)
        return np.column_stack([low[:, :4], vertical_motion(low, self.kinematics)])

    def push_grid(self, tick, low):
        try:
            physical = self.physical(low)
        except ValueError:
            self.reset()
            raise
        return self.push_physical(tick, physical)

    def push_physical(self, tick, physical):
        if self.last_tick is not None and tick != self.last_tick + 1:
            self.reset()
        self.last_tick = tick
        self.history.append(physical)
        index = self.segment_index
        self.segment_index += 1
        if len(self.history) < 24 or (index - 23) % 10:
            return None
        h = np.asarray(self.history)
        x = np.concatenate([
            np.r_[h[-1-lag//2], (h[-1-lag//2] - h[-4-lag//2]) / .006]
            for lag in LAGS
        ])
        return x

    def predict(self, features):
        with threadpool_limits(limits=1):
            return self.bundle['model'].predict(np.asarray(features)[:, self.columns])


def low_vector(message):
    motors = message.motor_state[:12]
    if len(motors) != 12:
        raise ValueError('LowState needs at least 12 motor states')
    return np.r_[message.foot_force, message.imu_state.quaternion,
                 message.imu_state.gyroscope, [m.q for m in motors], [m.dq for m in motors]]


class StateStream:
    """Causal LowState sample-and-hold on the training 500 Hz grid."""
    def __init__(self, predictor, max_backlog=.1):
        self.predictor = predictor
        self.low = deque(maxlen=5000)
        self.next_tick = None
        self.max_backlog = max_backlog
        self.invalid_grid_count = 0
        self.skipped_backlog_count = 0

    def add(self, timestamp, value):
        value = np.asarray(value, dtype=float)
        if not np.isfinite(timestamp) or (self.low and timestamp <= self.low[-1][0]):
            raise ValueError('Receive timestamps must be finite and strictly increasing')
        if value.shape != (35,) or not np.isfinite(value).all():
            raise ValueError('Invalid LowState vector')
        self.low.append((float(timestamp), value.copy()))

    def drain(self, now=None):
        if not self.low:
            return []
        start, end = self.low[0][0], self.low[-1][0]
        if self.next_tick is None:
            self.next_tick = int(np.ceil(start * HZ))
        if now is not None and now-end > .020:
            self.predictor.reset()
            self.next_tick = max(self.next_tick, int(np.ceil(now * HZ)))
            return []
        first_allowed = int(np.ceil((end-self.max_backlog)*HZ))
        if self.next_tick < first_allowed:
            self.skipped_backlog_count += first_allowed-self.next_tick
            self.next_tick = first_allowed
            self.predictor.reset()
        stop = int(np.floor(end*HZ))
        if stop < self.next_tick:
            return []
        snapshot = list(self.low)
        times = np.array([row[0] for row in snapshot])
        features, anchors, ages = [], [], []
        grid, valid_low = [], []
        for tick in range(self.next_tick, stop+1):
            t = tick/HZ
            index = int(np.searchsorted(times,t,side='right')-1)
            if index < 0 or t-times[index] > .020+1e-9 or np.linalg.norm(snapshot[index][1][4:8]) < 1e-8:
                self.invalid_grid_count += 1
                grid.append((tick,None,None))
                continue
            grid.append((tick,len(valid_low),t-times[index]))
            valid_low.append(snapshot[index][1])
        physical = self.predictor.physical_batch(valid_low) if valid_low else []
        for tick,index,age in grid:
            if index is None:
                self.predictor.reset()
                continue
            x = self.predictor.push_physical(tick,physical[index])
            if x is not None:
                features.append(x); anchors.append(tick/HZ); ages.append(age)
        self.next_tick = stop+1
        while len(self.low)>2 and self.low[1][0]<end-1:
            self.low.popleft()
        if not features:
            return []
        predictions = self.predictor.predict(features)
        return [dict(anchor_time=t,target_time=t+.020,target_start=t+.010,target_stop=t+.030,
                     predicted_noise_dbfs=float(prediction),low_age_ms=age*1000)
                for t,prediction,age in zip(anchors,predictions,ages)]
