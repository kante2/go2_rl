"""Local direct-velocity and measured-state protocols; no hardware I/O."""
import json
import math
import uuid

import numpy as np

from ..core import CommandWatchdog, Velocity, command_packet
from .real_control import gravity_body

KEYBOARD_SPEED_KEYS = ('keyboard_forward_speed', 'keyboard_lateral_speed', 'keyboard_yaw_rate')
VELOCITY_LIMIT_KEYS = ('max_forward_speed', 'max_lateral_speed', 'max_yaw_rate')


def keyboard_speed_settings(config):
    limits = [config[key] for key in VELOCITY_LIMIT_KEYS]
    speeds = [min(config[key], limit) for key, limit in zip(KEYBOARD_SPEED_KEYS, limits)]
    return dict(keyboard_speeds=speeds, velocity_limits=limits)


def velocity_packet(command, sequence, now, session, intent):
    data = json.loads(command_packet(command, sequence, now, session))
    data.update(kind='go2_direct_velocity', intent=intent)
    return json.dumps(data, allow_nan=False).encode()


class DirectCommandGate:
    """Latch timeout until a new key intent; never resume an expired command."""
    def __init__(self, config):
        self.config = config
        self.session = None
        self.intent = None
        self.expired_intent = None
        self.reason = 'waiting for a direction key'
        self._new_watchdog()

    def _new_watchdog(self):
        c = self.config
        self.watchdog = CommandWatchdog(c['command_timeout'], c['max_forward_speed'],
                                       c['max_lateral_speed'], c['max_yaw_rate'])

    def set_session(self, session):
        if session != self.session:
            self.session = session
            self.intent = self.expired_intent = None
            self._new_watchdog()
            self.reason = 'waiting for a direction key'

    def sample(self, now):
        if self.intent is not None and now - self.watchdog.stamp > self.config['command_timeout']:
            self.expired_intent = self.intent
            self.reason = 'keyboard timeout; press a direction key again'
        if self.intent is None or self.intent == self.expired_intent:
            return Velocity()
        return self.watchdog.sample(now)

    def receive(self, payload, now):
        self.sample(now)  # Expire before considering a late resumed sender.
        try:
            packet = json.loads(payload)
            intent = packet['intent']
            if (packet['kind'] != 'go2_direct_velocity' or packet['session'] != self.session
                    or not isinstance(intent, str) or not intent or intent == self.expired_intent):
                return False
            if not self.watchdog.receive(payload, now):
                return False
        except (ValueError, KeyError, TypeError, AttributeError):
            return False
        self.intent = intent
        self.reason = 'active' if self.watchdog.command != Velocity() else 'stopped by keyboard'
        return True


def decode_velocity_keys(buffer):
    keys = []
    while buffer:
        if buffer[0] == '\x1b':
            if len(buffer) < 3:
                break
            if buffer[1] == '[':
                key = {'A': 'w', 'B': 's', 'C': 'd', 'D': 'a'}.get(buffer[2])
                if key:
                    keys.append(key)
                buffer = buffer[3:]
            else:
                buffer = buffer[2:]
        else:
            key, buffer = buffer[0].lower(), buffer[1:]
            if key in ('w', 's', 'a', 'd', 'j', 'l', 'q', ' ', '\x03'):
                keys.append(key)
    return keys, buffer


class MeasuredStateReceiver:
    """Validate measured joints/IMU; no extrapolation when the source is stale."""
    def __init__(self, checkpoint_sha256=None, timeout=.3):
        self.checkpoint_sha256 = checkpoint_sha256
        self.timeout = timeout
        self.packet = None

    def fresh(self, now):
        return self.packet is not None and 0 <= now - self.packet['monotonic'] <= self.timeout

    def receive(self, payload, now):
        try:
            p = json.loads(payload)
            if p['version'] != 1 or p['kind'] != 'go2_measured_state':
                return False
            if self.checkpoint_sha256 is not None and p['checkpoint_sha256'] != self.checkpoint_sha256:
                return False
            if (not isinstance(p['session'], str) or not p['session']
                    or type(p['sequence']) is not int or p['sequence'] < 0
                    or type(p['ready']) is not bool or type(p['execute']) is not bool
                    or not isinstance(p['phase'], str)):
                return False
            stamp = p['monotonic'] = float(p['monotonic'])
            if not math.isfinite(stamp) or not -.005 <= now - stamp <= self.timeout:
                return False
            if self.packet is not None:
                if stamp <= self.packet['monotonic']:
                    return False
                if p['session'] == self.packet['session'] and p['sequence'] <= self.packet['sequence']:
                    return False
            for key, size in (('q_policy', 12), ('dq_policy', 12), ('gyro', 3), ('command', 3)):
                values = np.asarray(p[key], dtype=float)
                if values.shape != (size,) or not np.isfinite(values).all():
                    return False
            gravity_body(p['quaternion_wxyz'])
            if 'keyboard_speeds' in p or 'velocity_limits' in p:
                for field in ('keyboard_speeds', 'velocity_limits'):
                    values = p[field]
                    if (not isinstance(values, list) or len(values) != 3
                            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)):
                        return False
                if any(speed > limit for speed, limit in zip(p['keyboard_speeds'], p['velocity_limits'])):
                    return False
            self.packet = p
            return True
        except (ValueError, TypeError, KeyError, OverflowError):
            return False


class KeyboardVelocityIntent:
    """A source change/loss clears travel; only a new key can start it again."""
    def __init__(self, config, speed_overrides=()):
        self.config = config
        self.speed_overrides = set(speed_overrides)
        self.state = MeasuredStateReceiver(timeout=config['sensor_timeout'])
        self.session = None
        self.intent = uuid.uuid4().hex
        self.command = Velocity()

    def speeds(self):
        """Explicit keyboard options > controller defaults > local JSON defaults."""
        packet = self.state.packet or {}
        advertised = packet.get('keyboard_speeds')
        limits = packet.get('velocity_limits')
        result = []
        for i, (key, limit_key) in enumerate(zip(KEYBOARD_SPEED_KEYS, VELOCITY_LIMIT_KEYS)):
            value = self.config[key]
            if advertised is not None and key not in self.speed_overrides:
                value = advertised[i]
            limit = self.config[limit_key]
            if limits is not None:
                limit = min(limit, limits[i])
            result.append(min(value, limit))
        return tuple(result)

    def ready(self, now):
        return self.state.fresh(now) and self.state.packet['ready']

    def sync(self, now):
        session = self.state.packet['session'] if self.state.packet else None
        reset = session != self.session or not self.ready(now)
        if reset:
            self.command = Velocity()
            self.intent = uuid.uuid4().hex
        self.session = session

    def handle_key(self, key, now):
        self.sync(now)
        if key in (' ', 'q', '\x03'):
            self.command = Velocity()
        else:
            if not self.ready(now):
                return False
            c = self.config
            forward, lateral, yaw = self.speeds()
            directions = {'w': Velocity(forward, 0, 0),
                          's': Velocity(-forward, 0, 0),
                          'a': Velocity(0, lateral, 0),
                          'd': Velocity(0, -lateral, 0),
                          'j': Velocity(0, 0, yaw),
                          'l': Velocity(0, 0, -yaw)}
            if key not in directions:
                return False
            self.command = directions[key].bounded(c['max_forward_speed'], c['max_lateral_speed'], c['max_yaw_rate'])
        self.intent = uuid.uuid4().hex
        return True
