"""Direct body-velocity keyboard for the real closed-loop backend; no planner."""
import argparse
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import sys
import termios
import time
import tty

from .direct_control import (KeyboardVelocityIntent, decode_velocity_keys, velocity_packet,
                             KEYBOARD_SPEED_KEYS, VELOCITY_LIMIT_KEYS)


def apply_forward_speed(config, speed):
    if not math.isfinite(speed) or not 0 < speed <= config['max_forward_speed']:
        raise ValueError(f'--speed must be positive and at most {config["max_forward_speed"]} m/s')
    return dict(config, keyboard_forward_speed=speed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--speed', type=float, help='W/S forward/backward speed in m/s (default from config)')
    parser.add_argument('--lateral-speed', type=float, help='A/D lateral speed in m/s')
    parser.add_argument('--yaw-rate', type=float, help='J/L yaw rate in rad/s')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    overrides = set()
    for option, key, limit in zip(('speed', 'lateral_speed', 'yaw_rate'), KEYBOARD_SPEED_KEYS, VELOCITY_LIMIT_KEYS):
        value = getattr(args, option)
        if value is not None:
            if not math.isfinite(value) or not 0 < value <= config[limit]:
                parser.error(f'{option} must be positive and at most {config[limit]}')
            config[key] = value
            overrides.add(key)
    if not sys.stdin.isatty():
        raise SystemExit('Run the real keyboard in an interactive terminal')
    intent = KeyboardVelocityIntent(config, speed_overrides=overrides)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', config['keyboard_status_port']))
    receiver.setblocking(False)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    original = termios.tcgetattr(sys.stdin)
    sequence = 0
    buffer = ''
    running = True
    last_status = None
    last_speeds = None
    last_loop = time.monotonic()
    def interrupted(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    def send():
        nonlocal sequence
        if intent.session is not None:
            sender.sendto(velocity_packet(intent.command, sequence, time.monotonic(), intent.session, intent.intent),
                          ('127.0.0.1', config['command_port']))
            sequence += 1
    print('REAL keyboard: W/Up forward, S/Down backward, A/D/Left/Right lateral; '
          'J/L turn left/right. Direction stays active until Space. Q exits.\n'
          'Speed priority: keyboard arguments > controller settings > local config.\n'
          'Space requests controlled deceleration. Emergency damping: remote L2+B or terminal 2 Ctrl+C.\n'
          'Waiting for DRY READY or REAL POLICY READY from terminal 2.', flush=True)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while running:
            now = time.monotonic()
            if now - last_loop > config['command_timeout']:
                intent.handle_key(' ', now)  # A frozen client must not resume travel.
            last_loop = now
            # Observe loss BEFORE reading queued/new status, so a brief loss cannot
            # silently re-enable an old direction when the backend comes back.
            intent.sync(now)
            for _ in range(64):
                try:
                    payload, _ = receiver.recvfrom(8192)
                except BlockingIOError:
                    break
                intent.state.receive(payload, now)
            intent.sync(now)
            status = intent.state.packet['phase'] if intent.state.fresh(now) else 'NO FRESH ROBOT STATE'
            if intent.ready(now) and intent.speeds() != last_speeds:
                last_speeds = intent.speeds()
                print(f'Keyboard speeds: W/S={last_speeds[0]:.2f} m/s, '
                      f'A/D={last_speeds[1]:.2f} m/s, J/L={last_speeds[2]:.2f} rad/s', flush=True)
            if status != last_status:
                print(f'Backend: {status}; press a direction key after ready.', flush=True)
                last_status = status
            if select.select([sys.stdin], [], [], 0)[0]:
                buffer += os.read(sys.stdin.fileno(), 64).decode(errors='ignore')
                keys, buffer = decode_velocity_keys(buffer)
                for key in keys:
                    if intent.handle_key(key, now):
                        c = intent.command
                        print(f'command=[{c.vx:.2f}, {c.vy:.2f}, {c.wz:.2f}]', flush=True)
                    elif not intent.ready(now):
                        print('Not ready: movement ignored. Wait for terminal 2 READY.', flush=True)
                    if key in ('q', '\x03'):
                        running = False
                        break
            send()
            time.sleep(.05)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original)
        intent.handle_key(' ', time.monotonic())
        for _ in range(3):
            send()
            time.sleep(.02)
        receiver.close()
        sender.close()


if __name__ == '__main__':
    main()
