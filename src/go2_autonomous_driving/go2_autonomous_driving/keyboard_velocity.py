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

from .direct_control import KeyboardVelocityIntent, decode_velocity_keys, velocity_packet


def apply_forward_speed(config, speed):
    if not math.isfinite(speed) or not 0 < speed <= config['max_forward_speed']:
        raise ValueError(f'--speed must be positive and at most {config["max_forward_speed"]} m/s')
    return dict(config, keyboard_forward_speed=speed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--speed', type=float, help='W/S forward/backward speed in m/s (default from config)')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.speed is not None:
        try:
            config = apply_forward_speed(config, args.speed)
        except ValueError as error:
            parser.error(str(error))
    if not sys.stdin.isatty():
        raise SystemExit('Run the real keyboard in an interactive terminal')
    intent = KeyboardVelocityIntent(config)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', config['keyboard_status_port']))
    receiver.setblocking(False)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    original = termios.tcgetattr(sys.stdin)
    sequence = 0
    buffer = ''
    running = True
    last_status = None
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
          f'W/S speed: {config["keyboard_forward_speed"]:.2f} m/s.\n'
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
