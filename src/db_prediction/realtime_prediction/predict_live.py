#!/usr/bin/env python3
"""Subscribe to Unitree Go2 states and run the frozen model A (no robot commands)."""
import argparse
import csv
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import numpy as np

from stream_model import Predictor, StateStream, low_vector
from remote_audio import RemotePCM


FIELDS = ['anchor_time', 'target_time', 'target_start', 'target_stop', 'predicted_noise_dbfs',
          'low_age_ms', 'emitted_time', 'processing_age_ms', 'remaining_horizon_ms']


def replay_npz(path, predictor, emit):
    """Replay receive-time streams saved by sync_record_rl.py, without ROS/audio."""
    state = StateStream(predictor)
    count = 0
    with np.load(path, allow_pickle=False) as data:
        lt = data['ls_t']
        if not len(lt):
            raise ValueError('Replay needs a nonempty ls_t stream')
        low = np.column_stack([data['ls_ff'], data['ls_quaternion'], data['ls_gyro'], data['ls_q'], data['ls_dq']])
        next_drain = lt[0]
        for now, vector in zip(lt, low):
            now = float(now)
            state.add(now, vector)
            if now >= next_drain:
                for row in state.drain():
                    emit(row, now); count += 1
                next_drain = now+.010
        for row in state.drain():
            emit(row, now); count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Defaults to the saved model-A selection')
    parser.add_argument('--low', default='/lowstate')
    parser.add_argument('--output-topic', default='/noise_prediction/model_a')
    parser.add_argument('--csv', type=Path, help='New CSV path (existing files are never overwritten)')
    parser.add_argument('--replay-npz', type=Path, help='Offline replay of sync_record_rl.py data; no ROS required')
    parser.add_argument('--audio-host', help='Optional SSH host, e.g. unitree@192.168.123.18')
    parser.add_argument('--card', default='Microphone')
    parser.add_argument('--device', help='ALSA name; overrides --card, exactly as in sync_record_rl.py')
    parser.add_argument('--web-port', type=int, help='Serve the live HTML dashboard, e.g. 8080')
    parser.add_argument('--web-host', default='127.0.0.1', help='Dashboard bind address; defaults to local PC only')
    args, ros_args = parser.parse_known_args()
    if args.replay_npz and args.audio_host:
        parser.error('--audio-host is for live ROS mode only')
    if args.replay_npz and args.web_port is not None:
        parser.error('--web-port is for live ROS mode; use web_dashboard.py --demo for an HTML preview')
    if args.web_port is not None and not 1 <= args.web_port <= 65535:
        parser.error('--web-port must be between 1 and 65535')
    if args.replay_npz and ros_args:
        parser.error('Unrecognized replay arguments: ' + ' '.join(ros_args))
    predictor = Predictor(args.model)
    print(f'Model: {predictor.path}\nOutput: predicted raw PCM dBFS, not calibrated SPL', flush=True)
    with ExitStack() as resources:
        writer = None
        output = None
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            output = resources.enter_context(args.csv.open('x', newline=''))
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
        last_print = -float('inf')

        def emit(row, emitted):
            nonlocal last_print
            row.update(emitted_time=emitted, processing_age_ms=(emitted-row['anchor_time'])*1000,
                       remaining_horizon_ms=(row['target_time']-emitted)*1000)
            if writer:
                writer.writerow({key: row[key] for key in FIELDS})
            if emitted - last_print >= 1:
                print(f"prediction={row['predicted_noise_dbfs']:.2f} dBFS | "
                      f"state age={row['processing_age_ms']:.1f} ms | "
                      f"remaining horizon={row['remaining_horizon_ms']:.1f} ms", flush=True)
                last_print = emitted
                if output:
                    output.flush()

        if args.replay_npz:
            print(f'Replay complete: {replay_npz(args.replay_npz, predictor, emit)} predictions')
            return
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            from unitree_go.msg import LowState
            from std_msgs.msg import String, Float64
        except ImportError as error:
            parser.error('Source ROS 2 and the unitree_ros2 workspace before running: ' + str(error))

        audio = None
        if args.audio_host:
            audio = RemotePCM(args.audio_host, args.device or f'plughw:CARD={args.card},DEV=0')
            resources.callback(audio.close)

        dashboard = None
        if args.web_port is not None:
            from web_dashboard import Dashboard
            dashboard = Dashboard(args.web_host, args.web_port)
            resources.callback(dashboard.close)
            dashboard.update_status(audio_enabled=audio is not None, model=predictor.bundle['candidate'])
            print('HTML dashboard: '+dashboard.url, flush=True)

        rclpy.init(args=ros_args)
        resources.callback(lambda: rclpy.shutdown() if rclpy.ok() else None)

        class LiveNode(Node):
            def __init__(self):
                super().__init__('model_a_noise_predictor')
                self.states = StateStream(predictor)
                self.received = {'low': 0}
                self.predictions = 0
                self.audio_cursor = 0
                qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST)
                self.create_subscription(LowState, args.low, self.low, qos)
                self.publisher = self.create_publisher(String, args.output_topic, 10)
                self.level_publisher = self.create_publisher(Float64, args.output_topic+'/dbfs', 10)
                self.create_timer(.010, self.process)
                self.create_timer(2., self.health)
                self.last_error = ''

            def low(self, message):
                received = time.monotonic()
                try:
                    self.states.add(received, low_vector(message))
                    self.received['low'] += 1
                except ValueError as error:
                    self.last_error = str(error)
                    predictor.reset()

            def process(self):
                rows = self.states.drain(time.monotonic())
                for row in rows:
                    emit(row, time.monotonic())
                    row.update(clock='PC time.monotonic receive time', units='dBFS',
                               target='DC-removed 20ms PCM RMS at anchor+20ms',
                               candidate=predictor.bundle['candidate'])
                    if audio:
                        latest = audio.latest()
                        if latest:
                            row.update(latest)
                        row['audio_note'] = 'Latest received 20ms RMS, not aligned to this prediction; SSH delay included'
                    self.publisher.publish(String(data=json.dumps(row, allow_nan=False)))
                    self.level_publisher.publish(Float64(data=row['predicted_noise_dbfs']))
                    self.predictions += 1
                    if dashboard:
                        dashboard.append('predicted', row['target_time'], row['predicted_noise_dbfs'])
                        dashboard.update_status(remaining_horizon_ms=row['remaining_horizon_ms'])
                if dashboard:
                    dashboard.update_status(predictions=self.predictions,
                                            low_received_time=self.states.low[-1][0] if self.states.low else None,
                                            last_error=self.last_error)
                    if audio:
                        dashboard.update_status(audio_alive=audio.thread.is_alive() and audio.proc.poll() is None,
                                                audio_received_time=audio.last_received, audio_error=audio.error)
                        for window in audio.level_windows(self.audio_cursor):
                            dashboard.append('measured', window['time'], window['value'], clipping=window['clipping'])
                            self.audio_cursor = window['end_sample']

            def health(self):
                now = time.monotonic()
                ages = {name: None if not queue else round((now-queue[-1][0])*1000, 1)
                        for name, queue in [('low', self.states.low)]}
                text = f'received={self.received}, predictions={self.predictions}, source_age_ms={ages}, invalid_grid={self.states.invalid_grid_count}, skipped_backlog={self.states.skipped_backlog_count}'
                if self.last_error:
                    text += ', last_error=' + self.last_error
                if audio:
                    latest = audio.latest()
                    text += ', audio=' + json.dumps(latest)
                    if not audio.thread.is_alive() or audio.proc.poll() is not None:
                        text += ', AUDIO STOPPED: check SSH/ALSA device; state predictions continue'
                    if audio.error:
                        text += ', audio_error=' + audio.error
                if any(age is None or age > 20 for age in ages.values()):
                    self.get_logger().warning(text + ' | Waiting for fresh states; check LowState topic/DDS')
                else:
                    self.get_logger().info(text)

        node = LiveNode()
        resources.callback(node.destroy_node)
        print(f'Subscribing: {args.low} (LowState only)\nPublishing: {args.output_topic} (JSON), {args.output_topic}/dbfs (Float64)\nCtrl+C to stop.', flush=True)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        except rclpy.executors.ExternalShutdownException:
            pass


if __name__ == '__main__':
    main()
