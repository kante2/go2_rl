#!/usr/bin/env python3
"""Model-A noise prediction from MuJoCo simulation states, shown on the v2 dashboard.

The simulation (simulation.py --state-port) sends a LowState-like 35-vector per physics step
over local UDP. Features are computed on the packet's simulation clock: physics steps arrive in
bursts, and the simulation may run slower than real time, so arrival time would distort the
force/velocity change rates. Only the dashboard plot is shifted onto this PC's clock.
No microphone: only the predicted curve is shown. Units are raw PCM dBFS, not SPL.
Recording is separate: src/db_collector/record_sim.py (both cannot share one UDP port).
"""
import argparse
import json
import socket
import time

from stream_model import Predictor, StateStream


class SimClock:
    """Track one simulation run; a new session or a dropped packet restarts the history."""
    def __init__(self):
        self.session = self.tick = None

    def restart(self, session, tick):
        restart = session != self.session or self.tick is None or tick != self.tick + 1
        self.session, self.tick = session, tick
        return restart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Defaults to the saved model-A selection')
    parser.add_argument('--state-port', type=int, default=9881, help='simulation.py --state-port')
    parser.add_argument('--web-port', type=int, default=8081)
    parser.add_argument('--web-host', default='127.0.0.1')
    args = parser.parse_args()

    from web_dashboard_v2 import Dashboard
    predictor = Predictor(args.model)
    states = StateStream(predictor)
    clock = SimClock()
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', args.state_port))
    receiver.settimeout(.05)
    dashboard = Dashboard(args.web_host, args.web_port)
    dashboard.update_status(audio_enabled=False, model=predictor.bundle['candidate'] + ' · MuJoCo sim states')
    print(f'Model: {predictor.path}\nListening for simulation states on UDP {args.state_port}\n'
          f'HTML dashboard: {dashboard.url}\nOutput: predicted raw PCM dBFS, not calibrated SPL. Ctrl+C to stop.',
          flush=True)
    predictions, last_print, last_error = 0, 0., ''
    try:
        while True:
            try:
                payload, _ = receiver.recvfrom(65536)
            except socket.timeout:
                continue
            received = time.monotonic()
            try:
                packet = json.loads(payload)
                sim_time = float(packet['sim_time'])
                if clock.restart(packet['session'], packet['tick']):
                    # New run or dropped packet: start a fresh causal history.
                    states = StateStream(predictor)
                    predictor.reset()
                states.add(sim_time, packet['low'])
            except (ValueError, KeyError, TypeError) as error:
                last_error = str(error)
                predictor.reset()
                continue
            for row in states.drain():
                # Plot at arrival time of the newest state, keeping the 20 ms forecast offset.
                plotted = received + row['target_time'] - sim_time
                dashboard.append('predicted', plotted, row['predicted_noise_dbfs'])
                dashboard.update_status(remaining_horizon_ms=(row['target_time'] - sim_time) * 1000)
                predictions += 1
                if received - last_print >= 1:
                    print(f"prediction={row['predicted_noise_dbfs']:.2f} dBFS", flush=True)
                    last_print = received
            dashboard.update_status(predictions=predictions, low_received_time=received, last_error=last_error)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.close()
        receiver.close()


if __name__ == '__main__':
    main()
