#!/usr/bin/env python3
"""Check mode, or explicitly restore Unitree's normal controller. Never publish LowCmd."""
import argparse
import json
import sys
import time

import rclpy
from unitree_api.msg import Request, Response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('go2_normal_recovery')
    replies = {}
    pub = node.create_publisher(Request, '/api/motion_switcher/request', 10)
    node.create_subscription(Response, '/api/motion_switcher/response',
                             lambda msg: replies.update({msg.header.identity.id: msg}), 10)

    def call(api_id, parameter=None):
        deadline = time.monotonic() + 8
        while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
        req = Request()
        req.header.identity.id = time.time_ns()
        req.header.identity.api_id = api_id
        req.parameter = json.dumps(parameter or {})
        pub.publish(req)
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            if req.header.identity.id in replies:
                response = replies.pop(req.header.identity.id)
                if response.header.status.code != 0:
                    raise RuntimeError(f'API {api_id}: code {response.header.status.code}, {response.data}')
                return json.loads(response.data or '{}')
        raise RuntimeError(f'API {api_id}: timed out; inspect state before retrying')

    try:
        mode = call(1001)
        print('Current mode:', mode, flush=True)
        if not isinstance(mode, dict) or not isinstance(mode.get('name'), str):
            raise RuntimeError('Invalid CheckMode response')
        if not args.restore or mode['name'] == 'normal':
            return
        if mode['name']:
            raise RuntimeError('Another named controller is active; no changes made')
        if not sys.stdin.isatty():
            raise RuntimeError('--restore requires an interactive terminal')
        print('Stop the learned controller and support the robot. Restoring normal may change its posture.', flush=True)
        if input('Type NORMAL to restore Unitree control: ').strip() != 'NORMAL':
            print('Cancelled; no mode change sent.')
            return
        if any(info.node_name.startswith('go2_learned_robot_')
               for info in node.get_publishers_info_by_topic('/lowcmd')):
            raise RuntimeError('Learned LowCmd publisher still exists; stop terminal 2 first')
        call(1002, {'name': 'normal'})
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            mode = call(1001)
            if mode.get('name') == 'normal':
                print('NORMAL RESTORED. Use the Unitree app/remote to stand up if needed.', flush=True)
                return
            time.sleep(.2)
        raise RuntimeError(f'Restore request sent but normal not confirmed: {mode}')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        raise SystemExit(str(error))
