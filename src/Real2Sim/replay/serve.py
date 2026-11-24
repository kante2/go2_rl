#!/usr/bin/env python3
"""Serve a completed analysis result on loopback using only the standard library."""
import argparse
import errno
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import build_opener, ProxyHandler


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, ui_directory, identity, **kwargs):
        self.ui_directory = Path(ui_directory)
        self.identity = identity
        super().__init__(*args, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def _result_request(self, head=False):
        path = urlsplit(self.path).path
        if path == '/__replay__/identity.json':
            payload = json.dumps(self.identity).encode()
        elif path.startswith('/data/'):
            relative = Path(unquote(path[len('/data/'):]))
            destination = (self.ui_directory / relative).resolve()
            if relative.is_absolute() or '..' in relative.parts or not destination.is_relative_to(self.ui_directory.resolve()):
                self.send_error(403)
                return True
            if not destination.is_file() or destination.suffix != '.json':
                self.send_error(404)
                return True
            payload = destination.read_bytes()
        else:
            return False
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        if not head:
            self.wfile.write(payload)
        return True

    def do_GET(self):
        if not self._result_request():
            super().do_GET()

    def do_HEAD(self):
        if not self._result_request(head=True):
            super().do_HEAD()


def existing_replay_matches(identity, port):
    """Reuse only a server connected to the same assets and result directory."""
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(f'http://127.0.0.1:{port}/__replay__/identity.json', timeout=2) as response:
            return response.status == 200 and json.loads(response.read(16384)) == identity
    except (OSError, ValueError):
        return False


def main(argv=None, root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8791)
    parser.add_argument('--dataset', default='1118_RL_base')
    parser.add_argument('--result-dir', type=Path, help='Completed dataset result directory')
    args = parser.parse_args(argv)
    root = Path(root) if root is not None else Path(__file__).resolve().parent / 'web'
    if Path(args.dataset).name != args.dataset or args.dataset in ('', '.', '..'):
        parser.error('dataset must be a single folder name')
    result_dir = (args.result_dir or Path(__file__).resolve().parents[1] / 'result' / args.dataset).absolute()
    try:
        manifest = json.loads((result_dir / 'manifest.json').read_text())
        index = json.loads((result_dir / 'ui/index.json').read_text())
    except (OSError, ValueError) as error:
        parser.error(f'Completed result not found at {result_dir}. '
                     f'Run scripts/real2sim/run_converter.sh --dataset {args.dataset} first. ({error})')
    if (manifest.get('resultSchema') != 'go2-real2sim-results-v1' or manifest.get('status') != 'complete'
            or manifest.get('dataset') != args.dataset or index.get('dataset') != args.dataset
            or manifest.get('trials') != index.get('trials')):
        parser.error('Result manifest and UI dataset do not match or analysis is incomplete; rebuild the result.')
    identity = dict(schema='go2-replay-server-v1', dataset=args.dataset,
                    resultDirectory=str(result_dir), webDirectory=str(root.resolve()))
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(root),
            ui_directory=result_dir / 'ui', identity=identity))
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        if existing_replay_matches(identity, args.port):
            print(f'Replay UI already running: http://127.0.0.1:{args.port}\n'
                  f'Result: {result_dir}\nRefresh the browser to view the saved results.', flush=True)
            return
        parser.error(f'Port {args.port} is already used by another service. '
                     'Try ./run_replay.sh --port <free-port> (for example 8792).')
    print(f'Replay UI: http://127.0.0.1:{server.server_port}', flush=True)
    print(f'Result: {result_dir} (read only; no analysis or rebuild)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
