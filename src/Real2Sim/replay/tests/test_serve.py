"""Serve only saved results, without analysis, writes or implicit rebuilds."""
from contextlib import contextmanager
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
from threading import Thread
from urllib.error import HTTPError
from urllib.request import build_opener, ProxyHandler

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import Handler, main

OPENER = build_opener(ProxyHandler({}))


def fixture(tmp_path, name='result'):
    root = tmp_path / 'web'
    root.mkdir(exist_ok=True)
    (root / 'index.html').write_text('<html>saved replay</html>')
    (root / 'app.js').write_text('console.log("replay");')
    bundle = tmp_path / name
    (bundle / 'ui').mkdir(parents=True)
    index = dict(dataset='1118_RL_base', trials=[dict(id='trial', file='trial.json')])
    (bundle / 'manifest.json').write_text(json.dumps(dict(index, resultSchema='go2-real2sim-results-v1', status='complete')))
    (bundle / 'ui/index.json').write_text(json.dumps(index))
    (bundle / 'ui/trial.json').write_text('{"saved":true,"lagMs":45}')
    identity = dict(schema='go2-replay-server-v1', dataset='1118_RL_base',
                    resultDirectory=str(bundle.absolute()), webDirectory=str(root.resolve()))
    return root, bundle, identity


@contextmanager
def listener(root, bundle, identity):
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(root),
        ui_directory=bundle / 'ui', identity=identity))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_serves_result_ui_without_raw_data_or_modifying_files(tmp_path):
    root, bundle, identity = fixture(tmp_path)
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob('*') if p.is_file()}
    with listener(root, bundle, identity) as port:
        base = f'http://127.0.0.1:{port}'
        for path in ('/', '/app.js', '/data/index.json', '/data/trial.json'):
            with OPENER.open(base + path) as response:
                assert response.status == 200
        with OPENER.open(base + '/data/trial.json') as response:
            assert json.load(response) == {'saved': True, 'lagMs': 45}
        with pytest.raises(HTTPError) as error:
            OPENER.open(base + '/data/%2e%2e/manifest.json')
        assert error.value.code == 403
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after
    assert not (tmp_path / 'data').exists()


def test_second_launch_reuses_same_result_without_rebuild(tmp_path, capsys):
    root, bundle, identity = fixture(tmp_path)
    with listener(root, bundle, identity) as port:
        main(['--port', str(port), '--result-dir', str(bundle)], root=root)
    assert f'already running: http://127.0.0.1:{port}' in capsys.readouterr().out


def test_different_result_on_same_port_is_not_reused(tmp_path, capsys):
    root, bundle, identity = fixture(tmp_path)
    _, other, other_identity = fixture(tmp_path, 'other')
    with listener(root, other, other_identity) as port:
        with pytest.raises(SystemExit) as error:
            main(['--port', str(port), '--result-dir', str(bundle)], root=root)
        assert error.value.code == 2
    assert '--port <free-port>' in capsys.readouterr().err


@pytest.mark.parametrize('condition', ['missing', 'incomplete', 'wrong_dataset', 'stale_ui'])
def test_bad_result_explains_build_requirement_without_creating_it(tmp_path, capsys, condition):
    root, bundle, _ = fixture(tmp_path)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if condition == 'missing':
        (bundle / 'manifest.json').unlink()
    else:
        if condition == 'incomplete': manifest['status'] = 'building'
        if condition == 'wrong_dataset': manifest['dataset'] = 'another_session'
        if condition == 'stale_ui': manifest['trials'] = []
        (bundle / 'manifest.json').write_text(json.dumps(manifest))
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(SystemExit) as error:
        main(['--result-dir', str(bundle)], root=root)
    assert error.value.code == 2
    assert 'result' in capsys.readouterr().err.lower()
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
