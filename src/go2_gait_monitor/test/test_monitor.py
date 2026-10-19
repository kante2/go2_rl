import csv
from http.server import ThreadingHTTPServer
from functools import partial
import io
import json
from pathlib import Path
import sys
import threading
from urllib.request import urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_gait_monitor.metrics import FootMetrics, FEET
from go2_gait_monitor.server import TelemetryStore, validate_packet, DashboardHandler


def packet(sequence=0, session="sim", reset=0):
    return dict(version=1, source="mujoco", session=session, sequence=sequence,
                reset_counter=reset, sim_time=sequence*.05, physics_dt=.005,
                model="quietpaw_long", checkpoint="model_29999.pt", quietness=.2,
                feet=list(FEET), command=[.5,0,0], linear_velocity=[.4,0,0],
                angular_velocity=[0,0,.1], **FootMetrics().snapshot())


def test_integrates_physics_samples_and_keeps_short_peak():
    metrics = FootMetrics()
    for _ in range(4):
        metrics.sample([0]*4, .005)
    metrics.sample([200, 100, 0, 0], .005)
    for _ in range(9):
        metrics.sample([100, 100, 0, 0], .005)
    data = metrics.snapshot()
    assert data["landing_impulse_ns"] == [pytest.approx(5.5), pytest.approx(5), None, None]
    assert data["landing_peak_n"] == [200, 100, None, None]
    assert data["landing_count"] == [1, 1, 0, 0]
    assert data["peak_n"] == [200, 100, 0, 0]
    metrics.clear_interval_peak()
    assert metrics.snapshot()["peak_n"] == [0]*4
    assert metrics.snapshot()["landing_peak_n"][0] == 200


def test_standing_and_contact_chatter_are_not_landings():
    metrics = FootMetrics()
    for _ in range(40):
        metrics.sample([40]*4, .005)
    metrics.sample([0]*4, .005)
    metrics.sample([40]*4, .005)
    assert metrics.snapshot()["landing_count"] == [0]*4
    assert metrics.snapshot()["landing_impulse_ns"] == [None]*4


def test_short_contact_and_noninteger_window():
    metrics = FootMetrics()
    metrics.sample([0]*4, .03)
    metrics.sample([-100]*4, .03)
    metrics.sample([-100]*4, .03)
    assert metrics.snapshot()["landing_impulse_ns"] == [pytest.approx(5)]*4
    metrics.sample([0]*4, .03)
    metrics.sample([100]*4, .01)
    metrics.sample([0]*4, .01)
    assert metrics.snapshot()["landing_impulse_ns"] == [pytest.approx(1)]*4


def test_reset_new_session_sequence_and_missing_packets():
    store = TelemetryStore()
    assert store.accept(packet(), now=1)
    assert not store.accept(packet(), now=1.05)
    assert store.accept(packet(3), now=1.15)
    assert store.dropped == 2
    assert not store.accept(packet(0, session="other"), now=1.2)
    assert store.accept(packet(4, reset=1), now=1.3)
    assert len(store.history) == 1 and store.dropped == 0
    assert store.accept(packet(0, session="other"), now=3)
    assert len(store.history) == 1 and store.latest["session"] == "other"


@pytest.mark.parametrize("field,value", [
    ("force_n", [float("nan")]*4), ("peak_n", [-1]*4),
    ("feet", list(reversed(FEET))), ("command", [0,0]),
    ("physics_dt", 0), ("landing_count", [True]*4),
    ("source", "robot"), ("sim_time", float("inf")),
])
def test_invalid_packet_rejected(field,value):
    data=packet(); data[field]=value
    with pytest.raises(ValueError):
        validate_packet(data)


def test_http_live_state_csv_and_assets():
    store=TelemetryStore()
    store.accept(packet())
    server=ThreadingHTTPServer(("127.0.0.1",0),partial(DashboardHandler,store=store))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base=f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base+"/api/state") as response:
            data=json.load(response)
        assert data["connected"] and data["latest"]["model"]=="quietpaw_long"
        with urlopen(base+"/api/state?after=1") as response:
            assert json.load(response)["samples"]==[]
        with urlopen(base+"/api/export.csv") as response:
            rows=list(csv.DictReader(io.StringIO(response.read().decode("utf-8-sig"))))
        assert rows[0]["FR_landing_impulse_ns"]==""  # No invented zero measurement.
        assert rows[0]["vx_mps"]=="0.4"
        for name in ("/", "/app.js", "/style.css"):
            with urlopen(base+name) as response:
                assert response.status==200 and len(response.read())>100
    finally:
        server.shutdown(); server.server_close(); thread.join()
