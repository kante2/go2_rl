"""Blocked console I/O and timing attribution without ROS or motor output."""
import threading
import time

import pytest

from go2_autonomous_driving.real.runtime_monitor import AsyncStatusPrinter, GCPauseMonitor, MotorLoopTiming, PublishTiming


def test_blocked_terminal_does_not_block_or_grow_status_queue():
    entered, release = threading.Event(), threading.Event()

    def blocked(line):
        entered.set()
        release.wait(timeout=2)

    reporter = AsyncStatusPrinter(sink=blocked)
    try:
        reporter.submit('LIVE')
        assert entered.wait(timeout=1)
        before = time.monotonic()
        for _ in range(1000):
            reporter.submit('next status')
        assert time.monotonic() - before < .1
        assert reporter.pending.qsize() == 8
        assert reporter.dropped == 992
        reporter.close()  # Must return even while the writer is still blocked.
        assert reporter.thread.is_alive()
    finally:
        release.set()
        reporter.close()
    assert not reporter.thread.is_alive()


def test_console_failure_does_not_escape_to_motor_loop():
    attempted = threading.Event()

    def broken(line):
        attempted.set()
        raise BrokenPipeError('terminal disconnected')

    reporter = AsyncStatusPrinter(sink=broken)
    try:
        reporter.submit('LIVE')
        assert attempted.wait(timeout=1)
    finally:
        reporter.close()
    assert reporter.error == 'terminal disconnected'


def test_timing_separates_wakeup_delay_from_publish_delay(monkeypatch):
    # Woke 4 ms late, then spent 11 ms inside publish. Original deadline was 1 s.
    times = iter([1.0045, 1.0050, 1.0160, 1.0162, 1.0162])
    monkeypatch.setattr('go2_autonomous_driving.real.runtime_monitor.time.monotonic',
                        lambda: next(times))
    timer = MotorLoopTiming(1.004, 1.000)
    timer.mark('state_check')
    timer.mark('tracking_check')
    timer.mark('publish')
    timer.mark('status_enqueue')
    finished, frame = timer.finish(1.002)
    assert frame['wake_lateness_ms'] == pytest.approx(4.)
    assert frame['publish_ms'] == pytest.approx(11.)
    assert frame['status_enqueue_ms'] == pytest.approx(.2)
    assert frame['loop_ms'] == pytest.approx(12.2)
    assert frame['deadline_lateness_ms'] == pytest.approx(14.2)
    assert 1.002 < finished - .01  # Existing 10 ms guard remains exceeded.


def test_publish_timing_separates_allocation_crc_and_dds_wait(monkeypatch):
    wall = iter([1., 1.001, 1.0012, 1.0672, 1.0673])
    cpu = iter([0., .0005, .0007, .0017, .0018])
    monkeypatch.setattr('go2_autonomous_driving.real.runtime_monitor.time.monotonic', lambda: next(wall))
    monkeypatch.setattr('go2_autonomous_driving.real.runtime_monitor.time.thread_time', lambda: next(cpu))
    timer = PublishTiming()
    timer.mark('prepare')
    timer.mark('crc_pack')
    timer.mark('dds')
    values = timer.finish()
    assert values['prepare_ms'] == pytest.approx(1.)
    assert values['crc_pack_ms'] == pytest.approx(.2)
    assert values['dds_ms'] == pytest.approx(66.)
    assert values['dds_cpu_ms'] == pytest.approx(1.)
    assert values['total_ms'] == pytest.approx(67.3)


def test_gc_monitor_records_bounded_pauses_and_unregisters():
    import gc
    monitor = GCPauseMonitor()
    try:
        assert monitor.callback in gc.callbacks
        for _ in range(40):
            monitor.callback('start', {'generation': 2})
            monitor.callback('stop', {'generation': 2, 'collected': 3})
        events = monitor.snapshot()
        assert len(events) == 32
        assert events[-1]['generation'] == 2 and events[-1]['collected'] == 3
        assert events[-1]['duration_ms'] >= 0
    finally:
        monitor.close()
    assert monitor.callback not in gc.callbacks
