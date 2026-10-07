"""Exercise real TCP restart/occupancy without touching the robot or HTTP 8080."""
import errno
import socket

import pytest

from go2_autonomous_driving.real_state_viewer import check_viewer_port


def test_viewer_restart_after_server_closes_a_connection():
    with socket.socket() as server, socket.socket() as client:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', 0))
        port = server.getsockname()[1]
        server.listen(1)
        client.settimeout(1)
        client.connect(('127.0.0.1', port))
        connection, _ = server.accept()
        connection.close()  # Server actively closes; its port can remain in TIME_WAIT.
        assert client.recv(1) == b''
    check_viewer_port(port)


def test_an_actual_listener_is_still_rejected():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', 0))
        server.listen(1)
        with pytest.raises(OSError) as error:
            check_viewer_port(server.getsockname()[1])
        assert error.value.errno == errno.EADDRINUSE
