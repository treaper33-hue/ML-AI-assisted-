"""Contract tests for the control panel's server.

The server is a viewer. Two things it must get right:

  * the address it prints has to be an address a reader can use, and
  * it must answer reads and refuse everything else, because a viewer
    that can accept a POST is one change away from being able to edit a
    threshold after seeing a result [VERIFICATION.md §2.6].
"""

from panel import serve


class _FakeServer:
    """Stands in for a bound server with a chosen address."""

    def __init__(self, address):
        self.server_address = address


def test_bound_address_reports_a_real_bound_port(tmp_path):
    """Port 0 asks the OS for a port; we must report the one we got."""
    server = serve.build_server('127.0.0.1', 0, tmp_path, True)
    try:
        host, port = serve.bound_address(server)
    finally:
        server.server_close()
    assert host == '127.0.0.1'
    assert isinstance(port, int)
    assert port > 0


def test_bound_address_decodes_a_bytes_host():
    """A bytes host would print as "b'127.0.0.1'" in the URL.

    `server_address` is typed loosely enough to carry bytes, and the one
    moment a reader needs to trust the printed URL is the moment it
    appears. This is the regression the type checker caught.
    """
    host, port = serve.bound_address(_FakeServer((b'127.0.0.1', 8765)))
    assert host == '127.0.0.1'
    assert port == 8765
    assert f'http://{host}:{port}/' == 'http://127.0.0.1:8765/'


def test_bound_address_always_returns_text_and_a_number():
    """Whatever the socket layer hands back, the URL is printable."""
    for address in (('localhost', 80), (b'::1', 8080), ('0.0.0.0', 1)):
        host, port = serve.bound_address(_FakeServer(address))
        assert isinstance(host, str)
        assert isinstance(port, int)
        assert "b'" not in f'http://{host}:{port}/'


def test_check_runs_reports_a_missing_directory_without_crashing(tmp_path):
    """`--check` is the non-browser path; it must survive an empty root."""
    code = serve.check_runs(tmp_path / 'does-not-exist')
    assert isinstance(code, int)
