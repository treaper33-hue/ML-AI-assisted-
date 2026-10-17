"""Serves the control panel from the standard library alone.

Run it from the testbed directory:

    python -m panel.serve
    python -m panel.serve --port 9000 --no-browser
    python -m panel.serve --runs runs --check

There is no build step and no database. `runs/` is read on every
request, so a run finished a second ago appears on the next refresh.

The server answers GET and HEAD and nothing else. That is not a
simplification: the panel is a viewer, and a viewer that can accept a
POST is one change away from being able to edit a threshold after
seeing a result [VERIFICATION.md §2.6].
"""

import argparse
import functools
import http.server
import json
import pathlib
import sys
import webbrowser
from typing import Any, Dict, Optional, Self, Sequence, Tuple

from panel import reports

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8765
API_PATH = '/api/runs.json'
STATIC_DIR = pathlib.Path(__file__).resolve().parent / 'static'


def _testbed_root() -> pathlib.Path:
    """Returns the testbed directory this package lives in.

    Returns:
        The parent of the `panel` package, which is where `runs/` and
        `configs/` sit.
    """
    return pathlib.Path(__file__).resolve().parent.parent


def encode_payload(runs_root: pathlib.Path) -> bytes:
    """Builds the API payload as UTF-8 JSON the browser can parse.

    `allow_nan=False` is the point of this function. A diverged run
    writes `NaN` into `report.json`, which Python's encoder emits
    happily and the browser's `JSON.parse` rejects outright. Any
    non-finite value that survived `reports.metric_cell` is therefore a
    defect in this panel, and this is where it surfaces instead of
    arriving as an unexplained blank dashboard.

    Args:
        runs_root: The `runs/` directory to read.

    Returns:
        The encoded payload.

    Raises:
        ValueError: If a non-finite number reached the payload.
    """
    payload = reports.build_payload(runs_root)
    return json.dumps(payload, allow_nan=False).encode('utf-8')


# `self` carries a `typing.Self` annotation in every method below:
# the code contract asks for an annotation on every parameter
# [GPSG p.46 §3.19], and this file is linted like any other.
class PanelHandler(http.server.SimpleHTTPRequestHandler):
    """Serves `panel/static/` plus one generated JSON endpoint.

    Attributes:
        runs_root: The `runs/` directory, read fresh per request.
        quiet: Whether to suppress the per-request log line.
    """

    runs_root: pathlib.Path = pathlib.Path('runs')
    quiet: bool = False

    def __init__(
        self: Self,
        *args: Any,
        runs_root: pathlib.Path,
        quiet: bool = False,
        **kwargs: Any,
    ) -> None:
        """Binds the handler to a runs directory.

        Args:
            *args: Forwarded to the base handler.
            runs_root: The `runs/` directory to read.
            quiet: Whether to suppress per-request logging.
            **kwargs: Forwarded to the base handler.
        """
        self.runs_root = runs_root
        self.quiet = quiet
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self: Self) -> None:  # noqa: N802  (http.server's spelling)
        """Answers a GET for the API endpoint or a static file."""
        if self.path.split('?')[0] == API_PATH:
            self._send_payload()
            return
        super().do_GET()

    def _send_payload(self: Self) -> None:
        """Builds and sends the run payload, or explains why it could not."""
        try:
            body = encode_payload(self.runs_root)
            status = 200
        except (OSError, ValueError) as error:
            body = json.dumps(
                {
                    'runs': [],
                    'unverified': [],
                    'sweeps': [],
                    'statuses': list(reports.KNOWN_STATUSES),
                    'notices': [f'panel error: {error}'],
                }
            ).encode('utf-8')
            status = 500
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        # The panel reflects what is on disk now, so nothing is cached.
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self: Self) -> None:
        """Adds the no-store header to every static response too."""
        if self.path.split('?')[0] != API_PATH:
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(
        self: Self,
        format: str,
        *args: Any,
    ) -> None:
        """Logs one line per request unless running quietly.

        Args:
            format: Printf-style format from the base class.
            *args: Values for the format.
        """
        if self.quiet:
            return
        sys.stderr.write(f'  {self.address_string()} {format % args}\n')


def check_runs(runs_root: pathlib.Path) -> int:
    """Parses `runs/` and prints what the panel would show.

    This is the headless form of the dashboard, for a terminal or CI
    where no browser is going to open.

    Args:
        runs_root: The `runs/` directory to read.

    Returns:
        0 when every run parsed and every run is VERIFIED, 1 otherwise,
        so a check can gate on it.
    """
    payload = reports.build_payload(runs_root)
    runs = payload['runs']
    print(f'runs root: {payload["runs_root"]}')
    print(f'runs found: {len(runs)}')
    for notice in payload['notices']:
        print(f'  notice: {notice}')
    for run in runs:
        tag = run['status'] or payload['unreadable_label']
        print(f'  {run["run_id"]:<34}{tag:<18}{run["experiment"]}')
        for problem in run['errors']:
            print(f'      - {problem}')
    for group in payload['sweeps']:
        print(
            f'  sweep {group["key"]} ({len(group["points"])} points, '
            f'{group["diverged"]} diverged)'
        )
    if payload['unverified']:
        print(
            f'\nNOT VERIFIED: {len(payload["unverified"])} run(s) -- '
            f'{", ".join(payload["unverified"])}'
        )
        return 1
    return 0


def build_server(
    host: str,
    port: int,
    runs_root: pathlib.Path,
    quiet: bool = False,
) -> http.server.ThreadingHTTPServer:
    """Builds the HTTP server without starting it.

    Args:
        host: Interface to bind. The default is loopback: the panel
            shows an unpublished research log and has no auth.
        port: Port to bind; 0 asks the OS for a free one.
        runs_root: The `runs/` directory to read.
        quiet: Whether to suppress per-request logging.

    Returns:
        A server ready for `serve_forever`.

    Raises:
        FileNotFoundError: If the static assets are missing.
    """
    if not (STATIC_DIR / 'index.html').is_file():
        raise FileNotFoundError(f'panel assets missing from {STATIC_DIR}')
    handler = functools.partial(PanelHandler, runs_root=runs_root, quiet=quiet)
    return http.server.ThreadingHTTPServer((host, port), handler)


def _parse_args(
    argv: Optional[Sequence[str]],
) -> Tuple[argparse.Namespace, pathlib.Path]:
    """Parses the command line and resolves the runs directory.

    Args:
        argv: Optional argument vector, for testing.

    Returns:
        A tuple of (parsed arguments, resolved runs directory).
    """
    parser = argparse.ArgumentParser(
        prog='python -m panel.serve',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--host',
        default=DEFAULT_HOST,
        help=f'interface to bind (default {DEFAULT_HOST})',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=DEFAULT_PORT,
        help=f'port to bind (default {DEFAULT_PORT})',
    )
    parser.add_argument(
        '--runs',
        type=pathlib.Path,
        default=None,
        help='run directory to read (default <testbed>/runs)',
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='do not open a browser window',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='do not log one line per request',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='print what the panel would show and exit',
    )
    args = parser.parse_args(argv)
    runs_root = args.runs or (_testbed_root() / reports.RUNS_DIRNAME)
    return args, runs_root


def bound_address(
    server: http.server.ThreadingHTTPServer,
) -> Tuple[str, int]:
    """Returns the address the server actually bound, as text.

    `server_address` is typed loosely enough to carry bytes, and a bytes
    host formats as "b'127.0.0.1'" -- a URL nobody can click, printed at
    the one moment the reader needs to trust it.

    Args:
        server: The bound server.

    Returns:
        A (host, port) pair with the host as text.
    """
    host, port = server.server_address[:2]
    if isinstance(host, (bytes, bytearray)):
        host = host.decode('ascii', 'replace')
    return str(host), int(port)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the control panel server.

    Args:
        argv: Optional argument vector, for testing.

    Returns:
        0 on a clean exit; `check_runs`' own code under `--check`.
    """
    args, runs_root = _parse_args(argv)
    if args.check:
        return check_runs(runs_root)

    server = build_server(args.host, args.port, runs_root, args.quiet)
    host, port = bound_address(server)
    url = f'http://{host}:{port}/'
    payload: Dict[str, Any] = reports.build_payload(runs_root)
    print(f'ML mastery control panel -- {url}')
    print(f'reading {runs_root} ({len(payload["runs"])} run(s) now)')
    if payload['unverified']:
        print(
            f'{len(payload["unverified"])} run(s) NOT VERIFIED; the panel '
            'banners them rather than hiding them'
        )
    print('read-only: GET and HEAD only. Ctrl-C to stop.')
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
