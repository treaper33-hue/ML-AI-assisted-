"""Project 5 — Serving a model, and watching it afterwards.

The project the other four lead to: a model that leaves the notebook. It
trains two versions, serves one over real HTTP, drives real requests at
it, and then does the things that actually break in production:

    model registry -> local service -> input validation -> training /
    serving skew check -> latency under load (p50, p95, p99) -> input
    drift detection -> output drift -> rollback -> audit pack

Nothing is mocked. The server is a real socket on loopback, the client
is real `urllib`, and the latencies are measured, not estimated.

Chapter 28 (serving, monitoring, rollback), using the data generator
from Project 1.

Usage:
    python p5_serving_monitoring.py
    python p5_serving_monitoring.py --requests 500 --port 8713
"""

import argparse
import dataclasses
import json
import pathlib
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:                 # pragma: no cover - environment gate
    HAS_SKLEARN = False

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from p1_tabular_churn import CATEGORICAL, NUMERIC
    from p1_tabular_churn import build_preprocessor, generate_churn_data
    HAS_P1 = True
except ImportError:                 # pragma: no cover - environment gate
    HAS_P1 = False

RUNS = HERE / 'runs'
REQUIRED_FIELDS = None              # filled in once NUMERIC/CATEGORICAL load


# --------------------------------------------------------------------- #
#                        1. A tiny model registry                        #
# --------------------------------------------------------------------- #


@dataclasses.dataclass
class ModelVersion:
    """One deployable model version.

    Attributes:
        name: Version identifier, e.g. 'v1-logistic'.
        pipeline: The fitted estimator, preprocessing included.
        trained_at: ISO timestamp.
        dev_pr_auc: Score on the development set, for the record.
    """

    name: str
    pipeline: 'Pipeline'
    trained_at: str
    dev_pr_auc: float


class Registry:
    """Holds model versions and which one is live.

    Shipping the whole fitted `Pipeline` -- preprocessing included -- is
    what prevents training/serving skew. Re-implementing the transforms
    in the serving layer is the most common way production predictions
    silently diverge from validated ones.

    Attributes:
        versions: Mapping from name to ModelVersion.
        live: Name of the version currently serving.
    """

    def __init__(self) -> None:
        """Creates an empty registry."""
        self.versions: Dict[str, ModelVersion] = {}
        self.live: Optional[str] = None

    def register(self, version: ModelVersion) -> None:
        """Adds a version without promoting it.

        Args:
            version: The version to add.
        """
        self.versions[version.name] = version

    def promote(self, name: str) -> None:
        """Makes a registered version live.

        Args:
            name: Version to promote.

        Raises:
            KeyError: If the version is not registered.
        """
        if name not in self.versions:
            raise KeyError(f'unknown version {name!r}')
        self.live = name

    def current(self) -> ModelVersion:
        """Returns the live version.

        Raises:
            RuntimeError: If nothing has been promoted.
        """
        if self.live is None:
            raise RuntimeError('no live version')
        return self.versions[self.live]


# --------------------------------------------------------------------- #
#                          2. The service                                #
# --------------------------------------------------------------------- #


class ServiceState:
    """Shared state between the handler and the test harness.

    Attributes:
        registry: The model registry.
        request_count: Requests served.
        error_count: Requests rejected.
        latencies: Per-request service time in milliseconds.
        scores: Every probability returned, for output-drift checks.
    """

    def __init__(self, registry: Registry) -> None:
        """Initialises counters.

        Args:
            registry: The registry to serve from.
        """
        self.registry = registry
        self.request_count = 0
        self.error_count = 0
        self.latencies: List[float] = []
        self.scores: List[float] = []
        self.lock = threading.Lock()


def validate_payload(payload: object) -> Tuple[bool, str]:
    """Checks one request body against the expected schema.

    Rejecting at the boundary is the whole job: a malformed request must
    produce a 400 with a readable reason, never a 500 and never a
    silently imputed prediction.

    Args:
        payload: The decoded JSON body.

    Returns:
        A tuple (ok, message).
    """
    if not isinstance(payload, dict):
        return False, 'body must be a JSON object'
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return False, f'missing fields: {", ".join(sorted(missing))}'
    for field in NUMERIC:
        value = payload[field]
        if value is None:
            continue                      # null is allowed; the imputer
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, f'{field} must be numeric, got {type(value).__name__}'
        if not np.isfinite(float(value)):
            return False, f'{field} must be finite'
    for field in CATEGORICAL:
        if not isinstance(payload[field], str):
            return False, f'{field} must be a string'
    return True, 'ok'


def make_handler(state: ServiceState):
    """Builds a request handler bound to the given state.

    Args:
        state: Shared service state.

    Returns:
        A BaseHTTPRequestHandler subclass.
    """

    class Handler(BaseHTTPRequestHandler):
        """Handles health, metrics and prediction requests."""

        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt: str, *args) -> None:
            """Silences the default stderr access log."""

        def _send(self, code: int, body: Dict) -> None:
            """Writes a JSON response.

            Args:
                code: HTTP status code.
                body: Payload to serialise.
            """
            encoded = json.dumps(body).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:          # noqa: N802 - http.server API
            """Serves /health and /metrics."""
            if self.path == '/health':
                version = state.registry.live
                self._send(200, {'status': 'ok', 'live_version': version})
            elif self.path == '/metrics':
                with state.lock:
                    self._send(200, {
                        'requests': state.request_count,
                        'errors': state.error_count,
                        'live_version': state.registry.live,
                    })
            else:
                self._send(404, {'error': 'not found'})

        def do_POST(self) -> None:         # noqa: N802 - http.server API
            """Serves /predict."""
            started = time.perf_counter()
            if self.path != '/predict':
                self._send(404, {'error': 'not found'})
                return

            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length) if length else b''
            try:
                payload = json.loads(raw.decode('utf-8')) if raw else None
            except json.JSONDecodeError as error:
                with state.lock:
                    state.error_count += 1
                self._send(400, {'error': f'invalid JSON: {error.msg}'})
                return

            ok, message = validate_payload(payload)
            if not ok:
                with state.lock:
                    state.error_count += 1
                self._send(400, {'error': message})
                return

            version = state.registry.current()
            frame = pd.DataFrame([payload])[NUMERIC + CATEGORICAL]
            score = float(version.pipeline.predict_proba(frame)[0, 1])
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            with state.lock:
                state.request_count += 1
                state.latencies.append(elapsed_ms)
                state.scores.append(score)

            self._send(200, {'churn_probability': score,
                             'model_version': version.name,
                             'latency_ms': round(elapsed_ms, 3)})

    return Handler


def start_service(state: ServiceState,
                  port: int) -> Tuple['ThreadingHTTPServer', threading.Thread]:
    """Starts the service on loopback in a background thread.

    Args:
        state: Shared service state.
        port: Port to bind; 0 asks the OS for a free one.

    Returns:
        A tuple (server, thread).
    """
    server = ThreadingHTTPServer(('127.0.0.1', port), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def post_predict(base: str, payload: Dict,
                 timeout: float = 5.0) -> Tuple[int, Dict]:
    """Sends one prediction request.

    Args:
        base: Base URL, e.g. 'http://127.0.0.1:8712'.
        payload: Request body.
        timeout: Socket timeout in seconds.

    Returns:
        A tuple (status_code, decoded_body).
    """
    request = urllib.request.Request(
        f'{base}/predict', method='POST',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def get_json(base: str, path: str, timeout: float = 5.0) -> Tuple[int, Dict]:
    """Performs a GET and decodes the JSON body.

    Args:
        base: Base URL.
        path: Path to request.
        timeout: Socket timeout in seconds.

    Returns:
        A tuple (status_code, decoded_body).
    """
    with urllib.request.urlopen(f'{base}{path}', timeout=timeout) as response:
        return response.status, json.loads(response.read())


# --------------------------------------------------------------------- #
#                           3. Monitoring                                #
# --------------------------------------------------------------------- #


def population_stability_index(reference: np.ndarray, live: np.ndarray,
                               bins: int = 10) -> float:
    """Population stability index between a reference and a live sample.

    PSI sums (p_live - p_ref) * ln(p_live / p_ref) over quantile bins of
    the reference. The conventional readings are: below 0.1 no material
    shift, 0.1 to 0.25 moderate, above 0.25 significant.

    Args:
        reference: Values seen during training.
        live: Values seen in production.
        bins: Number of quantile bins.

    Returns:
        The PSI, always non-negative.
    """
    reference = np.asarray(reference, dtype=np.float64)
    live = np.asarray(live, dtype=np.float64)
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    live_counts, _ = np.histogram(live, bins=edges)
    # A zero bin makes the log infinite, so floor both at a small share.
    ref_share = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    live_share = np.clip(live_counts / max(live_counts.sum(), 1), 1e-6, None)
    return float(np.sum((live_share - ref_share) *
                        np.log(live_share / ref_share)))


def drift_report(reference: 'pd.DataFrame', live: 'pd.DataFrame',
                 threshold: float = 0.25) -> List[Dict]:
    """Computes PSI per numeric feature and flags the shifted ones.

    Args:
        reference: Training-time feature table.
        live: Production feature table.
        threshold: PSI above which a feature counts as drifted.

    Returns:
        One record per numeric feature, worst first.
    """
    rows = []
    for column in NUMERIC:
        ref = reference[column].dropna().to_numpy()
        now = live[column].dropna().to_numpy()
        psi = population_stability_index(ref, now)
        rows.append({'feature': column, 'psi': round(psi, 4),
                     'drifted': bool(psi > threshold)})
    return sorted(rows, key=lambda r: -r['psi'])


def apply_covariate_shift(frame: 'pd.DataFrame',
                          seed: int) -> 'pd.DataFrame':
    """Returns a copy of the data with a deliberate input shift.

    The story: a marketing push brings in newer customers who log in
    less often and raise more tickets. The *relationship* between
    features and churn is unchanged -- only the input distribution moves
    -- which is exactly the case input monitoring is meant to catch and
    output monitoring alone might not.

    Args:
        frame: The original feature table.
        seed: Seed for the perturbation.

    Returns:
        A shifted copy.
    """
    rng = np.random.default_rng(seed)
    shifted = frame.copy()
    shifted['tenure_months'] = shifted['tenure_months'] * 0.45
    shifted['days_since_login'] = (shifted['days_since_login'] +
                                   rng.uniform(8, 20, len(shifted)))
    shifted['support_tickets'] = shifted['support_tickets'] + 2.0
    return shifted


@dataclasses.dataclass
class Gate:
    """One acceptance check declared before the run."""

    name: str
    rule: str
    observed: str
    passed: bool


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs the project end to end.

    Args:
        argv: Optional argument vector.

    Returns:
        0 when every gate passed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260101)
    parser.add_argument('--requests', type=int, default=300)
    parser.add_argument('--port', type=int, default=0,
                        help='0 lets the OS choose a free port')
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args(argv)

    if not HAS_SKLEARN or not HAS_P1:
        print('UNVERIFIED: scikit-learn/pandas or Project 1 is unavailable. '
              'Run testbed/setup.ps1 first.')
        return 2

    global REQUIRED_FIELDS
    REQUIRED_FIELDS = NUMERIC + CATEGORICAL

    started = time.perf_counter()
    print('=' * 74)
    print('PROJECT 5 - Serving a model, and watching it afterwards')
    print('=' * 74)

    # --- 1. train two versions ---------------------------------------- #
    frame, labels, groups = generate_churn_data(seed=args.seed)
    cut = int(len(frame) * 0.8)
    x_dev, y_dev = frame.iloc[:cut], labels[:cut]
    x_live, y_live = frame.iloc[cut:], labels[cut:]

    registry = Registry()
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    for name, estimator in (
            ('v1-logistic', LogisticRegression(max_iter=2000,
                                               class_weight='balanced')),
            ('v2-boosting', HistGradientBoostingClassifier(random_state=0))):
        pipeline = Pipeline([('prep', build_preprocessor()),
                             ('model', estimator)]).fit(x_dev, y_dev)
        score = float(average_precision_score(
            y_live, pipeline.predict_proba(x_live)[:, 1]))
        registry.register(ModelVersion(name, pipeline, stamp, score))
        print(f'\n[1] trained {name}: holdout PR-AUC {score:.4f}')

    registry.promote('v2-boosting')
    print(f'    promoted {registry.live} to live')

    # --- 2. start the service ------------------------------------------ #
    state = ServiceState(registry)
    server, _ = start_service(state, args.port)
    base = f'http://127.0.0.1:{server.server_address[1]}'
    print(f'\n[2] service listening on {base}')

    try:
        code, health = get_json(base, '/health')
        print(f'    GET /health -> {code} {health}')

        # --- 3. input validation --------------------------------------- #
        print(f'\n[3] Input validation (a bad request must 400, not 500)')
        good = x_live.iloc[0].to_dict()
        good = {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                for k, v in good.items()}

        bad_cases = {
            'missing field': {k: v for k, v in good.items()
                              if k != 'tenure_months'},
            'wrong type': {**good, 'monthly_charge': 'free'},
            'infinite value': {**good, 'days_since_login': float('inf')},
            'not an object': [1, 2, 3],
        }
        validation_rows = []
        for label, payload in bad_cases.items():
            status, body = post_predict(base, payload)
            validation_rows.append({'case': label, 'status': status,
                                    'error': body.get('error', '')})
            print(f'    {label:<16} -> {status} {body.get("error", "")[:48]}')
        all_rejected = all(r['status'] == 400 for r in validation_rows)

        status, body = post_predict(base, good)
        print(f'    {"valid request":<16} -> {status} '
              f'p={body["churn_probability"]:.4f} '
              f'({body["model_version"]})')

        # --- 4. training / serving skew -------------------------------- #
        print(f'\n[4] Training/serving skew')
        sample = x_live.iloc[:50]
        direct = registry.current().pipeline.predict_proba(sample)[:, 1]
        served = []
        for _, row in sample.iterrows():
            payload = {k: (None if (isinstance(v, float) and np.isnan(v))
                           else v) for k, v in row.to_dict().items()}
            _, response = post_predict(base, payload)
            served.append(response['churn_probability'])
        skew = float(np.abs(np.array(served) - direct).max())
        print(f'    max |served - in-process| over 50 rows: {skew:.2e}')
        print('    the fitted Pipeline is shipped whole, so there is no '
              'second\n    implementation of the preprocessing to drift '
              'from the first')

        # --- 5. latency under load ------------------------------------- #
        print(f'\n[5] Latency over {args.requests} requests')
        payloads = []
        for _, row in x_live.sample(
                n=min(args.requests, len(x_live)), replace=True,
                random_state=args.seed).iterrows():
            payloads.append({k: (None if (isinstance(v, float)
                                          and np.isnan(v)) else v)
                             for k, v in row.to_dict().items()})
        wall_start = time.perf_counter()
        for payload in payloads:
            post_predict(base, payload)
        wall = time.perf_counter() - wall_start

        with state.lock:
            measured = sorted(state.latencies[-len(payloads):])
        p50 = statistics.median(measured)
        p95 = measured[int(0.95 * len(measured)) - 1]
        p99 = measured[int(0.99 * len(measured)) - 1]
        throughput = len(payloads) / wall
        print(f'    server-side p50 {p50:.2f} ms   p95 {p95:.2f} ms   '
              f'p99 {p99:.2f} ms')
        print(f'    end-to-end throughput {throughput:.0f} req/s '
              f'(single process, single client)')
        print('    p95 is the number users feel; the mean would have hidden '
              'the tail')

        # --- 6. input drift --------------------------------------------- #
        print(f'\n[6] Input drift (PSI, quantile bins of the training data)')
        quiet = drift_report(x_dev, x_live)
        shifted_frame = apply_covariate_shift(x_live, args.seed)
        loud = drift_report(x_dev, shifted_frame)
        print(f'    {"feature":<22}{"in-distribution":>18}{"after shift":>14}')
        for calm, noisy in zip(quiet, sorted(loud,
                                             key=lambda r: r['feature'])):
            match = next(r for r in loud if r['feature'] == calm['feature'])
            print(f'    {calm["feature"]:<22}{calm["psi"]:>18.4f}'
                  f'{match["psi"]:>14.4f}'
                  f'{"  DRIFT" if match["drifted"] else ""}')
        quiet_flags = sum(r['drifted'] for r in quiet)
        loud_flags = sum(r['drifted'] for r in loud)
        print(f'    flagged: {quiet_flags} in-distribution, '
              f'{loud_flags} after the shift')

        # --- 7. output drift -------------------------------------------- #
        shifted_scores = registry.current().pipeline.predict_proba(
            shifted_frame)[:, 1]
        baseline_scores = registry.current().pipeline.predict_proba(
            x_live)[:, 1]
        output_psi = population_stability_index(baseline_scores,
                                                shifted_scores)
        print(f'\n[7] Output drift')
        print(f'    mean score {baseline_scores.mean():.4f} -> '
              f'{shifted_scores.mean():.4f}, PSI {output_psi:.4f}')
        print('    inputs moved and the score distribution moved with them; '
              'labels\n    arrive later, so this is the earliest warning '
              'available')

        # --- 8. rollback -------------------------------------------------- #
        print(f'\n[8] Rollback')
        before = post_predict(base, good)[1]
        registry.promote('v1-logistic')
        after = post_predict(base, good)[1]
        expected = float(registry.versions['v1-logistic']
                         .pipeline.predict_proba(
                             pd.DataFrame([good])[NUMERIC + CATEGORICAL]
                         )[0, 1])
        rollback_exact = abs(after['churn_probability'] - expected) < 1e-12
        print(f'    before: {before["model_version"]} '
              f'p={before["churn_probability"]:.4f}')
        print(f'    after : {after["model_version"]} '
              f'p={after["churn_probability"]:.4f}')
        print(f'    matches v1 computed directly: {rollback_exact}')
        registry.promote('v2-boosting')

        _, metrics = get_json(base, '/metrics')
        print(f'    /metrics -> {metrics}')

    finally:
        server.shutdown()
        server.server_close()
        print('\n    service stopped cleanly')

    # --- 9. gates ---------------------------------------------------------- #
    gates = [
        Gate('health_endpoint', 'GET /health returns 200',
             str(health.get('status')), health.get('status') == 'ok'),
        Gate('rejects_bad_input',
             'every malformed request returns 400, never 500',
             f'{sum(1 for r in validation_rows if r["status"] == 400)}'
             f'/{len(validation_rows)}', all_rejected),
        Gate('no_training_serving_skew',
             'served predictions equal in-process predictions',
             f'{skew:.2e}', skew < 1e-12),
        Gate('latency_p95',
             'server-side p95 under 50 ms for a single row',
             f'{p95:.2f} ms', p95 < 50.0),
        Gate('no_false_drift_alarm',
             'no feature flagged on in-distribution traffic',
             f'{quiet_flags} flagged', quiet_flags == 0),
        Gate('detects_real_drift',
             'at least two features flagged after the shift',
             f'{loud_flags} flagged', loud_flags >= 2),
        Gate('output_drift_visible',
             'score distribution PSI above 0.25 after the shift',
             f'{output_psi:.4f}', output_psi > 0.25),
        Gate('rollback_exact',
             'rolled-back service reproduces v1 exactly',
             str(rollback_exact), rollback_exact),
    ]
    print(f'\n[9] Gates')
    for gate in gates:
        mark = 'PASS' if gate.passed else 'FAIL'
        print(f'    [{mark}] {gate.name}: {gate.observed}  '
              f'(rule: {gate.rule})')

    elapsed = time.perf_counter() - started
    status = 'VERIFIED' if all(g.passed for g in gates) else 'PENDING REVISION'
    print(f'\nstatus: {status}   ({elapsed:.1f}s)')

    if not args.no_write:
        run_dir = RUNS / 'p5_serving_monitoring'
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'project': 'p5_serving_monitoring',
            'status': status,
            'seed': args.seed,
            'versions': {name: {'dev_pr_auc': v.dev_pr_auc,
                                'trained_at': v.trained_at}
                         for name, v in registry.versions.items()},
            'live_version': registry.live,
            'validation': validation_rows,
            'skew_max_abs': skew,
            'latency_ms': {'p50': p50, 'p95': p95, 'p99': p99,
                           'requests': len(payloads),
                           'throughput_per_s': throughput},
            'input_drift': {'in_distribution': quiet, 'after_shift': loud},
            'output_drift_psi': output_psi,
            'rollback_exact': rollback_exact,
            'gates': [dataclasses.asdict(g) for g in gates],
            'seconds': round(elapsed, 1),
        }
        (run_dir / 'report.json').write_text(
            json.dumps(payload, indent=2, default=str), encoding='utf-8')
        print(f'audit pack: {run_dir}')

    return 0 if all(g.passed for g in gates) else 1


if __name__ == '__main__':
    sys.exit(main())
