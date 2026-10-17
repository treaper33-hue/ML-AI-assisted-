"""Experiment configuration: the control panel.

Experiments read values from a config file and never from edited source.
Configs are TOML so that `tomllib` from the standard library can read them --
one fewer dependency between an outside researcher and a working testbed.

Controls supported by every experiment:
    --config PATH            choose the experiment definition
    --set a.b=value          override one value, recorded in the report
    --sweep a.b=v1,v2,v3     run the same arm across a parameter range
    --compare                score every arm on identical data
    --show-source ARM        print an arm's implementation and its anchors
"""

import argparse
import copy
import pathlib
import tomllib
from typing import Any, Dict, List, Sequence, Tuple


def load_config(path: pathlib.Path) -> Dict[str, Any]:
    """Loads a TOML experiment configuration.

    Args:
        path: Path to the config file.

    Returns:
        The parsed configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f'config not found: {path}')
    with path.open('rb') as handle:
        return tomllib.load(handle)


def _coerce(text: str) -> Any:
    """Converts a command-line string to the narrowest matching type.

    Args:
        text: Raw value from the command line.

    Returns:
        A bool, int, float or the original string.
    """
    lowered = text.strip().lower()
    if lowered in ('true', 'false'):
        return lowered == 'true'
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    return text


def get_path(config: Dict[str, Any], dotted: str) -> Any:
    """Reads a dotted path out of a nested configuration.

    Args:
        config: The configuration.
        dotted: A path such as 'arms.gradient_descent.lr'.

    Returns:
        The value at that path.

    Raises:
        KeyError: If the path does not exist.
    """
    node: Any = config
    for part in dotted.split('.'):
        node = node[part]
    return node


def set_path(config: Dict[str, Any], dotted: str, value: Any) -> None:
    """Writes a value at a dotted path, creating tables as needed.

    Args:
        config: The configuration, modified in place.
        dotted: A path such as 'arms.gradient_descent.lr'.
        value: The value to store.
    """
    parts = dotted.split('.')
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def apply_overrides(
    config: Dict[str, Any],
    overrides: Sequence[str],
) -> Dict[str, Any]:
    """Applies `--set key=value` overrides to a copy of the config.

    Args:
        config: The base configuration.
        overrides: Strings of the form 'a.b=value'.

    Returns:
        A new configuration with the overrides applied and recorded under
        the 'overrides' key, so the run report shows what was changed.

    Raises:
        ValueError: If an override is not of the form 'key=value'.
    """
    updated = copy.deepcopy(config)
    recorded: List[str] = list(updated.get('overrides', []))
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'override must be key=value, got {item!r}')
        key, raw = item.split('=', 1)
        set_path(updated, key.strip(), _coerce(raw))
        recorded.append(item)
    if recorded:
        updated['overrides'] = recorded
    return updated


def parse_sweep(spec: str) -> Tuple[str, List[Any]]:
    """Parses a `--sweep a.b=v1,v2,v3` specification.

    Args:
        spec: The sweep specification.

    Returns:
        A tuple of (dotted path, list of coerced values).

    Raises:
        ValueError: If the specification is malformed.
    """
    if '=' not in spec:
        raise ValueError(f'sweep must be key=v1,v2,..., got {spec!r}')
    key, raw = spec.split('=', 1)
    values = [_coerce(part) for part in raw.split(',') if part.strip()]
    if not values:
        raise ValueError(f'sweep {spec!r} lists no values')
    return key.strip(), values


def build_parser(description: str) -> argparse.ArgumentParser:
    """Builds the control-panel argument parser shared by experiments.

    Args:
        description: Experiment description for `--help`.

    Returns:
        A parser carrying the standard controls.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--config',
        type=pathlib.Path,
        required=True,
        help='path to the TOML experiment configuration',
    )
    parser.add_argument(
        '--set',
        dest='overrides',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='override one configuration value',
    )
    parser.add_argument(
        '--sweep',
        default=None,
        metavar='KEY=V1,V2',
        help='run the experiment across a parameter range',
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='print the cross-methodology comparison table',
    )
    parser.add_argument(
        '--show-source',
        default=None,
        metavar='ARM',
        help='print an arm implementation and its source anchors',
    )
    parser.add_argument(
        '--no-write',
        action='store_true',
        help='do not write a run directory (useful while iterating)',
    )
    return parser
