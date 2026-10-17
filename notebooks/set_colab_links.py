"""Points the Colab links at your repository.

The notebooks ship without a badge, because a badge pointing at a
repository that does not exist is worse than no badge. Once yours
exists, run this once:

    python set_colab_links.py treaper33/ml-mastery

It adds an "Open in Colab" badge as the first cell of every notebook and
fills in the real URLs in the READMEs. Re-running it updates them, so
renaming the repository is one command, and `--remove` strips them
again.

The badge is a link, not an embed: nothing is fetched when you read the
notebook locally.

Usage:
    python set_colab_links.py OWNER/REPO [--branch main] [--path ml-mastery]
    python set_colab_links.py --remove
"""

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, List, Optional, Sequence

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

BADGE_MARKER = '<!-- colab-badge -->'
PLACEHOLDER = '<user>/<repo>'


def colab_url(slug: str, branch: str, path: str, notebook: str) -> str:
    """Builds the Colab URL for one notebook.

    Args:
        slug: 'owner/repo'.
        branch: Branch name.
        path: Path of the repository root inside the repo, '' when the
            repository root is this directory's parent.
        notebook: Notebook file name.

    Returns:
        The full URL.
    """
    prefix = f'{path}/' if path else ''
    return (f'https://colab.research.google.com/github/{slug}/blob/'
            f'{branch}/{prefix}notebooks/{notebook}')


def badge_cell(url: str) -> Dict:
    """Builds the markdown badge cell.

    Args:
        url: The Colab URL.

    Returns:
        An nbformat markdown cell.
    """
    source = (f'{BADGE_MARKER}\n'
              f'[![Open In Colab](https://colab.research.google.com/assets/'
              f'colab-badge.svg)]({url})')
    return {'cell_type': 'markdown', 'metadata': {},
            'source': [line + '\n' for line in source.split('\n')[:-1]]
                      + [source.split('\n')[-1]]}


def update_notebook(path: pathlib.Path, url: Optional[str]) -> str:
    """Adds, replaces or removes a notebook's badge cell.

    Args:
        path: The notebook.
        url: The Colab URL, or None to remove the badge.

    Returns:
        A one-word description of what changed.
    """
    notebook = json.loads(path.read_text(encoding='utf-8'))
    cells = notebook['cells']
    has_badge = bool(cells) and BADGE_MARKER in ''.join(cells[0]['source'])

    if url is None:
        if not has_badge:
            return 'unchanged'
        notebook['cells'] = cells[1:]
        action = 'removed'
    elif has_badge:
        notebook['cells'][0] = badge_cell(url)
        action = 'updated'
    else:
        notebook['cells'] = [badge_cell(url)] + cells
        action = 'added'

    path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    return action


def update_readmes(slug: Optional[str], branch: str, path: str) -> List[str]:
    """Replaces the placeholder slug in the READMEs.

    Args:
        slug: 'owner/repo', or None to restore the placeholder.
        branch: Branch name.
        path: Repository-relative path prefix.

    Returns:
        The names of the files changed.
    """
    targets = [HERE / 'README.md', REPO_ROOT / 'README.md']
    changed = []
    for target in targets:
        if not target.exists():
            continue
        text = target.read_text(encoding='utf-8')
        if slug is None:
            updated = re.sub(r'github/[\w.-]+/[\w.-]+/blob/[\w.-]+/',
                             f'github/{PLACEHOLDER}/blob/main/', text)
        else:
            prefix = f'{path}/' if path else ''
            updated = text.replace(
                f'github/{PLACEHOLDER}/blob/main/ml-mastery/notebooks/',
                f'github/{slug}/blob/{branch}/{prefix}notebooks/')
            updated = re.sub(
                r'github/[\w.-]+/[\w.-]+/blob/[\w.-]+/(\S*?)notebooks/',
                f'github/{slug}/blob/{branch}/{prefix}notebooks/', updated)
        if updated != text:
            target.write_text(updated, encoding='utf-8')
            changed.append(str(target.relative_to(REPO_ROOT)))
    return changed


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument vector.

    Returns:
        0 on success, 2 on a bad slug.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('slug', nargs='?',
                        help="your repository, as 'owner/repo'")
    parser.add_argument('--branch', default='main')
    parser.add_argument('--path', default='',
                        help='path of this folder inside the repository; '
                             'empty when this folder IS the repository root')
    parser.add_argument('--remove', action='store_true',
                        help='strip the badges again')
    args = parser.parse_args(argv)

    if not args.remove and not args.slug:
        parser.error('give a repository as owner/repo, or pass --remove')
    if args.slug and not re.fullmatch(r'[\w.-]+/[\w.-]+', args.slug):
        print(f'{args.slug!r} does not look like owner/repo')
        return 2

    notebooks = sorted(HERE.glob('*.ipynb'))
    if not notebooks:
        print('no notebooks found')
        return 2

    path = args.path.strip('/')
    for notebook in notebooks:
        url = (None if args.remove
               else colab_url(args.slug, args.branch, path, notebook.name))
        action = update_notebook(notebook, url)
        print(f'{notebook.name:<38} badge {action}')

    changed = update_readmes(None if args.remove else args.slug,
                             args.branch, path)
    for name in changed:
        print(f'{name:<38} links updated')

    if not args.remove:
        example = colab_url(args.slug, args.branch, path, notebooks[0].name)
        print(f'\nFirst notebook now opens at:\n  {example}')
        print('\nIf the repository is private, Colab will ask once for '
              'GitHub\naccess: File -> Open notebook -> GitHub -> tick '
              '"Include private repos".')
    return 0


if __name__ == '__main__':
    sys.exit(main())
