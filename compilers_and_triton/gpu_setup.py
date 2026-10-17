"""Installs Triton on Windows without touching your Python environment.

Triton's official wheels are Linux-only. On Linux or in Colab there is
nothing to do here -- `pip install triton`, or use the PyTorch that
already ships with it. This script exists for Windows, where the
community `triton-windows` wheel works but needs three things arranged
around it.

**Why not just pip install it into the environment you use?** Because
a Triton install drags in a compiler toolchain's worth of expectations,
and if the environment already has a working CUDA PyTorch you have
more to lose than to gain. This installs to a prefix of its own with
`pip --target`, leaves `site-packages` alone, and injects the prefix
through `PYTHONPATH` for the duration of one process. Nothing is
written into your environment and uninstalling is `--remove`.

The three things it arranges:

1. **MSVC on PATH.** Triton compiles a small launcher stub with
   `cl.exe` the first time a kernel runs. The environment comes from
   `vcvars64.bat`, located through `vswhere.exe`.
2. **The prefix on PYTHONPATH**, so `import triton` finds it.
3. **CUDA_PATH pointing at the wheel's own bundled stubs.** This is
   the step nobody tells you about. Without it the link fails:

       LINK : fatal error LNK1181: cannot open input file 'cuda.lib'

   and `triton-windows` advises installing the CUDA Toolkit. You do
   not need it. The wheel already contains `cuda.h` and
   `lib/x64/cuda.lib`, laid out exactly like a Toolkit -- it simply
   puts only `lib/` on the link path, not `lib/x64/`. Pointing
   CUDA_PATH at the bundled `triton/backends/nvidia` directory makes
   Triton add the right path itself. No download, no file copied.

Usage:
    python gpu_setup.py --install
    python gpu_setup.py --check
    python gpu_setup.py --run run_all.py
    python gpu_setup.py --run notebooks/validate_notebooks.py
    python gpu_setup.py --remove

The default prefix is under LOCALAPPDATA, deliberately outside this
repository: it is about 150 MB and belongs to your machine, not to the
project. Override it with --prefix or the ML_TRITON_PREFIX variable.
"""

import argparse
import glob
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Optional, Sequence

HERE = pathlib.Path(__file__).resolve().parent

PACKAGE = 'triton-windows'

PROBE = """
import torch
import triton
import triton.language as tl

print(f'  torch    {torch.__version__} / cuda {torch.version.cuda}')
print(f'  triton   {triton.__version__}')
if not torch.cuda.is_available():
    print('  gpu      none visible -- torch cannot see a CUDA device')
    raise SystemExit(1)
major, minor = torch.cuda.get_device_capability(0)
print(f'  gpu      {torch.cuda.get_device_name(0)} (sm_{major}{minor})')


@triton.jit
def _add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    \"\"\"Adds one, to prove a kernel compiles and launches.\"\"\"
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets,
             tl.load(x_ptr + offsets, mask=mask) + 1.0, mask=mask)


x = torch.arange(10, device='cuda', dtype=torch.float32)
out = torch.empty_like(x)
_add_one[(1,)](x, out, x.numel(), BLOCK=16)
torch.cuda.synchronize()
ok = bool(torch.allclose(out, x + 1))
print(f'  kernel   compiled and launched, result correct: {ok}')
raise SystemExit(0 if ok else 1)
"""


def default_prefix() -> pathlib.Path:
    """Returns where the isolated install lives.

    Outside the repository on purpose: it is a machine-sized binary
    artefact, and nothing in version control should be able to see it.

    Returns:
        The prefix directory.
    """
    override = os.environ.get('ML_TRITON_PREFIX')
    if override:
        return pathlib.Path(override).expanduser().resolve()
    # The home directory, not LOCALAPPDATA. Packaged applications get a
    # redirected LOCALAPPDATA -- run this from inside one and the
    # install lands in that application's private cache, where your own
    # terminal will not find it and a cleanup may delete it. This was
    # not hypothetical: the first run of this script put 158 MB in
    # AppData\Packages\...\LocalCache. Home is predictable everywhere.
    return pathlib.Path.home().resolve() / '.ml-mastery-gpu' / 'triton'


def find_vcvars() -> Optional[pathlib.Path]:
    """Locates `vcvars64.bat`.

    Tries `vswhere.exe`, which is the supported way and is installed
    with any Visual Studio since 2017, then falls back to globbing the
    standard install roots.

    Returns:
        The path to the script, or None when MSVC is not installed.
    """
    vswhere = pathlib.Path(
        os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
        'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
    if vswhere.is_file():
        try:
            output = subprocess.run(
                [str(vswhere), '-latest', '-products', '*', '-requires',
                 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                 '-property', 'installationPath'],
                capture_output=True, text=True, check=True).stdout.strip()
        except subprocess.CalledProcessError:
            output = ''
        if output:
            candidate = (pathlib.Path(output.splitlines()[0])
                         / 'VC' / 'Auxiliary' / 'Build' / 'vcvars64.bat')
            if candidate.is_file():
                return candidate

    for root in (os.environ.get('ProgramFiles', r'C:\Program Files'),
                 os.environ.get('ProgramFiles(x86)',
                                r'C:\Program Files (x86)')):
        pattern = os.path.join(root, 'Microsoft Visual Studio', '*', '*',
                               'VC', 'Auxiliary', 'Build', 'vcvars64.bat')
        matches = sorted(glob.glob(pattern))
        if matches:
            return pathlib.Path(matches[-1])
    return None


def msvc_environment(vcvars: pathlib.Path) -> Dict[str, str]:
    """Captures the environment `vcvars64.bat` sets.

    The batch file exports dozens of variables and there is no supported
    way to read them except to run it and ask the shell what changed,
    which is what every build system does.

    Args:
        vcvars: Path to `vcvars64.bat`.

    Returns:
        The resulting environment.

    Raises:
        RuntimeError: If the script produced no usable environment.
    """
    # Via a temporary batch file rather than `cmd /c "... && set"`.
    # The path contains spaces and parentheses, and every layer that
    # command passes through -- subprocess's list2cmdline, then cmd's
    # own parser -- gets a chance to mangle the quoting. Putting it in
    # a file removes all of those layers at once.
    with tempfile.TemporaryDirectory() as directory:
        script = pathlib.Path(directory) / 'capture.bat'
        script.write_text(f'@echo off\r\n'
                          f'call "{vcvars}" >nul 2>&1\r\n'
                          f'set\r\n', encoding='utf-8')
        result = subprocess.run(['cmd', '/c', str(script)],
                                capture_output=True, text=True)

    environment = dict(os.environ)
    found = 0
    for line in result.stdout.splitlines():
        if '=' in line:
            key, _, value = line.partition('=')
            environment[key] = value
            found += 1
    if 'VCToolsInstallDir' not in environment or found < 10:
        detail = (result.stderr.strip().splitlines() or ['no output'])[-1]
        raise RuntimeError(f'{vcvars.name} did not set up a build '
                           f'environment ({detail}). Try running it in a '
                           f'terminal to see what it reports.')
    return environment


def kernel_environment(prefix: pathlib.Path) -> Dict[str, str]:
    """Builds the environment a Triton kernel compile needs.

    Args:
        prefix: The isolated install directory.

    Returns:
        An environment mapping ready for `subprocess`.

    Raises:
        FileNotFoundError: If MSVC or the install is missing.
    """
    if not (prefix / 'triton').is_dir():
        raise FileNotFoundError(
            f'no Triton at {prefix}; run: python gpu_setup.py --install')

    vcvars = find_vcvars()
    if vcvars is None:
        raise FileNotFoundError(
            'no MSVC found. Install "Desktop development with C++" from '
            'the Visual Studio Build Tools; Triton needs cl.exe to build '
            'its launcher stub.')

    environment = msvc_environment(vcvars)
    cuda = prefix / 'triton' / 'backends' / 'nvidia'

    existing = environment.get('PYTHONPATH', '')
    environment['PYTHONPATH'] = (f'{prefix}{os.pathsep}{existing}'
                                 if existing else str(prefix))
    environment['CUDA_PATH'] = str(cuda)
    environment['CUDA_HOME'] = str(cuda)
    environment['LIB'] = (f'{cuda / "lib" / "x64"}{os.pathsep}'
                          f'{environment.get("LIB", "")}')
    environment['TRITON_CACHE_DIR'] = str(prefix.parent / 'cache')
    environment['PYTHONIOENCODING'] = 'utf-8'
    return environment


def install(prefix: pathlib.Path) -> int:
    """Installs the wheel into the prefix.

    Args:
        prefix: Where to put it.

    Returns:
        A process exit code.
    """
    prefix.mkdir(parents=True, exist_ok=True)
    print(f'installing {PACKAGE} into {prefix}')
    print('(nothing is written into your Python environment)')
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
         '--upgrade', '--target', str(prefix), PACKAGE])
    if result.returncode != 0:
        return result.returncode

    size = sum(f.stat().st_size for f in prefix.rglob('*') if f.is_file())
    print(f'\ninstalled, {size / 1e6:.0f} MB')
    return check(prefix)


def check(prefix: pathlib.Path) -> int:
    """Compiles and launches one kernel, to prove the setup works.

    Args:
        prefix: The install directory.

    Returns:
        A process exit code.
    """
    print('\nchecking the install')
    try:
        environment = kernel_environment(prefix)
    except (FileNotFoundError, RuntimeError) as error:
        print(f'  {error}')
        return 1

    print(f'  prefix   {prefix}')
    # The probe goes to a real file, not `python -c`. `@triton.jit`
    # reads its function's source through `inspect.getsourcelines`, so
    # code with no file raises "@jit functions should be defined in a
    # Python file". An earlier version of this script used -c and hit
    # exactly that -- the same defect this repository's notebook
    # validator had, for the same reason.
    with tempfile.TemporaryDirectory() as directory:
        probe = pathlib.Path(directory) / 'probe_triton.py'
        probe.write_text(PROBE, encoding='utf-8')
        result = subprocess.run([sys.executable, str(probe)],
                                env=environment)
    if result.returncode == 0:
        print('\nready. Run the ladder with:')
        print('  python gpu_setup.py --run run_all.py')
    return result.returncode


def run(prefix: pathlib.Path, target: str,
        arguments: Sequence[str]) -> int:
    """Runs a script with Triton available.

    Args:
        prefix: The install directory.
        target: Script path, relative to this directory or absolute.
        arguments: Arguments to pass through.

    Returns:
        The script's exit code.
    """
    script = pathlib.Path(target)
    if not script.is_absolute():
        script = HERE / target
    if not script.is_file():
        print(f'no such script: {script}')
        return 2

    try:
        environment = kernel_environment(prefix)
    except (FileNotFoundError, RuntimeError) as error:
        print(f'{error}')
        return 1

    return subprocess.run([sys.executable, str(script), *arguments],
                          env=environment).returncode


def remove(prefix: pathlib.Path) -> int:
    """Deletes the isolated install.

    Args:
        prefix: The install directory.

    Returns:
        A process exit code.
    """
    for path in (prefix, prefix.parent / 'cache'):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f'removed {path}')
        else:
            print(f'nothing at {path}')
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument vector.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prefix', type=pathlib.Path,
                        default=default_prefix(),
                        help='where the isolated install lives')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--install', action='store_true')
    group.add_argument('--check', action='store_true')
    group.add_argument('--remove', action='store_true')
    group.add_argument('--run', metavar='SCRIPT',
                       help='run a script with Triton available')
    parser.add_argument('arguments', nargs='*',
                        help='arguments for --run')
    args = parser.parse_args(argv)

    if sys.platform != 'win32':
        print('This script is for Windows. Elsewhere Triton installs')
        print('normally:  pip install triton')
        print('and the modules in this section need nothing special.')
        return 1

    prefix: pathlib.Path = args.prefix.expanduser().resolve()
    if args.install:
        return install(prefix)
    if args.check:
        return check(prefix)
    if args.remove:
        return remove(prefix)
    return run(prefix, args.run, args.arguments)


if __name__ == '__main__':
    sys.exit(main())
