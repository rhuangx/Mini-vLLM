"""Build the CUDA extension in place: uv run python src/extensions/build.py"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
TOOLKIT = ROOT / "build" / "cuda_toolkit"
DEFAULT_MAX_JOBS = 4
BYTES_PER_JOB = 2 * 1024**3


class ToolchainError(RuntimeError):
    """Raised when no CUDA toolkit matching torch's CUDA version can be found."""


def torch_cuda_major() -> int:
    if torch.version.cuda is None:
        raise ToolchainError(f"this torch ({torch.__version__}) is a CPU-only build")
    return int(torch.version.cuda.split(".")[0])


def nvcc_version(nvcc: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            [str(nvcc), "--version"], capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", out)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _nvidia_roots() -> list[Path]:
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return []
    return [Path(path) for path in spec.submodule_search_locations]


class WheelToolkit:
    def __init__(self, compiler_root: Path, include_dirs: list[Path], lib_dirs: list[Path]) -> None:
        self.compiler_root = compiler_root
        self.include_dirs = include_dirs
        self.lib_dirs = lib_dirs


def _find_wheel_toolkit(major: int) -> WheelToolkit | None:
    candidates: list[Path] = []
    for root in _nvidia_roots():
        candidates += [root / f"cu{major}", root / "cuda_nvcc", root / "cuda_runtime", root / "cuda_cccl"]
    compiler = next((candidate for candidate in candidates if (candidate / "bin" / "nvcc").is_file()), None)
    if compiler is None:
        return None
    include_dirs = [candidate / "include" for candidate in candidates if (candidate / "include").is_dir()]
    lib_dirs = [candidate / "lib" for candidate in candidates if (candidate / "lib").is_dir()]
    include_dirs.sort(key=lambda directory: not (directory / "cuda_runtime.h").is_file())
    if not include_dirs or not (include_dirs[0] / "cuda_runtime.h").is_file():
        return None
    return WheelToolkit(compiler, include_dirs, lib_dirs)


def _relink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.readlink() == target:
            return
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    elif link.exists():
        link.unlink()
    link.symlink_to(target)


def _ensure_real_dir(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def synthesize_cuda_home(toolkit: WheelToolkit, dest: Path = TOOLKIT) -> Path:
    """Assemble a directory that looks enough like a CUDA toolkit for nvcc and CMake.

    bin and nvvm are linked whole, because nvcc locates its backend relative to the real
    path of the binary. include and lib64 are merged symlinks from every wheel.
    """
    dest.mkdir(parents=True, exist_ok=True)
    _relink(dest / "bin", toolkit.compiler_root / "bin")
    if (toolkit.compiler_root / "nvvm").is_dir():
        _relink(dest / "nvvm", toolkit.compiler_root / "nvvm")

    include = dest / "include"
    _ensure_real_dir(include)
    for source_dir in toolkit.include_dirs:
        for src in sorted(source_dir.iterdir()):
            link = include / src.name
            if link.is_symlink() and link.readlink() != src:
                continue
            _relink(link, src)
        # nvcc 13 looks for CCCL next to the toolkit include root.
        cccl = source_dir / "cccl"
        if cccl.is_dir():
            for src in sorted(cccl.iterdir()):
                _relink(include / src.name, src)

    lib64 = dest / "lib64"
    _ensure_real_dir(lib64)
    for source_dir in toolkit.lib_dirs:
        for src in sorted(source_dir.iterdir()):
            if src.is_dir():
                continue
            _relink(lib64 / src.name, src)
            if ".so." in src.name:
                bare = lib64 / (src.name.split(".so.")[0] + ".so")
                if not bare.is_symlink():
                    _relink(bare, src)
    _relink(dest / "lib", lib64)
    return dest


def resolve_cuda_home() -> Path:
    """A CUDA toolkit whose nvcc major matches torch's, assembled from the pip wheels."""
    want = torch_cuda_major()
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(var):
            home = Path(os.environ[var])
            version = nvcc_version(home / "bin" / "nvcc")
            if version is not None and version[0] == want:
                return home
    wheel = _find_wheel_toolkit(want)
    if wheel is None:
        raise ToolchainError(f"no CUDA {want} toolkit in the nvidia wheels; install nvidia-cuda-nvcc")
    home = synthesize_cuda_home(wheel)
    version = nvcc_version(home / "bin" / "nvcc")
    if version is None or version[0] != want:
        raise ToolchainError(f"assembled nvcc at {home} is {version}, wanted major {want}")
    return home


def max_jobs() -> int:
    override = os.environ.get("MAX_JOBS")
    if override:
        return max(1, min(DEFAULT_MAX_JOBS, int(override)))
    jobs = min(DEFAULT_MAX_JOBS, os.cpu_count() or 1)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                jobs = min(jobs, max(1, available // BYTES_PER_JOB))
                break
    except OSError:
        pass
    return max(1, jobs)


def main() -> None:
    home = resolve_cuda_home()
    env = os.environ.copy()
    env["CUDA_HOME"] = str(home)
    env["CUDA_PATH"] = str(home)
    env["PATH"] = f"{home / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["MAX_JOBS"] = str(max_jobs())
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    subprocess.run(
        [
            "cmake", "-G", "Ninja", "-S", str(ROOT), "-B", str(build),
            f"-DCMAKE_CUDA_COMPILER={home / 'bin' / 'nvcc'}",
            f"-DPython_EXECUTABLE={sys.executable}",
        ],
        check=True, env=env, cwd=ROOT,
    )
    subprocess.run(["cmake", "--build", str(build)], check=True, env=env, cwd=ROOT)
    library = ROOT / "mini_vllm_ext" / "_ext.so"
    if not library.is_file():
        raise ToolchainError(f"build finished without {library}")
    print(f"built {library}")


if __name__ == "__main__":
    main()
