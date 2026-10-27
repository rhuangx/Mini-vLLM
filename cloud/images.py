"""Container images, layered so the slow steps stay cached: the torch layer depends only on
pyproject.toml and uv.lock, the CUDA extension only on src/extensions, and the Python sources
are added at container start, so editing them rebuilds nothing."""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
REMOTE = "/root/mini-vllm"   # the checkout's layout, inside the container
HF_HOME = "/hf"              # the Hugging Face cache, on a volume
RESULTS = "/results"         # results, on a volume
VLLM_VERSION = "0.31.0"      # the baseline; pinned so its numbers stay comparable

_PYCACHE = ["**/__pycache__", "**/*.pyc"]


def mini_image(cuda_archs: str = "90a") -> modal.Image:
    """Mini-vLLM's locked environment (torch cu130, nvcc 13.2 from the dev group) with the CUDA
    extension built for cuda_archs. nvcc needs no GPU, so the build runs on a CPU builder."""
    return (
        # gcc for CMake and for Triton's JIT launcher; this Python ships its headers.
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("build-essential", "git")
        .uv_sync(str(REPO), groups=["dev"])
        .env({
            "HF_HOME": HF_HOME,
            "PYTHONPATH": f"/root:{REMOTE}/src:{REMOTE}/src/extensions",
            "MINI_VLLM_CUDA_ARCHS": cuda_archs,
        })
        .add_local_dir(REPO / "src" / "extensions", f"{REMOTE}/src/extensions", copy=True,
                       ignore=["build", "**/*.so", *_PYCACHE])
        .run_commands(f"cd {REMOTE} && python src/extensions/build.py")
        .add_local_file(REPO / "pyproject.toml", f"{REMOTE}/pyproject.toml")
        .add_local_dir(REPO / "src" / "mini_vllm", f"{REMOTE}/src/mini_vllm", ignore=_PYCACHE)
        .add_local_dir(REPO / "tests", f"{REMOTE}/tests", ignore=_PYCACHE)
        .add_local_python_source("cloud")
    )


def vllm_image() -> modal.Image:
    """The pinned vLLM release with its own torch; it shares only cloud/ with Mini-vLLM."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("build-essential")
        .uv_pip_install(f"vllm=={VLLM_VERSION}")
        # FlashInfer's top-k/top-p sampler JIT-compiles with an nvcc this image does not have.
        # Every workload is greedy, which never reaches that sampler, so its absence costs nothing.
        .env({"HF_HOME": HF_HOME, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
        .add_local_python_source("cloud")
    )
