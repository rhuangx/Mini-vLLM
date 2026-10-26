"""layers/backends: names resolve to one shared backend each, and every backend has every operator."""

import pytest
from utils import CUDA_BUILT

from mini_vllm.layers.backends import NAMES, Backend, get_backend
from mini_vllm.layers.backends.torch_backend import TorchBackend

OPERATORS = [name for name in vars(Backend) if not name.startswith("_") and callable(vars(Backend)[name])]


def test_a_name_resolves_to_one_shared_backend():
    assert get_backend("torch") is get_backend("torch")
    assert get_backend() is get_backend("torch")
    assert get_backend("torch").name == "torch"


def test_a_backend_passes_through():
    backend = TorchBackend()
    assert get_backend(backend) is backend


def test_an_unknown_name_is_refused():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("tpu")


@pytest.mark.parametrize("name", [n for n in NAMES if n == "torch" or CUDA_BUILT])
def test_every_backend_has_every_operator(name):
    backend = get_backend(name)
    assert OPERATORS, "the protocol lists no operators"
    for operator in OPERATORS:
        assert callable(getattr(backend, operator, None)), f"{name} has no {operator}"


@pytest.mark.skipif(CUDA_BUILT, reason="the extension is built here")
def test_cuda_without_the_extension_fails_when_asked_for():
    with pytest.raises((ImportError, OSError)):
        get_backend("cuda")
