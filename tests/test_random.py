import warnings

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch

import lab as B

# noinspection PyUnresolvedReferences
from .util import Tensor, approx, to_np, check_lazy_shapes


@pytest.mark.parametrize(
    "dtype, f_plain",
    [
        (np.float32, np.random.randn),
        (tf.float32, lambda: tf.random.normal(())),
        (torch.float32, lambda: torch.randn(())),
        (jnp.float32, lambda: 1),
    ],
)
def test_set_seed_set_global_random_state(dtype, f_plain, check_lazy_shapes):
    B.set_random_seed(0)
    x1 = to_np(B.rand(dtype))
    x2 = to_np(f_plain())
    B.set_random_seed(0)
    y1 = to_np(B.rand(dtype))
    y2 = to_np(f_plain())
    assert x1 == y1
    assert x2 == y2

    B.set_global_random_state(B.create_random_state(dtype, seed=0))
    x1 = to_np(B.rand(dtype))
    x2 = to_np(f_plain())
    B.set_global_random_state(B.create_random_state(dtype, seed=0))
    y1 = to_np(B.rand(dtype))
    y2 = to_np(f_plain())
    assert x1 == y1
    # TODO: Make this work with TF!
    if not isinstance(dtype, B.TFDType):
        assert x2 == y2


@pytest.mark.parametrize("dtype", [np.float32, tf.float32, torch.float32, jnp.float32])
def test_create_random_state(dtype):
    # Test specification without argument.
    B.create_random_state(dtype)

    # Check that it does the right thing.
    state = B.create_random_state(dtype, seed=0)
    state, x1 = B.rand(state, dtype)
    state, x2 = B.rand(state, dtype)
    x1, x2 = to_np(x1), to_np(x2)

    state = B.create_random_state(dtype, seed=0)
    state, y1 = B.rand(state, dtype)
    state, y2 = B.rand(state, dtype)
    y1, y2 = to_np(y1), to_np(y2)

    assert x1 != x2
    assert x1 == y1
    assert x2 == y2


@pytest.mark.parametrize("f", [B.rand, B.randn])
def test_random_generators(f, check_lazy_shapes):
    # Test without specifying data type.
    assert B.dtype(f()) is B.default_dtype
    assert B.shape(f()) == ()
    assert B.dtype(f(2)) is B.default_dtype
    approx(B.shape(f(2)), (2,))
    assert B.dtype(f(2, 3)) is B.default_dtype
    assert B.shape(f(2, 3)) == (2, 3)

    # Test with specifying data type.
    for t in [np.float32, tf.float32, torch.float32, jnp.float32]:
        state = B.create_random_state(t, 0)

        # Test direct specification.
        assert B.dtype(f(t)) is t
        assert B.shape(f(t)) == ()
        assert B.dtype(f(t, 2)) is t
        assert B.shape(f(t, 2)) == (2,)
        assert B.dtype(f(t, 2, 3)) is t
        assert B.shape(f(t, 2, 3)) == (2, 3)

        assert isinstance(f(state, t)[0], B.RandomState)
        assert B.dtype(f(state, t)[1]) is t
        assert B.shape(f(state, t)[1]) == ()
        assert B.dtype(f(state, t, 2)[1]) is t
        assert B.shape(f(state, t, 2)[1]) == (2,)
        assert B.dtype(f(state, t, 2, 3)[1]) is t
        assert B.shape(f(state, t, 2, 3)[1]) == (2, 3)

        # Test reference specification.
        assert B.dtype(f(f(t))) is t
        assert B.shape(f(f())) == ()
        assert B.dtype(f(f(t, 2))) is t
        assert B.shape(f(f(t, 2))) == (2,)
        assert B.dtype(f(f(t, 2, 3))) is t
        assert B.shape(f(f(t, 2, 3))) == (2, 3)

        # Must stay within the framework now.
        assert isinstance(f(state, f(t))[0], B.RandomState)
        assert B.dtype(f(state, f(t))[1]) is t
        assert B.dtype(f(state, f(t, 2))[1]) is t
        assert B.shape(f(state, f(t, 2))[1]) == (2,)
        assert B.dtype(f(state, f(t, 2, 3))[1]) is t
        assert B.shape(f(state, f(t, 2, 3))[1]) == (2, 3)


def test_torch_global_random_state(mocker):
    # Check CPU specifications.
    B.ActiveDevice.active_name = None
    assert B.global_random_state(torch.float32) is torch.random.default_generator
    B.ActiveDevice.active_name = "cpu"
    assert B.global_random_state(torch.float32) is torch.random.default_generator

    # Test that `cuda.seed` is called to initialise the default generators.
    torch_cuda_seed = mocker.patch("torch.cuda.seed")
    B.ActiveDevice.active_name = "cuda"
    # The call is allowed to fail, because `torch.cuda.seed` is mocked, so it won't
    # actually populate `torch.cuda.default_generators`.
    with pytest.raises(IndexError):
        B.global_random_state(torch.float32)
    assert torch_cuda_seed.called_once()

    # Now set some fake default generators.
    torch.cuda.default_generators = (0, 1)

    # Check GPU specifications.
    B.ActiveDevice.active_name = "cuda"
    assert B.global_random_state(torch.float32) == 0
    B.ActiveDevice.active_name = "gpu"
    assert B.global_random_state(torch.float32) == 0
    B.ActiveDevice.active_name = "gpu:0"
    assert B.global_random_state(torch.float32) == 0
    B.ActiveDevice.active_name = "gpu:1"
    assert B.global_random_state(torch.float32) == 1
    with pytest.raises(RuntimeError):
        B.ActiveDevice.active_name = "weird-device"
        assert B.global_random_state(torch.float32) == 1

    # Reset back to defaults.
    torch.cuda.default_generators = ()
    B.ActiveDevice.active_name = None


@pytest.mark.parametrize("f", [B.rand, B.randn])
def test_conversion_warnings(f, check_lazy_shapes):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Trigger the warning!
        f(int, 5)

        assert len(w) == 1


def test_choice(check_lazy_shapes):
    # TODO: Can we use a parametrised test here?
    for x in Tensor(2).forms() + Tensor(2, 3).forms() + Tensor(2, 3, 4).forms():
        state = B.create_random_state(B.dtype(x))

        # Check shape.
        assert B.shape(B.choice(x)) == B.shape(x)[1:]
        assert B.shape(B.choice(x, 1)) == B.shape(x)[1:]
        assert B.shape(B.choice(x, 5))[0] == 5
        assert B.shape(B.choice(x, 5))[1:] == B.shape(x)[1:]

        assert isinstance(B.choice(state, x)[0], B.RandomState)
        assert B.shape(B.choice(state, x)[1]) == B.shape(x)[1:]
        assert B.shape(B.choice(state, x, 1)[1]) == B.shape(x)[1:]
        assert B.shape(B.choice(state, x, 5)[1])[0] == 5
        assert B.shape(B.choice(state, x, 5)[1])[1:] == B.shape(x)[1:]

        # Check correctness.
        dtype = B.dtype(x)
        choices = set(to_np(B.choice(B.range(dtype, 5), 1000)))
        assert choices == set(to_np(B.range(dtype, 5)))
