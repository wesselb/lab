import warnings

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch

import lab as B

# noinspection PyUnresolvedReferences
from .util import Tensor, PositiveTensor, approx, to_np, check_lazy_shapes


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


@pytest.mark.parametrize(
    "f, dtype_transform, just_single_arg",
    [
        (B.rand, lambda x: x, False),
        (B.randn, lambda x: x, False),
        (lambda *args: B.randint(*args, lower=0, upper=10), B.dtype_int, False),
        (B.randperm, B.dtype_int, True),
        (lambda *args: B.randgamma(*args, alpha=0.5, scale=0.5), lambda x: x, False),
        (lambda *args: B.randbeta(*args, alpha=0.5, beta=0.5), lambda x: x, False),
    ],
)
@pytest.mark.parametrize("t", [np.float32, tf.float32, torch.float32, jnp.float32])
def test_random_generators(f, t, dtype_transform, just_single_arg, check_lazy_shapes):
    # Test without specifying data type.
    if not just_single_arg:
        assert B.dtype(f()) is dtype_transform(B.default_dtype)
        assert B.shape(f()) == ()
    assert B.dtype(f(2)) is dtype_transform(B.default_dtype)
    assert B.shape(f(2)) == (2,)
    if not just_single_arg:
        assert B.dtype(f(2, 3)) is dtype_transform(B.default_dtype)
        assert B.shape(f(2, 3)) == (2, 3)

    # Test with specifying data type.
    state = B.create_random_state(t, 0)

    # Test direct specification.
    if not just_single_arg:
        assert B.dtype(f(t)) is dtype_transform(t)
        assert B.shape(f(t)) == ()
    assert B.dtype(f(t, 2)) is dtype_transform(t)
    assert B.shape(f(t, 2)) == (2,)
    if not just_single_arg:
        assert B.dtype(f(t, 2, 3)) is dtype_transform(t)
        assert B.shape(f(t, 2, 3)) == (2, 3)

    # Test state specification.
    if not just_single_arg:
        assert isinstance(f(state, t)[0], B.RandomState)
        assert B.dtype(f(state, t)[1]) is dtype_transform(t)
        assert B.shape(f(state, t)[1]) == ()
    assert isinstance(f(state, t, 2)[0], B.RandomState)
    assert B.dtype(f(state, t, 2)[1]) is dtype_transform(t)
    assert B.shape(f(state, t, 2)[1]) == (2,)
    if not just_single_arg:
        assert isinstance(f(state, t, 2, 3)[0], B.RandomState)
        assert B.dtype(f(state, t, 2, 3)[1]) is dtype_transform(t)
        assert B.shape(f(state, t, 2, 3)[1]) == (2, 3)

    if not just_single_arg:
        # Test reference specification.
        assert B.dtype(f(f(t))) is dtype_transform(t)
        assert B.shape(f(f())) == ()
        assert B.dtype(f(f(t, 2))) is dtype_transform(t)
        assert B.shape(f(f(t, 2))) == (2,)
        assert B.dtype(f(f(t, 2, 3))) is dtype_transform(t)
        assert B.shape(f(f(t, 2, 3))) == (2, 3)

        # Test state and reference specification.
        assert isinstance(f(state, f(t))[0], B.RandomState)
        assert B.dtype(f(state, f(t))[1]) is dtype_transform(t)
        assert B.shape(f(state, f(t))[1]) == ()
        assert isinstance(f(state, f(t, 2))[0], B.RandomState)
        assert B.dtype(f(state, f(t, 2))[1]) is dtype_transform(t)
        assert B.shape(f(state, f(t, 2))[1]) == (2,)
        assert isinstance(f(state, f(t, 2, 3))[0], B.RandomState)
        assert B.dtype(f(state, f(t, 2, 3))[1]) is dtype_transform(t)
        assert B.shape(f(state, f(t, 2, 3))[1]) == (2, 3)


@pytest.mark.parametrize("t", [np.float32, tf.float32, torch.float32, jnp.float32])
def test_randint_bounds(t, check_lazy_shapes):
    assert B.randint(t, lower=10, upper=11) == 10


@pytest.mark.parametrize("t", [np.float32, tf.float32, torch.float32, jnp.float32])
def test_randgamma_parameters(t, check_lazy_shapes):
    approx(B.randgamma(t, alpha=1, scale=0), 0, atol=1e-6)


@pytest.mark.parametrize("t", [np.float32, tf.float32, torch.float32, jnp.float32])
def test_randgamma_broadcasting(t, check_lazy_shapes):
    assert B.shape(B.randgamma(t, alpha=1, scale=0)) == ()
    assert B.shape(B.randgamma(t, alpha=B.rand(5), scale=0)) == (5,)
    assert B.shape(B.randgamma(t, alpha=B.rand(5), scale=B.rand(5))) == (5,)
    assert B.shape(B.randgamma(t, alpha=1, scale=B.rand(5))) == (5,)
    assert B.shape(B.randgamma(t, 3, alpha=B.rand(5), scale=0)) == (3, 5)
    assert B.shape(B.randgamma(t, 3, alpha=B.rand(5), scale=B.rand(5))) == (3, 5)
    assert B.shape(B.randgamma(t, 3, alpha=1, scale=B.rand(5))) == (3, 5)


@pytest.mark.parametrize("t", [np.float32, tf.float32, torch.float32, jnp.float32])
def test_randbeta_parameters(t, check_lazy_shapes):
    approx(B.randbeta(t, alpha=1e-6, beta=1), 0, atol=1e-6)
    approx(B.randbeta(t, alpha=1, beta=1e-6), 1, atol=1e-6)


def test_torch_global_random_state(mocker):
    # Check CPU specifications.
    B.ActiveDevice.active_name = None
    assert B.global_random_state(torch.float32) is torch.random.default_generator
    B.ActiveDevice.active_name = "cpu"
    assert B.global_random_state(torch.float32) is torch.random.default_generator

    # Test that `cuda.seed` is called to initialise the default generators.
    torch_cuda_init = mocker.patch("torch.cuda.init")
    B.ActiveDevice.active_name = "cuda"
    # The call is allowed to fail, because `torch.cuda.seed` is mocked, so it won't
    # actually populate `torch.cuda.default_generators`.
    with pytest.raises(IndexError):
        B.global_random_state(torch.float32)
    assert torch_cuda_init.called_once()

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


_test_randcat_ps = [PositiveTensor(2).forms(), PositiveTensor(3, 2).forms()]


@pytest.mark.parametrize("p", sum(_test_randcat_ps, []))
def test_randcat(p, check_lazy_shapes):
    state = B.create_random_state(B.dtype(p))

    # Determine the shape of a single sample.
    if p is not None:
        sample_shape = B.shape(p)[:-1]
    else:
        sample_shape = ()

    # Check shape.
    assert B.shape(B.randcat(p)) == sample_shape
    assert B.shape(B.randcat(p, 5)) == (5,) + sample_shape
    assert B.shape(B.randcat(p, 5, 5)) == (5, 5) + sample_shape

    assert isinstance(B.randcat(state, p)[0], B.RandomState)
    assert B.shape(B.randcat(state, p)[1]) == sample_shape
    assert B.shape(B.randcat(state, p, 5)[1]) == (5,) + sample_shape
    assert B.shape(B.randcat(state, p, 5, 5)[1]) == (5, 5) + sample_shape

    # Check correctness.
    dtype = B.dtype(p)
    choices = set(to_np(B.randcat(B.ones(dtype, 5), 1000)))
    assert choices == set(to_np(B.range(dtype, 5)))


def _test_choice_with_p(forms):
    pairs = [(form, None) for form in forms]
    for alternate in _test_randcat_ps:
        pairs += list(zip(forms, alternate))
    return pairs


@pytest.mark.parametrize(
    "x,p",
    _test_choice_with_p(Tensor(2).forms())
    + _test_choice_with_p(Tensor(2, 3).forms())
    + _test_choice_with_p(Tensor(2, 3, 4).forms()),
)
def test_choice(x, p, check_lazy_shapes):
    state = B.create_random_state(B.dtype(x))

    # Determine the shape of a single sample.
    sample_shape = B.shape(x)[1:]
    if p is not None:
        sample_shape = B.shape(p)[:-1] + sample_shape

    # Make `p` a dictionary so that we can optionally give it.
    p = {"p": p}

    # Check shape.
    assert B.shape(B.choice(x, **p)) == sample_shape
    assert B.shape(B.choice(x, 5, **p)) == (5,) + sample_shape
    assert B.shape(B.choice(x, 5, 5, **p)) == (5, 5) + sample_shape

    assert isinstance(B.choice(state, x, **p)[0], B.RandomState)
    assert B.shape(B.choice(state, x, **p)[1]) == sample_shape
    assert B.shape(B.choice(state, x, 5, **p)[1]) == (5,) + sample_shape
    assert B.shape(B.choice(state, x, 5, 5, **p)[1]) == (5, 5) + sample_shape

    # Check correctness.
    dtype = B.dtype(x)
    choices = set(to_np(B.choice(B.range(dtype, 5), 1000)))
    assert choices == set(to_np(B.range(dtype, 5)))
