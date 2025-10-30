import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch
from autograd import grad
from fdm import check_sensitivity, gradient
from plum import isinstance

import lab as B
from lab.custom import (
    bvn_cdf,
    expm,
    logm,
    s_bvn_cdf,
    s_expm,
    s_logm,
    s_toeplitz_solve,
    toeplitz_solve,
)
from lab.tensorflow.custom import as_tf
from lab.torch.custom import as_torch

# noinspection PyUnresolvedReferences
from .util import PSD, approx, check_function, check_lazy_shapes


def test_as_tf(check_lazy_shapes):
    assert isinstance(as_tf(B.randn()), B.TFNumeric)
    assert isinstance(as_tf((B.randn(),))[0], B.TFNumeric)


def test_as_torch(check_lazy_shapes):
    assert isinstance(as_torch(B.randn()), B.TorchNumeric)
    assert isinstance(as_torch((B.randn(),))[0], B.TorchNumeric)


def check_grad(f, args, kw_args=None, rtol=1e-8, atol=0):
    """Check the gradients of a function.

    Args:
        f (function): Function to check gradients of.
        args (tuple): Arguments to check `f` at.
        kw_args (tuple, optional): Keyword arguments to check `f` at. Defaults
            to no keyword arguments.
        rtol (float, optional): Relative tolerance. Defaults to `1e-8`.
        atol (float, optional): Absolute tolerance. Defaults to `0`.
    """
    # Default to no keyword arguments.
    if kw_args is None:
        kw_args = {}

    # Get the associated function in LAB.
    lab_f = getattr(B, f.__name__)

    def create_f_i(i, args_):
        # Create a function that only varies the `i`th argument.
        def f_i(x):
            return B.mean(lab_f(*(args_[:i] + (x,) + args_[i + 1 :]), **kw_args))

        return f_i

    # Walk through the arguments.
    for i in range(len(args)):
        # Numerically compute gradient.
        f_i = create_f_i(i, args)
        numerical_grad = gradient(f_i)(args[i])

        # Check AutoGrad gradient.
        autograd_grad = grad(f_i)(args[i])
        approx(numerical_grad, autograd_grad, rtol=rtol, atol=atol)

        # Check TensorFlow gradient.
        tf_args = tuple([as_tf(arg) for arg in args])
        f_i = tf.function(create_f_i(i, tf_args), autograph=False)
        with tf.GradientTape() as t:
            t.watch(tf_args[i])
            tf_grad = t.gradient(f_i(tf_args[i]), tf_args[i]).numpy()
        approx(numerical_grad, tf_grad, rtol=rtol, atol=atol)

        # Check PyTorch gradient.
        torch_args = tuple([as_torch(arg, grad=False) for arg in args])
        f_i = torch.jit.trace(create_f_i(i, torch_args), torch_args[i])
        arg = torch_args[i].requires_grad_(True)
        f_i(arg).backward()
        approx(numerical_grad, arg.grad, rtol=rtol, atol=atol)

        # Check JAX gradient.
        jax_args = tuple([jnp.asarray(arg) for arg in args])
        f_i = create_f_i(i, jax_args)
        jax_grad = jax.jit(jax.grad(f_i))(jax_args[i])
        approx(numerical_grad, jax_grad, rtol=rtol, atol=atol)


def test_toeplitz_solve(check_lazy_shapes):
    check_sensitivity(
        toeplitz_solve, s_toeplitz_solve, (B.randn(3), B.randn(2), B.randn(3))
    )
    check_sensitivity(
        toeplitz_solve, s_toeplitz_solve, (B.randn(3), B.randn(2), B.randn(3, 4))
    )
    check_grad(toeplitz_solve, (B.randn(3), B.randn(2), B.randn(3)))
    check_grad(toeplitz_solve, (B.randn(3), B.randn(2), B.randn(3, 4)))


def test_bvn_cdf(check_lazy_shapes):
    check_sensitivity(bvn_cdf, s_bvn_cdf, (B.rand(3), B.rand(3), B.rand(3)))
    check_grad(bvn_cdf, (B.rand(3), B.rand(3), B.rand(3)), atol=1e-8)

    # Check that function runs on both `float32`s and `float64`s.
    a, b, c = B.rand(3), B.rand(3), B.rand(3)
    approx(
        B.bvn_cdf(a, b, c),
        B.bvn_cdf(B.cast(np.float32, a), B.cast(np.float32, b), B.cast(np.float32, c)),
    )

    # Check that, in JAX, the function check the shape of the inputs.
    with pytest.raises(ValueError):
        B.bvn_cdf(
            B.rand(jnp.float32, 2), B.rand(jnp.float32, 3), B.rand(jnp.float32, 3)
        )
    with pytest.raises(ValueError):
        B.bvn_cdf(
            B.rand(jnp.float32, 3), B.rand(jnp.float32, 2), B.rand(jnp.float32, 3)
        )
    with pytest.raises(ValueError):
        B.bvn_cdf(
            B.rand(jnp.float32, 3), B.rand(jnp.float32, 3), B.rand(jnp.float32, 2)
        )


def test_expm(check_lazy_shapes):
    check_sensitivity(expm, s_expm, (B.randn(3, 3),))
    check_grad(expm, (B.randn(3, 3),))


def test_logm_forward(check_lazy_shapes):
    # This test can be removed once the gradient is implemented and the below test
    # passes.
    check_function(B.logm, (PSD(3),))


@pytest.mark.xfail
def test_logm(check_lazy_shapes):
    mat = B.eye(3) + 0.1 * B.randn(3, 3)
    check_sensitivity(logm, s_logm, (mat,))
    check_grad(logm, (mat,))
