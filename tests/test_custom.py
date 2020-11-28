import jax
import pytest
import tensorflow as tf
from autograd import grad
from fdm import check_sensitivity, gradient

import lab as B
from lab.custom import (
    toeplitz_solve, s_toeplitz_solve,
    bvn_cdf, s_bvn_cdf,
    expm, s_expm,
    logm, s_logm
)
from lab.tensorflow.custom import as_tf
from lab.torch.custom import as_torch, as_np
from .util import approx


def test_as_tf():
    assert isinstance(as_tf(B.randn()), B.TFNumeric)
    assert isinstance(as_tf((B.randn(),))[0], B.TFNumeric)


def test_as_torch_and_as_np():
    assert isinstance(as_torch(B.randn()), B.TorchNumeric)
    assert isinstance(as_torch((B.randn(),))[0], B.TorchNumeric)
    assert isinstance(as_np(as_torch(B.randn())), B.NPNumeric)
    assert isinstance(as_np(as_torch((B.randn(),)))[0], B.NPNumeric)


def check_grad(f, args, kw_args=None, digits=6):
    """Check the gradients of a function.

    Args:
        f (function): Function to check gradients of.
        args (tuple): Arguments to check `f` at.
        kw_args (tuple, optional): Keyword arguments to check `f` at. Defaults
            to no keyword arguments.
        digits (int, optional): Number of digits accuracy to check. Defaults to
            `6`.
    """
    # Default to no keyword arguments.
    if kw_args is None:
        kw_args = {}

    # Get the associated function in LAB.
    lab_f = getattr(B, f.__name__)

    def create_f_i(i, args_):
        # Create a function that only varies the `i`th argument.
        def f_i(x):
            return B.mean(lab_f(*(args_[:i] + (x,) + args_[i + 1:]), **kw_args))

        return f_i

    # Walk through the arguments.
    for i in range(len(args)):
        # Numerically compute gradient.
        f_i = create_f_i(i, args)
        numerical_grad = gradient(f_i)(args[i])

        # Check AutoGrad gradient.
        autograd_grad = grad(f_i)(args[i])
        approx(numerical_grad, autograd_grad, digits)

        # Check TensorFlow gradient.
        tf_args = tuple([as_tf(arg) for arg in args])
        f_i = create_f_i(i, tf_args)
        with tf.GradientTape() as t:
            t.watch(tf_args[i])
            tf_grad = t.gradient(f_i(tf_args[i]), tf_args[i]).numpy()
        approx(numerical_grad, tf_grad, digits)

        # Check PyTorch gradient.
        torch_args = tuple([as_torch(arg, grad=True) for arg in args])
        f_i = create_f_i(i, torch_args)
        f_i(torch_args[i]).backward()
        approx(numerical_grad, torch_args[i].grad, digits)

        # Check Jax gradient.
        torch_args = tuple([jax.device_put(arg) for arg in args])
        f_i = create_f_i(i, torch_args)
        jax_grad = jax.grad(f_i)(args[i])
        approx(numerical_grad, jax_grad, digits)


def test_toeplitz_solve():
    check_sensitivity(toeplitz_solve, s_toeplitz_solve,
                      (B.randn(3), B.randn(2), B.randn(3)))
    check_sensitivity(toeplitz_solve, s_toeplitz_solve,
                      (B.randn(3), B.randn(2), B.randn(3, 4)))
    check_grad(toeplitz_solve, (B.randn(3), B.randn(2), B.randn(3)))
    check_grad(toeplitz_solve, (B.randn(3), B.randn(2), B.randn(3, 4)))


def test_bvn_cdf():
    check_sensitivity(bvn_cdf, s_bvn_cdf, (B.rand(3), B.rand(3), B.rand(3)))
    check_grad(bvn_cdf, (B.rand(3), B.rand(3), B.rand(3)))


def test_expm():
    check_sensitivity(expm, s_expm, (B.randn(3, 3),))
    check_grad(expm, (B.randn(3, 3),))


@pytest.mark.xfail
def test_logm():
    mat = B.eye(3) + 0.1 * B.randn(3, 3)
    check_sensitivity(logm, s_logm, (mat,))
    check_grad(logm, (mat,))
