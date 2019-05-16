# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import lab as B
import tensorflow as tf
from autograd import grad
from fdm import check_sensitivity, gradient
from lab.custom import toeplitz_solve, s_toeplitz_solve
from lab.tensorflow.custom import as_tf
from lab.torch.custom import as_torch, as_np

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx, \
    eeq, assert_isinstance


def test_as_tf():
    yield assert_isinstance, as_tf(B.randn()), B.TFNumeric
    yield assert_isinstance, as_tf((B.randn(),))[0], B.TFNumeric


def test_as_torch_and_as_np():
    yield assert_isinstance, as_torch(B.randn()), B.TorchNumeric
    yield assert_isinstance, as_torch((B.randn(),))[0], B.TorchNumeric
    yield assert_isinstance, as_np(as_torch(B.randn())), B.NPNumeric
    yield assert_isinstance, as_np(as_torch((B.randn(),)))[0], B.NPNumeric


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

    with tf.Session() as sess:
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
            tf_grad = sess.run(tf.gradients(f_i(tf_args[i]), tf_args[i])[0])
            approx(numerical_grad, tf_grad, digits)

            # Check PyTorch gradient.
            torch_args = tuple([as_torch(arg, grad=True) for arg in args])
            f_i = create_f_i(i, torch_args)
            f_i(torch_args[i]).backward()
            approx(numerical_grad, torch_args[i].grad, digits)


def test_toeplitz_solve():
    yield check_sensitivity, toeplitz_solve, s_toeplitz_solve, \
          (B.randn(3), B.randn(2), B.randn(3))
    yield check_sensitivity, toeplitz_solve, s_toeplitz_solve, \
          (B.randn(3), B.randn(2), B.randn(3, 4))
    yield check_grad, toeplitz_solve, \
          (B.randn(3), B.randn(2), B.randn(3))
    yield check_grad, toeplitz_solve, \
          (B.randn(3), B.randn(2), B.randn(3, 4))


def test_bvn_cdf():
    pass
