# [LAB](http://github.com/wesselb/lab)

[![CI](https://github.com/wesselb/lab/workflows/CI/badge.svg?branch=master)](https://github.com/wesselb/lab/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/lab/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/lab?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/lab)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A generic interface for linear algebra backends: code it once, run it on any 
backend

* [Requirements and Installation](#requirements-and-installation)
* [Basic Usage](#basic-usage)
* [List of Types](#list-of-types)
    - [General](#general)
    - [NumPy](#numpy)
    - [AutoGrad](#autograd)
    - [TensorFlow](#tensorflow)
    - [PyTorch](#pytorch)
    - [JAX](#jax)
* [List of Methods](#list-of-methods)
    - [Constants](#constants)
    - [Generic](#generic)
    - [Linear Algebra](#linear-algebra)
    - [Random](#random)
    - [Shaping](#shaping)
* [Devices](#devices)
* [Lazy Shapes](#lazy-shapes)
* [Control Flow Cache](#control-flow-cache)

## Requirements and Installation

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).
Then simply

```
pip install backends
```

## Basic Usage

The basic use case for the package is to write code that automatically 
determines the backend to use depending on the types of its arguments.

Example:

```python
import lab as B
import lab.autograd    # Load the AutoGrad extension.
import lab.torch       # Load the PyTorch extension.
import lab.tensorflow  # Load the TensorFlow extension.
import lab.jax         # Load the JAX extension.


def objective(matrix):
    outer_product = B.matmul(matrix, matrix, tr_b=True)
    return B.mean(outer_product)
```

The AutoGrad, PyTorch, TensorFlow, and JAX extensions are not loaded automatically to
not enforce a dependency on all three frameworks.
An extension can alternatively be loaded via `import lab.autograd as B`.

Run it with NumPy and AutoGrad:

```python
>>> import autograd.numpy as np

>>> objective(B.randn(np.float64, 2, 2))
0.15772589216756833

>>> grad(objective)(B.randn(np.float64, 2, 2))
array([[ 0.23519042, -1.06282928],
       [ 0.23519042, -1.06282928]])
```

Run it with TensorFlow:
```python
>>> import tensorflow as tf

>>> objective(B.randn(tf.float64, 2, 2))
<tf.Tensor 'Mean:0' shape=() dtype=float64>
```

Run it with PyTorch:
```python
>>> import torch

>>> objective(B.randn(torch.float64, 2, 2))
tensor(1.9557, dtype=torch.float64)
```

Run it with JAX:
```python
>>> import jax

>>> import jax.numpy as jnp

>>> jax.jit(objective)(B.randn(jnp.float32, 2, 2))
DeviceArray(0.3109299, dtype=float32)

>>> jax.jit(jax.grad(objective))(B.randn(jnp.float32, 2, 2))
DeviceArray([[ 0.2525182, -1.26065  ],
             [ 0.2525182, -1.26065  ]], dtype=float32)
```

## List of Types
This section lists all available types, which can be used to check types of 
objects or extend functions.

Example:

```python
>>> import lab as B

>>> from plum import List, Tuple

>>> import numpy as np

>>> isinstance([1., np.array([1., 2.])], List(B.NPNumeric))
True

>>> isinstance([1., np.array([1., 2.])], List(B.TFNumeric))
False

>>> import tensorflow as tf

>>> import lab.tensorflow

>>> isinstance((tf.constant(1.), tf.ones(5)), Tuple(B.TFNumeric))
True
```

### General

```
Int          # Integers
Float        # Floating-point numbers
Bool         # Booleans
Number       # Numbers
Numeric      # Numerical objects, including booleans
DType        # Data type
Framework    # Anything accepted by supported frameworks
```

### NumPy

```
NPNumeric
NPDType
 
NP           # Anything NumPy
```

### AutoGrad

```
AGNumeric
AGDType
 
AG           # Anything AutoGrad
```

### TensorFlow

```
TFNumeric
TFDType
 
TF           # Anything TensorFlow
```


### PyTorch

```
TorchNumeric
TorchDType
 
Torch        # Anything PyTorch
```


### JAX

```
JAXNumeric
JAXDType
 
JAX          # Anything JAX
```


## List of Methods
This section lists all available constants and methods.

*
    Arguments *must* be given as arguments and keyword arguments *must* be 
    given as keyword arguments.
    For example, `sum(tensor, axis=1)` is valid, but `sum(tensor, 1)` is not.
    
* The names of arguments are indicative of their function:
    - `a`, `b`, and `c` indicate general tensors.
    -
        `dtype` indicates a data type. E.g, `np.float32` or `tf.float64`; and
        `rand(np.float32)` creates a NumPy random number, whereas
        `rand(tf.float64)` creates a TensorFlow random number.
        Data types are always given as the first argument.
    -
        `shape` indicates a shape.
        The dimensions of a shape are always given as separate arguments to 
        the function.
        E.g., `reshape(tensor, 2, 2)` is valid, but `reshape(tensor, (2, 2))`
        is not.
    -
        `axis` indicates an axis over which the function may perform its action.
        An axis is always given as a keyword argument.
    -
        `ref` indicates a *reference tensor* from which properties, like its
        shape and data type, will be used. E.g., `zeros(tensor)` creates a 
        tensor full of zeros of the same shape and data type as `tensor`.
    
See the documentation for more detailed descriptions of each function. 

### Special Variables
```
default_dtype  # Default data type.
epsilon        # Magnitude of diagonal to regularise matrices with.
```

### Constants
```
nan
pi
log_2_pi
```

### Generic
```
isnan(a)

device(a)
device(name)
move_to_active_device(a)

zeros(dtype, *shape)
zeros(*shape)
zeros(ref)

ones(dtype, *shape)
ones(*shape)
ones(ref)

one(dtype)
one(ref)

zero(dtype)
zero(ref)

eye(dtype, *shape)
eye(*shape)
eye(ref)

linspace(dtype, a, b, num)
linspace(a, b, num)

range(dtype, start, stop, step)
range(dtype, stop)
range(dtype, start, stop)
range(start, stop, step)
range(start, stop)
range(stop)

cast(dtype, a)

identity(a)
negative(a)
abs(a)
sign(a)
sqrt(a)
exp(a)
log(a)
sin(a)
cos(a)
tan(a)
tanh(a)
erf(a)
sigmoid(a)
softplus(a)
relu(a)

add(a, b)
subtract(a, b)
multiply(a, b)
divide(a, b)
power(a, b)
minimum(a, b)
maximum(a, b)
leaky_relu(a, alpha)

min(a, axis=None)
max(a, axis=None)
sum(a, axis=None)
mean(a, axis=None)
std(a, axis=None)
logsumexp(a, axis=None)

all(a, axis=None)
any(a, axis=None)

lt(a, b)
le(a, b)
gt(a, b)
ge(a, b)

bvn_cdf(a, b, c)

cond(condition, f_true, f_false, xs**)
scan(f, xs, *init_state)

sort(a, axis=-1, descending=False)
argsort(a, axis=-1, descending=False)

to_numpy(a)
```

### Linear Algebra
```
transpose(a, perm=None) (alias: t, T)
matmul(a, b, tr_a=False, tr_b=False) (alias: mm, dot)
trace(a, axis1=0, axis2=1)
kron(a, b)
svd(a, compute_uv=True)
solve(a, b)
inv(a)
det(a) 
logdet(a) 
expm(a)
logm(a)
cholesky(a) (alias: chol)

cholesky_solve(a, b)  (alias: cholsolve)
triangular_solve(a, b, lower_a=True) (alias: trisolve)
toeplitz_solve(a, b, c) (alias: toepsolve)
toeplitz_solve(a, c)

outer(a, b)
reg(a, diag=None, clip=True)

pw_dists2(a, b)
pw_dists2(a)
pw_dists(a, b)
pw_dists(a)

ew_dists2(a, b)
ew_dists2(a)
ew_dists(a, b)
ew_dists(a)

pw_sums2(a, b)
pw_sums2(a)
pw_sums(a, b)
pw_sums(a)

ew_sums2(a, b)
ew_sums2(a)
ew_sums(a, b)
ew_sums(a)
```
### Random
```
set_random_seed(seed) 

rand(dtype, *shape)
rand(*shape)
rand(ref)

randn(dtype, *shape)
randn(*shape)
randn(ref)

choice(a, n)
choice(a)
```

### Shaping
```
shape(a)
rank(a)
length(a) (alias: size)
isscalar(a)
expand_dims(a, axis=0)
squeeze(a)
uprank(a, rank=2)

diag(a)
diag_extract(a)
diag_construct(a)
flatten(a)
vec_to_tril(a, offset=0)
tril_to_vec(a, offset=0)
stack(*elements, axis=0)
unstack(a, axis=0)
reshape(a, *shape)
concat(*elements, axis=0)
concat2d(*rows)
tile(a, *repeats)
take(a, indices, axis=0)
```

## Devices
You can get the device of a tensor with `B.device(a)`.
The device of a tensor is always returned as a string.

You can execute a computation on a device by entering `B.device(name)` as a context:

```python
with B.device("gpu:0"):
    a = B.randn(tf.float32, 2, 2)
    b = B.randn(tf.float32, 2, 2)
    c = a @ b
```

## Lazy Shapes
If a function is evaluated abstractly, then elements of the shape of a tensor, e.g.
`B.shape(a)[0]`, may also be tensors, which can break dispatch.
By entering `B.lazy_shapes`, shapes and elements of shapes will be wrapped in a custom 
type to fix this issue.

```python
with B.lazy_shapes:
    a = B.eye(2)
    print(type(B.shape(a)))
    # <class 'lab.shape.Shape'>
    print(type(B.shape(a)[0]))
    # <class 'lab.shape.Dimension'>
```

## Control Flow Cache
Coming soon!