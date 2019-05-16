# [LAB](http://github.com/wesselb/lab)

[![Build](https://travis-ci.org/wesselb/lab.svg?branch=master)](https://travis-ci.org/wesselb/lab)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/lab/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/lab?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/lab)

A generic interface for linear algebra backends: code it once, run it on any 
backend

* [Installation](#installation)
* [Basic Usage](#basic-usage)
* [List of Types](#list-of-types)
    - [General](#general)
    - [NumPy](#numpy)
    - [TensorFlow](#tensorflow)
    - [PyTorch](#pytorch)
* [List of Methods](#list-of-methods)
    - [Constants](#constants)
    - [Generic](#generic)
    - [Linear Algebra](#linear-algebra)
    - [Random](#random)
    - [Shaping](#shaping)

## Installation
The package is tested for Python 2.7 and Python 3.6, which are the versions 
recommended to use.
To install the package, use the following sequence of commands:

```
git clone https://github.com/wesselb/lab
cd lab
make install
```

## Basic Usage

The basic use case for the package is to write code that automatically 
determines the backend to use depending on the types of its arguments.


Example:

```python
import lab as B
import lab.torch       # Load the PyTorch extension.
import lab.tensorflow  # Load the TensorFlow extension.

def objective(matrix):
    outer_product = B.matmul(matrix, matrix, tr_b=True)
    return B.mean(outer_product)
```

By default, the PyTorch and TensorFlow extensions are not loaded to save 
startup time. Alternatively, one can directly `import lab.torch as B` or
`import lab.tensorflow as B`.

Run it with NumPy and AutoGrad:

```python
>>> import autograd.numpy as np

>>> objective(B.randn(np.float64, 2, 2))
0.15772589216756833
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
Dimension    # Dimensions of shapes
DType        # Data type
Framework    # Anything accepted by supported frameworks
```


### NumPy

```
NPNumeric
NPDimension
NPDType
 
NP           # Anything NumPy
```

### TensorFlow

```
TFNumeric
TFDimension
TFDType
 
TF           # Anything TensorFlow
```


### PyTorch

```
TorchNumeric
TorchDimension
TorchDType
 
Torch        # Anything PyTorch
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
        The dimensions of a shape as always given as separate arguments to 
        the function.
        E.g., `reshape(tensor, 2, 2)` is valid, but `reshape(tensor, (2, 2))`
        is not.
    -
        `axis` indicates an axis over which the function may perform its action.
        An axis is always given as a keyword argument.
    -
        `ref` indicates a *reference tensor* from which a property (like its
        data type) will be inferred. E.g., `zeros(tensor)` creates a tensor
        full or zeros of the same shape and data type as `tensor`.
    
See the documentation for more detailed descriptions of each function. 

### Constants
```
nan
pi
log_2_pi

isnan(a)
```

### Generic
```
zeros(dtype, *shape)
zeros(*shape)
zeros(ref)

ones(dtype, *shape)
ones(*shape)
ones(ref)

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
abs(a)
sign(a)
sqrt(a)
exp(a)
log(a)
sin(a)
cos(a)
tan(a)
sigmoid(a)
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
cholesky(a)
cholesky_solve(a, b)
trisolve(a, b, lower_a=True)
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
rand(dtype)
rand()

randn(dtype, *shape)
randn(*shape)
randn(dtype)
randn()
```

### Shaping
```
shape(a)
shape_int(a)
rank(a)
length(a) (alias: size)
isscalar(a)
expand_dims(a, axis=0)
squeeze(a)
uprank(a)

diag(a)
flatten(a)
vec_to_tril(a)
tril_to_vec(a)
stack(*elements, axis=0)
unstack(a, axis=0)
reshape(a, *shape)
concat(*elements, axis=0)
concat2d(*rows)
take(a, indices, axis=0)
```