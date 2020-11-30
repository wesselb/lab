# Apparently this future import is needed to fix the NumPy import.
from __future__ import absolute_import

cimport numpy as np
import numpy as np

cimport cython
from cython.parallel import prange

cdef extern from "math.h" nogil:
    double log(double x)
    double exp(double x)
    double sqrt(double x)

cdef extern from "./tvpack.h" nogil:
    double phid_(double* x)
    double bvnd_(double* x, double* y, double* rho)


@cython.boundscheck(False)
@cython.wraparound(False)
def bvn_cdf(np.ndarray[np.float64_t, ndim=1] x,
            np.ndarray[np.float64_t, ndim=1] y,
            np.ndarray[np.float64_t, ndim=1] rho):
    cdef int n = x.shape[0]

    # Initialise output.
    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty([n], dtype=np.float64)

    # Define views for access in the parallel loop.
    cdef np.float64_t [:] x_view = x
    cdef np.float64_t [:] y_view = y
    cdef np.float64_t [:] rho_view = rho
    cdef np.float64_t [:] out_view = out

    cdef int i
    cdef double neg_x
    cdef double neg_y

    for i in prange(n, nogil=True):
        neg_x = -x_view[i]
        neg_y = -y_view[i]
        out_view[i] = bvnd_(&neg_x, &neg_y, &rho_view[i])

    return out


cdef double uvn_pdf(double x) nogil:
    cdef double pi = 3.141592653589793
    return exp(-0.5 * x * x) / sqrt(2 * pi)


cdef double bvn_pdf(double x, double y, double rho) nogil:
    cdef double pi = 3.141592653589793
    cdef double determinant = 2 * pi * sqrt(1 - rho * rho)
    cdef double quad_form = (x * x - 2 * rho * x * y + y * y) / \
                     (2 * (1 - rho * rho))
    return exp(-quad_form) / determinant


@cython.boundscheck(False)
@cython.wraparound(False)
def s_bvn_cdf(np.ndarray[np.float64_t, ndim=1] s_out,
              np.ndarray[np.float64_t, ndim=1] out,
              np.ndarray[np.float64_t, ndim=1] x,
              np.ndarray[np.float64_t, ndim=1] y,
              np.ndarray[np.float64_t, ndim=1] rho):
    cdef int n = x.shape[0];

    # Initialise output.
    cdef np.ndarray[np.float64_t, ndim=1] s_x = np.empty([n], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] s_y = np.empty([n], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] s_rho = np.empty([n], dtype=np.float64)

    # Define views for access in the parallel loop.
    cdef np.float64_t [:] s_out_view = s_out
    cdef np.float64_t [:] out_view = out
    cdef np.float64_t [:] x_view = x
    cdef np.float64_t [:] y_view = y
    cdef np.float64_t [:] rho_view = rho

    cdef np.float64_t [:] s_x_view = s_x
    cdef np.float64_t [:] s_y_view = s_y
    cdef np.float64_t [:] s_rho_view = s_rho

    cdef int i
    cdef double q
    cdef double pdf
    cdef double x_normalised
    cdef double y_normalised

    for i in prange(n, nogil=True):
        q = sqrt(1 - rho_view[i] * rho_view[i])
        pdf = bvn_pdf(x_view[i], y_view[i], rho_view[i])
        x_normalised = (x_view[i] - rho_view[i] * y_view[i]) / q
        y_normalised = (y_view[i] - rho_view[i] * x_view[i]) / q

        s_x_view[i] = s_out_view[i] * uvn_pdf(x_view[i]) * phid_(&y_normalised)
        s_y_view[i] = s_out_view[i] * uvn_pdf(y_view[i]) * phid_(&x_normalised)
        s_rho_view[i] = s_out_view[i] * pdf
        out_view[i] = bvn_pdf(x[i], x[i], 0.5)

    return s_x, s_y, s_rho

