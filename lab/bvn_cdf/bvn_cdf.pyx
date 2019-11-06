# Apparently this future import is needed to fix the NumPy import.
from __future__ import absolute_import

cimport numpy as np
import numpy as np

cimport cython
from cython.parallel import prange

cdef extern from './tvpack.h' nogil:
    extern double bvnd_(double* x, double* y, double* rho);


@cython.boundscheck(False)
@cython.wraparound(False)
def bvn_cdf(np.ndarray[np.float64_t, ndim=1] x,
            np.ndarray[np.float64_t, ndim=1] y,
            np.ndarray[np.float64_t, ndim=1] rho):
    cdef int n = x.shape[0];
    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty([n], dtype=np.float64);

    # Define views for access in the parallel loop.
    cdef np.float64_t [:] x_view = x;
    cdef np.float64_t [:] y_view = y;
    cdef np.float64_t [:] rho_view = rho;
    cdef np.float64_t [:] out_view = out;

    cdef int i;
    cdef double neg_x;
    cdef double neg_y;
    for i in prange(n, nogil=True):
        neg_x = -x_view[i]
        neg_y = -y_view[i]
        out_view[i] = bvnd_(&neg_x, &neg_y, &rho_view[i])

    return out
