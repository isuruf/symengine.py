cimport symengine
from libcpp.vector cimport vector
from libcpp cimport bool as cppbool
from libcpp.string cimport string
from .core cimport Basic
from cython.operator cimport dereference as deref

from symengine_wrapper import _sympify
from operator import mul
from functools import reduce
import os
import warnings

try:
    import numpy as np
    # Lambdify requires NumPy (since b713a61, see gh-112)
    have_numpy = True
except ImportError:
    have_numpy = False

include "config.pxi"

cdef class _Lambdify(object):
    cdef size_t args_size, tot_out_size
    cdef list out_shapes
    cdef readonly bint real
    cdef readonly size_t n_exprs
    cdef public str order
    cdef vector[int] accum_out_sizes
    cdef object numpy_dtype

    def __init__(self, args, *exprs, cppbool real=True, order='C', cppbool cse=False, cppbool _load=False):
        cdef:
            Basic e_
            size_t ri, ci, nr, nc
            symengine.MatrixBase *mtx
            symengine.rcp_const_basic b_
            symengine.vec_basic args_, outs_
            vector[int] out_sizes

        if _load:
            self.args_size, self.tot_out_size, self.out_shapes, self.real, \
                self.n_exprs, self.order, self.accum_out_sizes, self.numpy_dtype, \
                llvm_function = args
            self._load(llvm_function)
            return

        args = np.asanyarray(args)
        self.args_size = args.size
        exprs = tuple(np.asanyarray(expr) for expr in exprs)
        self.out_shapes = [expr.shape for expr in exprs]
        self.n_exprs = len(exprs)
        self.real = real
        self.order = order
        self.numpy_dtype = np.float64 if self.real else np.complex128
        if self.args_size == 0:
            raise NotImplementedError("Support for zero arguments not yet supported")
        self.tot_out_size = 0
        for idx, shape in enumerate(self.out_shapes):
            out_sizes.push_back(reduce(mul, shape or (1,)))
            self.tot_out_size += out_sizes[idx]
        for i in range(self.n_exprs + 1):
            self.accum_out_sizes.push_back(0)
            for j in range(i):
                self.accum_out_sizes[i] += out_sizes[j]

        for arg in np.ravel(args, order=self.order):
            e_ = _sympify(arg)
            args_.push_back(e_.thisptr)

        for curr_expr in exprs:
            if curr_expr.ndim == 0:
                e_ = _sympify(curr_expr.item())
                outs_.push_back(e_.thisptr)
            else:
                for e in np.ravel(curr_expr, order=self.order):
                    e_ = _sympify(e)
                    outs_.push_back(e_.thisptr)
        self._init(args_, outs_, cse)

    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse):
        raise ValueError("Not supported")

    cdef _load(self, const string &s):
        raise ValueError("Not supported")

    cpdef unsafe_real(self,
                      double[::1] inp, double[::1] out,
                      int inp_offset=0, int out_offset=0):
        raise ValueError("Not supported")

    cpdef unsafe_complex(self, double complex[::1] inp, double complex[::1] out,
                         int inp_offset=0, int out_offset=0):
        raise ValueError("Not supported")

    cpdef eval_real(self, inp, out):
        if inp.size != self.args_size:
            raise ValueError("Size of inp incompatible with number of args.")
        if out.size != self.tot_out_size:
            raise ValueError("Size of out incompatible with number of exprs.")
        self.unsafe_real(inp, out)

    cpdef eval_complex(self, inp, out):
        if inp.size != self.args_size:
            raise ValueError("Size of inp incompatible with number of args.")
        if out.size != self.tot_out_size:
            raise ValueError("Size of out incompatible with number of exprs.")
        self.unsafe_complex(inp, out)

    def __call__(self, *args, out=None):
        """
        Parameters
        ----------
        inp: array_like
            last dimension must be equal to number of arguments.
        out: array_like or None (default)
            Allows for low-overhead use (output argument, must be contiguous).
            If ``None``: an output container will be allocated (NumPy ndarray).
            If ``len(exprs) > 0`` output is found in the corresponding
            order.

        Returns
        -------
        If ``len(exprs) == 1``: ``numpy.ndarray``, otherwise a tuple of such.

        """
        cdef:
            size_t idx, new_tot_out_size, nbroadcast = 1
            long inp_size
            tuple inp_shape
            double[::1] real_out, real_inp
            double complex[::1] cmplx_out, cmplx_inp
        if self.order not in ('C', 'F'):
            raise NotImplementedError("Only C & F order supported for now.")

        if len(args) == 1:
            args = args[0]

        try:
            inp = np.asanyarray(args, dtype=self.numpy_dtype)
        except TypeError:
            inp = np.fromiter(args, dtype=self.numpy_dtype)

        if self.real:
            real_inp = np.ascontiguousarray(inp.ravel(order=self.order))
        else:
            cmplx_inp = np.ascontiguousarray(inp.ravel(order=self.order))

        if inp.size < self.args_size or inp.size % self.args_size != 0:
            raise ValueError("Broadcasting failed (input/arg size mismatch)")
        nbroadcast = inp.size // self.args_size

        if inp.ndim > 1:
            if self.args_size > 1:
                if self.order == 'C':
                    if inp.shape[inp.ndim-1] != self.args_size:
                        raise ValueError(("C order implies last dim (%d) == len(args)"
                                          " (%d)") % (inp.shape[inp.ndim-1], self.args_size))
                    extra_dim = inp.shape[:inp.ndim-1]
                elif self.order == 'F':
                    if inp.shape[0] != self.args_size:
                        raise ValueError("F order implies first dim (%d) == len(args) (%d)"
                                         % (inp.shape[0], self.args_size))
                    extra_dim = inp.shape[1:]
            else:
                extra_dim = inp.shape
        else:
            if nbroadcast > 1 and inp.ndim == 1:
                extra_dim = (nbroadcast,)  # special case
            else:
                extra_dim = ()
        extra_left = extra_dim if self.order == 'C' else ()
        extra_right = () if self.order == 'C' else extra_dim
        new_out_shapes = [extra_left + out_shape + extra_right
                          for out_shape in self.out_shapes]

        new_tot_out_size = nbroadcast * self.tot_out_size
        if out is None:
            out = np.empty(new_tot_out_size, dtype=self.numpy_dtype, order=self.order)
        else:
            if out.size < new_tot_out_size:
                raise ValueError("Incompatible size of output argument")
            if out.ndim > 1:
                if len(self.out_shapes) > 1:
                    raise ValueError("output array with ndim > 1 assumes one output")
                out_shape, = self.out_shapes
                if self.order == 'C':
                    if not out.flags['C_CONTIGUOUS']:
                        raise ValueError("Output argument needs to be C-contiguous")
                    if out.shape[-len(out_shape):] != tuple(out_shape):
                        raise ValueError("shape mismatch for output array")
                elif self.order == 'F':
                    if not out.flags['F_CONTIGUOUS']:
                        raise ValueError("Output argument needs to be F-contiguous")
                    if out.shape[:len(out_shape)] != tuple(out_shape):
                        raise ValueError("shape mismatch for output array")
            else:
                if not out.flags['F_CONTIGUOUS']:  # or C_CONTIGUOUS (ndim <= 1)
                    raise ValueError("Output array need to be contiguous")
            if not out.flags['WRITEABLE']:
                raise ValueError("Output argument needs to be writeable")
            out = out.ravel(order=self.order)

        if self.real:
            real_out = out
        else:
            cmplx_out = out

        if self.real:
            for idx in range(nbroadcast):
                self.unsafe_real(real_inp, real_out,
                                 idx*self.args_size, idx*self.tot_out_size)
        else:
            for idx in range(nbroadcast):
                self.unsafe_complex(cmplx_inp, cmplx_out,
                                    idx*self.args_size, idx*self.tot_out_size)

        if self.order == 'C':
            out = out.reshape((nbroadcast, self.tot_out_size), order='C')
            result = [
                out[:, self.accum_out_sizes[idx]:self.accum_out_sizes[idx+1]].reshape(
                    new_out_shapes[idx], order='C') for idx in range(self.n_exprs)
            ]
        elif self.order == 'F':
            out = out.reshape((self.tot_out_size, nbroadcast), order='F')
            result = [
                out[self.accum_out_sizes[idx]:self.accum_out_sizes[idx+1], :].reshape(
                    new_out_shapes[idx], order='F') for idx in range(self.n_exprs)
            ]
        if self.n_exprs == 1:
            return result[0]
        else:
            return result


cdef double _scipy_callback_lambda_real(int n, double *x, void *user_data) nogil:
    cdef symengine.LambdaRealDoubleVisitor* lamb = <symengine.LambdaRealDoubleVisitor *>user_data
    cdef double result
    deref(lamb).call(&result, x)
    return result


IF HAVE_SYMENGINE_LLVM:
    cdef double _scipy_callback_llvm_real(int n, double *x, void *user_data) nogil:
        cdef symengine.LLVMDoubleVisitor* lamb = <symengine.LLVMDoubleVisitor *>user_data
        cdef double result
        deref(lamb).call(&result, x)
        return result


def create_low_level_callable(lambdify, *args):
    from scipy import LowLevelCallable
    class LambdifyLowLevelCallable(LowLevelCallable):
        def __init__(self, lambdify, *args):
            self.lambdify = lambdify
        def __new__(cls, value, *args, **kwargs):
            return super(LambdifyLowLevelCallable, cls).__new__(cls, *args)
    return LambdifyLowLevelCallable(lambdify, *args)


cdef class LambdaDouble(_Lambdify):

    cdef vector[symengine.LambdaRealDoubleVisitor] lambda_double
    cdef vector[symengine.LambdaComplexDoubleVisitor] lambda_double_complex

    cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse):
        if self.real:
            self.lambda_double.resize(1)
            self.lambda_double[0].init(args_, outs_, cse)
        else:
            self.lambda_double_complex.resize(1)
            self.lambda_double_complex[0].init(args_, outs_, cse)

    cpdef unsafe_real(self, double[::1] inp, double[::1] out, int inp_offset=0, int out_offset=0):
        self.lambda_double[0].call(&out[out_offset], &inp[inp_offset])

    cpdef unsafe_complex(self, double complex[::1] inp, double complex[::1] out, int inp_offset=0, int out_offset=0):
        self.lambda_double_complex[0].call(&out[out_offset], &inp[inp_offset])

    cpdef as_scipy_low_level_callable(self):
        from ctypes import c_double, c_void_p, c_int, cast, POINTER, CFUNCTYPE
        if not self.real:
            raise RuntimeError("Lambda function has to be real")
        if self.tot_out_size > 1:
            raise RuntimeError("SciPy LowLevelCallable supports only functions with 1 output")
        addr1 = cast(<size_t>&_scipy_callback_lambda_real,
                        CFUNCTYPE(c_double, c_int, POINTER(c_double), c_void_p))
        addr2 = cast(<size_t>&self.lambda_double[0], c_void_p)
        return create_low_level_callable(self, addr1, addr2)


IF HAVE_SYMENGINE_LLVM:
    cdef class LLVMDouble(_Lambdify):

        cdef vector[symengine.LLVMDoubleVisitor] lambda_double

        cdef _init(self, symengine.vec_basic& args_, symengine.vec_basic& outs_, cppbool cse):
            self.lambda_double.resize(1)
            self.lambda_double[0].init(args_, outs_, cse)

        cdef _load(self, const string &s):
            self.lambda_double.resize(1)
            self.lambda_double[0].loads(s)

        def __reduce__(self):
            """
            Interface for pickle. Note that the resulting object is platform dependent.
            """
            cdef bytes s = self.lambda_double[0].dumps()
            return llvm_loading_func, (self.args_size, self.tot_out_size, self.out_shapes, self.real, \
                self.n_exprs, self.order, self.accum_out_sizes, self.numpy_dtype, s)

        cpdef unsafe_real(self, double[::1] inp, double[::1] out, int inp_offset=0, int out_offset=0):
            self.lambda_double[0].call(&out[out_offset], &inp[inp_offset])

        cpdef as_scipy_low_level_callable(self):
            from ctypes import c_double, c_void_p, c_int, cast, POINTER, CFUNCTYPE
            if not self.real:
                raise RuntimeError("Lambda function has to be real")
            if self.tot_out_size > 1:
                raise RuntimeError("SciPy LowLevelCallable supports only functions with 1 output")
            addr1 = cast(<size_t>&_scipy_callback_lambda_real,
                            CFUNCTYPE(c_double, c_int, POINTER(c_double), c_void_p))
            addr2 = cast(<size_t>&self.lambda_double[0], c_void_p)
            return create_low_level_callable(self, addr1, addr2)

    def llvm_loading_func(*args):
        return LLVMDouble(args, _load=True)

def Lambdify(args, *exprs, cppbool real=True, backend=None, order='C', as_scipy=False, cse=False):
    """
    Lambdify instances are callbacks that numerically evaluate their symbolic
    expressions from user provided input (real or complex) into (possibly user
    provided) output buffers (real or complex). Multidimensional data are
    processed in their most cache-friendly way (i.e. "ravelled").

    Parameters
    ----------
    args: iterable of Symbols
    \*exprs: array_like of expressions
        the shape of exprs is preserved
    real : bool
        Whether datatype is ``double`` (``double complex`` otherwise).
    backend : str
        'llvm' or 'lambda'. When ``None`` the environment variable
        'SYMENGINE_LAMBDIFY_BACKEND' is used (taken as 'lambda' if unset).
    order : 'C' or 'F'
        C- or Fortran-contiguous memory layout. Note that this affects
        broadcasting: e.g. a (m, n) matrix taking 3 arguments and given a
        (k, l, 3) (C-contiguous) input will give a (k, l, m, n) shaped output,
        whereas a (3, k, l) (C-contiguous) input will give a (m, n, k, l) shaped
        output. If ``None`` order is taken as ``self.order`` (from initialization).
    as_scipy : bool
        return a SciPy LowLevelCallable which can be used in SciPy's integrate
        methods
    cse : bool
        Run Common Subexpression Elimination on the output before generating
        the callback.

    Returns
    -------
    callback instance with signature f(inp, out=None)

    Examples
    --------
    >>> from symengine import var, Lambdify
    >>> var('x y z')
    >>> f = Lambdify([x, y, z], [x+y+z, x*y*z])
    >>> f([2, 3, 4])
    [ 9., 24.]
    >>> out = np.array(2)
    >>> f(x, out); out
    [ 9., 24.]

    """
    if backend is None:
        backend = os.getenv('SYMENGINE_LAMBDIFY_BACKEND', "lambda")
    if backend == "llvm":
        IF HAVE_SYMENGINE_LLVM:
            ret = LLVMDouble(args, *exprs, real=real, order=order, cse=cse)
            if as_scipy:
                return ret.as_scipy_low_level_callable()
            return ret
        ELSE:
            raise ValueError("""llvm backend is chosen, but symengine is not compiled
                                with llvm support.""")
    elif backend == "lambda":
        pass
    else:
        warnings.warn("Unknown SymEngine backend: %s\nUsing backend='lambda'" % backend)
    ret = LambdaDouble(args, *exprs, real=real, order=order, cse=cse)
    if as_scipy:
        return ret.as_scipy_low_level_callable()
    return ret


def LambdifyCSE(args, *exprs, order='C', **kwargs):
    """ Analogous with Lambdify but performs common subexpression elimination.
    """
    warnings.warn("LambdifyCSE is deprecated. Use Lambdify(..., cse=True)", DeprecationWarning)
    return Lambdify(args, *exprs, cse=True, order=order, **kwargs)


