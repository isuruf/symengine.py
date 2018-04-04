from cython.operator cimport dereference as deref
cimport symengine
from .core cimport Basic, _DictBasic, c2py
from .core import _sympify, sympify, get_dict
include "config.pxi"

cdef class MatrixBase:

    @property
    def is_Matrix(self):
        return True

    def __richcmp__(a, b, int op):
        A = _sympify(a, False)
        B = _sympify(b, False)
        if not (isinstance(A, MatrixBase) and isinstance(B, MatrixBase)):
            if (op == 2):
                return False
            elif (op == 3):
                return True
            return NotImplemented
        return A._richcmp_(B, op)

    def _richcmp_(MatrixBase A, MatrixBase B, int op):
        if (op == 2):
            return deref(A.thisptr).eq(deref(B.thisptr))
        elif (op == 3):
            return not deref(A.thisptr).eq(deref(B.thisptr))
        else:
            return NotImplemented

    def __dealloc__(self):
        del self.thisptr

    def _symbolic_(self, ring):
        return ring(self._sage_())

    # TODO: fix this
    def __hash__(self):
        return 0


class MatrixError(Exception):
    pass


class ShapeError(ValueError, MatrixError):
    """Wrong matrix shape"""
    pass


class NonSquareMatrixError(ShapeError):
    pass

cdef class DenseMatrixBase(MatrixBase):
    """
    Represents a two-dimensional dense matrix.

    Examples
    ========

    Empty matrix:

    >>> DenseMatrix(3, 2)

    2D Matrix:

    >>> DenseMatrix(3, 2, [1, 2, 3, 4, 5, 6])
    [1, 2]
    [3, 4]
    [5, 6]

    >>> DenseMatrix([[1, 2], [3, 4], [5, 6]])
    [1, 2]
    [3, 4]
    [5, 6]

    """

    def __cinit__(self, row=None, col=None, v=None):
        if row is None:
            self.thisptr = new symengine.DenseMatrix(0, 0)
            return
        if v is None and col is not None:
            self.thisptr = new symengine.DenseMatrix(row, col)
            return
        if col is None:
            v = row
            row = 0
        cdef symengine.vec_basic v_
        cdef DenseMatrixBase A
        cdef Basic e_
        #TODO: Add a constructor to DenseMatrix in C++
        if (isinstance(v, DenseMatrixBase)):
            matrix_to_vec(v, v_)
            if col is None:
                row = v.nrows()
                col = v.ncols()
            self.thisptr = new symengine.DenseMatrix(row, col, v_)
            return
        for e in v:
            f = sympify(e)
            if isinstance(f, DenseMatrixBase):
                matrix_to_vec(f, v_)
                if col is None:
                    row = row + f.nrows()
                continue
            try:
                for e_ in f:
                    v_.push_back(e_.thisptr)
                if col is None:
                    row = row + 1
            except TypeError:
                e_ = f
                v_.push_back(e_.thisptr)
                if col is None:
                    row = row + 1
        if (row == 0):
            if (v_.size() != 0):
                self.thisptr = new symengine.DenseMatrix(0, 0, v_)
                raise ValueError("sizes don't match.")
            else:
                self.thisptr = new symengine.DenseMatrix(0, 0, v_)
        else:
            self.thisptr = new symengine.DenseMatrix(row, v_.size() / row, v_)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return deref(self.thisptr).__str__().decode("utf-8")

    def __add__(a, b):
        a = _sympify(a, False)
        b = _sympify(b, False)
        if not isinstance(a, MatrixBase) or not isinstance(b, MatrixBase):
            return NotImplemented
        cdef MatrixBase a_ = a
        cdef MatrixBase b_ = b
        if (a_.shape == (0, 0)):
            return b_
        if (b_.shape == (0, 0)):
            return a_
        if (a_.shape != b_.shape):
            raise ShapeError("Invalid shapes for matrix addition. Got %s %s" % (a_.shape, b_.shape))
        return a_.add_matrix(b_)

    def __mul__(a, b):
        a = _sympify(a, False)
        b = _sympify(b, False)
        if isinstance(a, MatrixBase):
            if isinstance(b, MatrixBase):
                if (a.ncols() != b.nrows()):
                    raise ShapeError("Invalid shapes for matrix multiplication. Got %s %s" % (a.shape, b.shape))
                return a.mul_matrix(b)
            elif isinstance(b, Basic):
                return a.mul_scalar(b)
            else:
                return NotImplemented
        elif isinstance(a, Basic):
            return b.mul_scalar(a)
        else:
            return NotImplemented

    def __div__(a, b):
        return div_matrices(a, b)

    def __truediv__(a, b):
        return div_matrices(a, b)

    def __sub__(a, b):
        a = _sympify(a, False)
        b = _sympify(b, False)
        if not isinstance(a, MatrixBase) or not isinstance(b, MatrixBase):
            return NotImplemented
        cdef MatrixBase a_ = a
        cdef MatrixBase b_ = b
        if (a_.shape != b_.shape):
            raise ShapeError("Invalid shapes for matrix subtraction. Got %s %s" % (a.shape, b.shape))
        return a_.add_matrix(-b_)

    def __neg__(self):
        return self.mul_scalar(-1)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if (self.ncols() == 0 or self.nrows() == 0):
                return []
            return [self.get(i // self.ncols(), i % self.ncols()) for i in range(*item.indices(len(self)))]
        elif isinstance(item, int):
            return self.get(item // self.ncols(), item % self.ncols())
        elif isinstance(item, tuple):
            if isinstance(item[0], int) and isinstance(item[1], int):
                return self.get(item[0], item[1])
            else:
                s = [0, 0, 0, 0, 0, 0]
                for i in (0, 1):
                    if isinstance(item[i], slice):
                        s[i], s[i+2], s[i+4] = item[i].indices(self.nrows() if i == 0 else self.ncols())
                    else:
                        s[i], s[i+2], s[i+4] = item[i], item[i] + 1, 1
                if (s[0] < 0 or s[0] > self.rows or s[0] >= s[2] or s[2] < 0 or s[2] > self.rows):
                    raise IndexError
                if (s[1] < 0 or s[1] > self.cols or s[1] >= s[3] or s[3] < 0 or s[3] > self.cols):
                    raise IndexError
                return self._submatrix(*s)
        else:
            raise NotImplementedError

    def __setitem__(self, key, value):
        cdef unsigned k, l
        if isinstance(key, int):
            self.set(key // self.ncols(), key % self.ncols(), value)
        elif isinstance(key, slice):
            k = 0
            for i in range(*key.indices(len(self))):
                self.set(i // self.ncols(), i % self.ncols(), value[k])
                k = k + 1
        elif isinstance(key, tuple):
            if isinstance(key[0], int):
                if isinstance(key[1], int):
                    self.set(key[0], key[1], value)
                else:
                    k = 0
                    for i in range(*key[1].indices(self.cols)):
                        self.set(key[0], i, value[k])
                        k = k + 1
            else:
                if isinstance(key[1], int):
                    k = 0
                    for i in range(*key[0].indices(self.rows)):
                        self.set(i, key[1], value[k])
                        k = k + 1
                else:
                    k = 0
                    for i in range(*key[0].indices(self.rows)):
                        l = 0
                        for j in range(*key[1].indices(self.cols)):
                            try:
                                self.set(i, j, value[k, l])
                            except TypeError:
                                self.set(i, j, value[k][l])
                            l = l + 1
                        k = k + 1
        else:
            raise NotImplementedError

    def row_join(self, rhs):
        cdef DenseMatrixBase o = sympify(rhs)
        if self.rows != o.rows:
            raise ShapeError("`self` and `rhs` must have the same number of rows.")
        cdef DenseMatrixBase d = self.__class__(self)
        deref(symengine.static_cast_DenseMatrix(d.thisptr)).row_join(deref(symengine.static_cast_DenseMatrix(o.thisptr)))
        return d

    def col_join(self, bott):
        cdef DenseMatrixBase o = sympify(bott)
        if self.cols != o.cols:
            raise ShapeError("`self` and `rhs` must have the same number of columns.")
        cdef DenseMatrixBase d = self.__class__(self)
        deref(symengine.static_cast_DenseMatrix(d.thisptr)).col_join(deref(symengine.static_cast_DenseMatrix(o.thisptr)))
        return d

    def row_insert(self, pos, bott):
        cdef DenseMatrixBase o = sympify(bott)
        if pos < 0:
            pos = self.rows + pos
        if pos < 0:
            pos = 0
        elif pos > self.rows:
            pos = self.rows
        if self.cols != o.cols:
            raise ShapeError("`self` and `other` must have the same number of columns.")
        cdef DenseMatrixBase d = self.__class__(self)
        deref(symengine.static_cast_DenseMatrix(d.thisptr)).row_insert(deref(symengine.static_cast_DenseMatrix(o.thisptr)), pos)
        return d

    def col_insert(self, pos, bott):
        cdef DenseMatrixBase o = sympify(bott)
        if pos < 0:
            pos = self.cols + pos
        if pos < 0:
            pos = 0
        elif pos > self.cols:
            pos = self.cols
        if self.rows != o.rows:
            raise ShapeError("`self` and `other` must have the same number of rows.")
        cdef DenseMatrixBase d = self.__class__(self)
        deref(symengine.static_cast_DenseMatrix(d.thisptr)).col_insert(deref(symengine.static_cast_DenseMatrix(o.thisptr)), pos)
        return d

    def dot(self, b):
        cdef DenseMatrixBase o = sympify(b)
        cdef DenseMatrixBase result = self.__class__(self.rows, self.cols)
        symengine.dot(deref(symengine.static_cast_DenseMatrix(self.thisptr)), deref(symengine.static_cast_DenseMatrix(o.thisptr)), deref(symengine.static_cast_DenseMatrix(result.thisptr)))
        if len(result) == 1:
            return result[0, 0]
        else:
            return result

    def cross(self, b):
        cdef DenseMatrixBase o = sympify(b)
        if self.cols * self.rows != 3 or o.cols * o.rows != 3:
            raise ShapeError("Dimensions incorrect for cross product: %s x %s" %
                             ((self.rows, self.cols), (b.rows, b.cols)))
        cdef DenseMatrixBase result = self.__class__(self.rows, self.cols)
        symengine.cross(deref(symengine.static_cast_DenseMatrix(self.thisptr)), deref(symengine.static_cast_DenseMatrix(o.thisptr)), deref(symengine.static_cast_DenseMatrix(result.thisptr)))
        return result

    @property
    def rows(self):
        return self.nrows()

    @property
    def cols(self):
        return self.ncols()

    @property
    def is_square(self):
        return self.rows == self.cols

    def nrows(self):
        return deref(self.thisptr).nrows()

    def ncols(self):
        return deref(self.thisptr).ncols()

    def __len__(self):
        return self.nrows() * self.ncols()

    property shape:
        def __get__(self):
            return (self.nrows(), self.ncols())

    property size:
        def __get__(self):
            return self.nrows()*self.ncols()

    def ravel(self, order='C'):
        if order == 'C':
            return [self._get(i, j) for i in range(self.nrows())
                    for j in range(self.ncols())]
        elif order == 'F':
            return [self._get(i, j) for j in range(self.ncols())
                    for i in range(self.nrows())]
        else:
            raise NotImplementedError("Unknown order '%s'" % order)

    def reshape(self, rows, cols):
        if len(self) != rows*cols:
            raise ValueError("Invalid reshape parameters %d %d" % (rows, cols))
        cdef DenseMatrixBase r = self.__class__(self)
        deref(symengine.static_cast_DenseMatrix(r.thisptr)).resize(rows, cols)
        return r

    def _get_index(self, i, j):
        nr = self.nrows()
        nc = self.ncols()
        if i < 0:
            i += nr
        if j < 0:
            j += nc
        if i < 0 or i >= nr:
            raise IndexError("Row index out of bounds: %d" % i)
        if j < 0 or j >= nc:
            raise IndexError("Column index out of bounds: %d" % j)
        return i, j

    def get(self, i, j):
        i, j = self._get_index(i, j)
        return self._get(i, j)

    def _get(self, i, j):
        # No error checking is done
        return c2py(deref(self.thisptr).get(i, j))

    def col(self, j):
        return self[:, j]

    def row(self, i):
        return self[i, :]

    def set(self, i, j, e):
        i, j = self._get_index(i, j)
        return self._set(i, j, e)

    def _set(self, i, j, e):
        # No error checking is done
        cdef Basic e_ = sympify(e)
        if e_ is not None:
            deref(self.thisptr).set(i, j, e_.thisptr)

    def det(self):
        if self.nrows() != self.ncols():
            raise NonSquareMatrixError()
        return c2py(deref(self.thisptr).det())

    def inv(self, method='LU'):
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())

        if method.upper() == 'LU':
            ## inv() method of DenseMatrixBase uses LU factorization
            deref(self.thisptr).inv(deref(result.thisptr))
        elif method.upper() == 'FFLU':
            symengine.inverse_FFLU(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(result.thisptr)))
        elif method.upper() == 'GJ':
            symengine.inverse_GJ(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(result.thisptr)))
        else:
            raise Exception("Unsupported method.")
        return result

    def add_matrix(self, A):
        cdef MatrixBase A_ = sympify(A)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).add_matrix(deref(A_.thisptr), deref(result.thisptr))
        return result

    def mul_matrix(self, A):
        cdef MatrixBase A_ = sympify(A)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), A.ncols())
        deref(self.thisptr).mul_matrix(deref(A_.thisptr), deref(result.thisptr))
        return result

    def add_scalar(self, k):
        cdef Basic k_ = sympify(k)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).add_scalar(k_.thisptr, deref(result.thisptr))
        return result

    def mul_scalar(self, k):
        cdef Basic k_ = sympify(k)
        cdef DenseMatrixBase result = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).mul_scalar(k_.thisptr, deref(result.thisptr))
        return result

    def transpose(self):
        cdef DenseMatrixBase result = self.__class__(self.ncols(), self.nrows())
        deref(self.thisptr).transpose(deref(result.thisptr))
        return result

    @property
    def T(self):
        return self.transpose()

    def applyfunc(self, f):
        cdef DenseMatrixBase out = self.__class__(self)
        cdef int nr = self.nrows()
        cdef int nc = self.ncols()
        cdef Basic e_;
        for i in range(nr):
            for j in range(nc):
                e_ = sympify(f(self._get(i, j)))
                if e_ is not None:
                    deref(out.thisptr).set(i, j, e_.thisptr)
        return out

    def _applyfunc(self, f):
        cdef int nr = self.nrows()
        cdef int nc = self.ncols()
        for i in range(nr):
            for j in range(nc):
                self._set(i, j, f(self._get(i, j)))

    def msubs(self, *args):
        cdef _DictBasic D = get_dict(*args)
        return self.applyfunc(lambda x: x.msubs(D))

    def diff(self, x):
        cdef Basic x_ = sympify(x)
        cdef DenseMatrixBase R = self.__class__(self.rows, self.cols)
        symengine.diff(<const symengine.DenseMatrix &>deref(self.thisptr),
                x_.thisptr, <symengine.DenseMatrix &>deref(R.thisptr))
        return R

    #TODO: implement this in C++
    def subs(self, *args):
        cdef _DictBasic D = get_dict(*args)
        return self.applyfunc(lambda x: x.subs(D))


    @property
    def free_symbols(self):
        s = set()
        for i in range(self.nrows()):
            for j in range(self.ncols()):
                s.update(self._get(i, j).free_symbols)
        return s

    def _submatrix(self, unsigned r_i, unsigned c_i, unsigned r_j, unsigned c_j, unsigned r_s=1, unsigned c_s=1):
        r_j, c_j = r_j - 1, c_j - 1
        cdef DenseMatrixBase result = self.__class__(((r_j - r_i) // r_s) + 1, ((c_j - c_i) // c_s) + 1)
        deref(self.thisptr).submatrix(deref(result.thisptr), r_i, c_i, r_j, c_j, r_s, c_s)
        return result

    def LU(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase U = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).LU(deref(L.thisptr), deref(U.thisptr))
        return L, U

    def LDL(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase D = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).LDL(deref(L.thisptr), deref(D.thisptr))
        return L, D

    def solve(self, b, method='LU'):
        cdef DenseMatrixBase b_ = sympify(b)
        cdef DenseMatrixBase x = self.__class__(b_.nrows(), b_.ncols())

        if method.upper() == 'LU':
            ## solve() method of DenseMatrixBase uses LU factorization
            symengine.pivoted_LU_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        elif method.upper() == 'FFLU':
            symengine.FFLU_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        elif method.upper() == 'LDL':
            symengine.LDL_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        elif method.upper() == 'FFGJ':
            symengine.FFGJ_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
                deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
                deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        else:
            raise Exception("Unsupported method.")

        return x

    def LUsolve(self, b):
        cdef DenseMatrixBase b_ = sympify(b)
        cdef DenseMatrixBase x = self.__class__(b.nrows(), b.ncols())
        symengine.pivoted_LU_solve(deref(symengine.static_cast_DenseMatrix(self.thisptr)),
            deref(symengine.static_cast_DenseMatrix(b_.thisptr)),
            deref(symengine.static_cast_DenseMatrix(x.thisptr)))
        return x

    def FFLU(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase U = self.__class__(self.nrows(), self.ncols(), [0]*self.nrows()*self.ncols())
        deref(self.thisptr).FFLU(deref(L.thisptr))

        for i in range(self.nrows()):
            for j in range(i + 1, self.ncols()):
                U.set(i, j, L.get(i, j))
                L.set(i, j, 0)
            U.set(i, i, L.get(i, i))

        return L, U

    def FFLDU(self):
        cdef DenseMatrixBase L = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase D = self.__class__(self.nrows(), self.ncols())
        cdef DenseMatrixBase U = self.__class__(self.nrows(), self.ncols())
        deref(self.thisptr).FFLDU(deref(L.thisptr), deref(D.thisptr), deref(U.thisptr))
        return L, D, U

    def jacobian(self, x):
        cdef DenseMatrixBase x_ = sympify(x)
        cdef DenseMatrixBase R = self.__class__(self.nrows(), x.nrows())
        symengine.jacobian(<const symengine.DenseMatrix &>deref(self.thisptr),
                <const symengine.DenseMatrix &>deref(x_.thisptr),
                <symengine.DenseMatrix &>deref(R.thisptr))
        return R

    def _sympy_(self):
        s = []
        cdef symengine.DenseMatrix A = deref(symengine.static_cast_DenseMatrix(self.thisptr))
        for i in range(A.nrows()):
            l = []
            for j in range(A.ncols()):
                l.append(c2py(A.get(i, j))._sympy_())
            s.append(l)
        import sympy
        return sympy.Matrix(s)

    def _sage_(self):
        s = []
        cdef symengine.DenseMatrix A = deref(symengine.static_cast_DenseMatrix(self.thisptr))
        for i in range(A.nrows()):
            l = []
            for j in range(A.ncols()):
                l.append(c2py(A.get(i, j))._sage_())
            s.append(l)
        import sage.all as sage
        return sage.Matrix(s)

    def dump_real(self, double[::1] out):
        cdef size_t ri, ci, nr, nc
        if out.size < self.size:
            raise ValueError("out parameter too short")
        nr = self.nrows()
        nc = self.ncols()
        for ri in range(nr):
            for ci in range(nc):
                out[ri*nc + ci] = symengine.eval_double(deref(
                    <symengine.rcp_const_basic>(deref(self.thisptr).get(ri, ci))))

    def dump_complex(self, double complex[::1] out):
        cdef size_t ri, ci, nr, nc
        if out.size < self.size:
            raise ValueError("out parameter too short")
        nr = self.nrows()
        nc = self.ncols()
        for ri in range(nr):
            for ci in range(nc):
                out[ri*nc + ci] = symengine.eval_complex_double(deref(
                    <symengine.rcp_const_basic>(deref(self.thisptr).get(ri, ci))))

    def __iter__(self):
        return DenseMatrixBaseIter(self)

    def as_mutable(self):
        return MutableDenseMatrix(self)

    def as_immutable(self):
        return ImmutableDenseMatrix(self)

    def tolist(self):
        return [[self[rowi, coli] for coli in range(self.ncols())]
                for rowi in range(self.nrows())]

    def __array__(self):
        import numpy as np
        return np.array(self.tolist())

    def _mat(self):
        return self

    def atoms(self, *types):
        if types:
            s = set()
            if (isinstance(self, types)):
                s.add(self)
            for arg in self:
                s.update(arg.atoms(*types))
            return s
        else:
           return self.free_symbols

    def simplify(self, *args, **kwargs):
        return self._applyfunc(lambda x : x.simplify(*args, **kwargs))

    def expand(self, *args, **kwargs):
        return self.applyfunc(lambda x : x.expand(*args, **kwargs))


def div_matrices(a, b):
    a = _sympify(a, False)
    b = _sympify(b, False)
    if isinstance(a, MatrixBase):
        if isinstance(b, MatrixBase):
            return a.mul_matrix(b.inv())
        elif isinstance(b, Basic):
            return a.mul_scalar(1/b)
        else:
            return NotImplemented
    else:
        return NotImplemented

class DenseMatrixBaseIter(object):

    def __init__(self, d):
        self.curr = -1
        self.d = d

    def __iter__(self):
        return self

    def __next__(self):
        self.curr = self.curr + 1
        if (self.curr < self.d.rows * self.d.cols):
            return self.d._get(self.curr // self.d.cols, self.curr % self.d.cols)
        else:
            raise StopIteration

    next = __next__

cdef class MutableDenseMatrix(DenseMatrixBase):

    def col_swap(self, i, j):
        symengine.column_exchange_dense(deref(symengine.static_cast_DenseMatrix(self.thisptr)), i, j)

    def fill(self, value):
        for i in range(self.rows):
            for j in range(self.cols):
                self[i, j] = value

    def row_swap(self, i, j):
        symengine.row_exchange_dense(deref(symengine.static_cast_DenseMatrix(self.thisptr)), i, j)

    def rowmul(self, i, c, *args):
        cdef Basic _c = sympify(c)
        symengine.row_mul_scalar_dense(deref(symengine.static_cast_DenseMatrix(self.thisptr)), i, _c.thisptr)
        return self

    def rowadd(self, i, j, c, *args):
        cdef Basic _c = sympify(c)
        symengine.row_add_row_dense(deref(symengine.static_cast_DenseMatrix(self.thisptr)), i, j, _c.thisptr)
        return self

    def row_del(self, i):
        if i < -self.rows or i >= self.rows:
            raise IndexError("Index out of range: 'i = %s', valid -%s <= i"
                             " < %s" % (i, self.rows, self.rows))
        if i < 0:
            i += self.rows
        deref(symengine.static_cast_DenseMatrix(self.thisptr)).row_del(i)
        return self

    def col_del(self, i):
        if i < -self.cols or i >= self.cols:
            raise IndexError("Index out of range: 'i=%s', valid -%s <= i < %s"
                             % (i, self.cols, self.cols))
        if i < 0:
            i += self.cols
        deref(symengine.static_cast_DenseMatrix(self.thisptr)).col_del(i)
        return self

Matrix = DenseMatrix = MutableDenseMatrix

cdef class ImmutableDenseMatrix(DenseMatrixBase):

    def __setitem__(self, key, value):
        raise TypeError("Cannot set values of {}".format(self.__class__))

ImmutableMatrix = ImmutableDenseMatrix

cdef matrix_to_vec(DenseMatrixBase d, symengine.vec_basic& v):
    cdef Basic e_
    for i in range(d.nrows()):
        for j in range(d.ncols()):
            e_ = d._get(i, j)
            v.push_back(e_.thisptr)

def eye(n):
    cdef DenseMatrixBase d = DenseMatrix(n, n)
    symengine.eye(deref(symengine.static_cast_DenseMatrix(d.thisptr)), 0)
    return d

def diag(*values):
    cdef DenseMatrixBase d = DenseMatrix(len(values), len(values))
    cdef symengine.vec_basic V
    cdef Basic B
    for b in values:
        B = sympify(b)
        V.push_back(B.thisptr)
    symengine.diag(deref(symengine.static_cast_DenseMatrix(d.thisptr)), V, 0)
    return d

def ones(r, c = None):
    if c is None:
        c = r
    cdef DenseMatrixBase d = DenseMatrix(r, c)
    symengine.ones(deref(symengine.static_cast_DenseMatrix(d.thisptr)))
    return d

def zeros(r, c = None):
    if c is None:
        c = r
    cdef DenseMatrixBase d = DenseMatrix(r, c)
    symengine.zeros(deref(symengine.static_cast_DenseMatrix(d.thisptr)))
    return d

