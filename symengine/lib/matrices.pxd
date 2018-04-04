cimport symengine

cdef class MatrixBase(object):
    cdef symengine.MatrixBase* thisptr
