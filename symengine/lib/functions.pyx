cimport symengine
from symengine cimport RCP, pair, rcp_const_basic
from .core cimport Basic, PyModule, Boolean, Expr, c2py
from cpython cimport PyObject, Py_XINCREF
from cython.operator cimport dereference as deref

from .core import sympify, sage_module

class Function(Expr):

    def __new__(cls, *args, **kwargs):
        if cls == Function and len(args) == 1:
            return UndefFunction(args[0])
        return super(Function, cls).__new__(cls)

    @property
    def is_Function(self):
        return True

    def func(self, *values):
        import sys
        return getattr(sys.modules[__name__], self.__class__.__name__.lower())(*values)


class OneArgFunction(Function):

    def get_arg(Basic self):
        cdef RCP[const symengine.OneArgFunction] X = symengine.rcp_static_cast_OneArgFunction(self.thisptr)
        return c2py(deref(X).get_arg())

    def _sympy_(self):
        import sympy
        return getattr(sympy, self.__class__.__name__)(self.get_arg()._sympy_())

    def _sage_(self):
        import sage.all as sage
        return getattr(sage, self.__class__.__name__.lower())(self.get_arg()._sage_())


class HyperbolicFunction(OneArgFunction):
    pass

class TrigFunction(OneArgFunction):
    pass

class gamma(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.gamma(X.thisptr))

class LambertW(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.lambertw(X.thisptr))

    def _sage_(self):
        import sage.all as sage
        return sage.lambert_w(self.get_arg()._sage_())

class zeta(Function):
    def __new__(cls, s, a = None):
        cdef Basic S = sympify(s)
        if a == None:
            return c2py(symengine.zeta(S.thisptr))
        cdef Basic A = sympify(a)
        return c2py(symengine.zeta(S.thisptr, A.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.zeta(*self.args_as_sympy())

class dirichlet_eta(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.dirichlet_eta(X.thisptr))

class KroneckerDelta(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.kronecker_delta(X.thisptr, Y.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.KroneckerDelta(*self.args_as_sympy())

    def _sage_(self):
        import sage.all as sage
        return sage.kronecker_delta(*self.args_as_sage())

class LeviCivita(Function):
    def __new__(cls, *args):
        cdef symengine.vec_basic v
        cdef Basic e_
        for e in args:
            e_ = sympify(e)
            v.push_back(e_.thisptr)
        return c2py(symengine.levi_civita(v))

    def _sympy_(self):
        import sympy
        return sympy.LeviCivita(*self.args_as_sympy())

class erf(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.erf(X.thisptr))

class erfc(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.erfc(X.thisptr))

class lowergamma(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.lowergamma(X.thisptr, Y.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.lowergamma(*self.args_as_sympy())

    def _sage_(self):
        import sage.all as sage
        return sage.gamma_inc_lower(*self.args_as_sage())

class uppergamma(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.uppergamma(X.thisptr, Y.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.uppergamma(*self.args_as_sympy())

    def _sage_(self):
        import sage.all as sage
        return sage.gamma_inc(*self.args_as_sage())

class loggamma(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.loggamma(X.thisptr))

    def _sage_(self):
        import sage.all as sage
        return sage.log_gamma(self.get_arg()._sage_())

class beta(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.beta(X.thisptr, Y.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.beta(*self.args_as_sympy())

    def _sage_(self):
        import sage.all as sage
        return sage.beta(*self.args_as_sage())

class polygamma(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.polygamma(X.thisptr, Y.thisptr))

    def _sympy_(self):
        import sympy
        return sympy.polygamma(*self.args_as_sympy())

class sign(OneArgFunction):

    @property
    def is_complex(self):
        return True

    @property
    def is_finite(self):
        return True

    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sign(X.thisptr))

class floor(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.floor(X.thisptr))

class ceiling(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.ceiling(X.thisptr))

    def _sage_(self):
        import sage.all as sage
        return sage.ceil(self.get_arg()._sage_())

class conjugate(OneArgFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.conjugate(X.thisptr))

class log(OneArgFunction):
    def __new__(cls, x, y=None):
        cdef Basic X = sympify(x)
        if y == None:
            return c2py(symengine.log(X.thisptr))
        cdef Basic Y = sympify(y)
        return c2py(symengine.log(X.thisptr, Y.thisptr))

class sin(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sin(X.thisptr))

class cos(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.cos(X.thisptr))

class tan(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.tan(X.thisptr))

class cot(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.cot(X.thisptr))

class sec(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sec(X.thisptr))

class csc(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.csc(X.thisptr))

class asin(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asin(X.thisptr))

class acos(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acos(X.thisptr))

class atan(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.atan(X.thisptr))

class acot(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acot(X.thisptr))

class asec(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asec(X.thisptr))

class acsc(TrigFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acsc(X.thisptr))

class sinh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sinh(X.thisptr))

class cosh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.cosh(X.thisptr))

class tanh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.tanh(X.thisptr))

class coth(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.coth(X.thisptr))

class sech(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.sech(X.thisptr))

class csch(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.csch(X.thisptr))

class asinh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asinh(X.thisptr))

class acosh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acosh(X.thisptr))

class atanh(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.atanh(X.thisptr))

class acoth(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acoth(X.thisptr))

class asech(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.asech(X.thisptr))

class acsch(HyperbolicFunction):
    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.acsch(X.thisptr))

class atan2(Function):
    def __new__(cls, x, y):
        cdef Basic X = sympify(x)
        cdef Basic Y = sympify(y)
        return c2py(symengine.atan2(X.thisptr, Y.thisptr))

# For backwards compatibility

Sin = sin
Cos = cos
Tan = tan
Cot = cot
Sec = sec
Csc = csc
ASin = asin
ACos = acos
ATan = atan
ACot = acot
ASec = asec
ACsc = acsc
Sinh = sinh
Cosh = cosh
Tanh = tanh
Coth = coth
Sech = sech
Csch = csch
ASinh = asinh
ACosh = acosh
ATanh = atanh
ACoth = acoth
ASech = asech
ACsch = acsch
ATan2 = atan2
Log = log
Gamma = gamma


class Abs(OneArgFunction):

    @property
    def is_real(self):
        return True

    @property
    def is_negative(self):
        return False

    def __new__(cls, x):
        cdef Basic X = sympify(x)
        return c2py(symengine.abs(X.thisptr))

    def _sympy_(Basic self):
        cdef RCP[const symengine.Abs] X = symengine.rcp_static_cast_Abs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sympy_()
        return abs(arg)

    def _sage_(Basic self):
        cdef RCP[const symengine.Abs] X = symengine.rcp_static_cast_Abs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sage_()
        return abs(arg)

    @property
    def func(self):
        return self.__class__

class FunctionSymbol(Function):

    def get_name(Basic self):
        cdef RCP[const symengine.FunctionSymbol] X = \
            symengine.rcp_static_cast_FunctionSymbol(self.thisptr)
        name = deref(X).get_name().decode("utf-8")
        # In Python 2.7, function names cannot be unicode:
        return str(name)

    def _sympy_(self):
        import sympy
        name = self.get_name()
        return sympy.Function(name)(*self.args_as_sympy())

    def _sage_(self):
        import sage.all as sage
        name = self.get_name()
        return sage.function(name, *self.args_as_sage())

    def func(self, *values):
        name = self.get_name()
        return function_symbol(name, *values)


class UndefFunction(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, *values):
        return function_symbol(self.name, *values)


class Max(Function):

    def __new__(cls, *args):
        return _max(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Max(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.max(*s)

    @property
    def func(self):
        return self.__class__

class Min(Function):

    def __new__(cls, *args):
        return _min(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Min(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.min(*s)

    @property
    def func(self):
        return self.__class__



class Piecewise(Function):

    def __new__(self, *args):
        cdef symengine.PiecewiseVec vec
        cdef pair[rcp_const_basic, RCP[symengine.const_Boolean]] p
        cdef Basic e
        cdef Boolean b
        for expr, rel in args:
            e = sympify(expr)
            b = sympify(rel)
            p.first = <rcp_const_basic>e.thisptr
            p.second = <RCP[symengine.const_Boolean]>symengine.rcp_static_cast_Boolean(b.thisptr)
            vec.push_back(p)
        return c2py(symengine.piecewise(symengine.std_move_PiecewiseVec(vec)))

    def _sympy_(self):
        import sympy
        a = self.args
        l = []
        for i in range(0, len(a), 2):
            l.append((a[i]._sympy_(), a[i + 1]._sympy_()))
        return sympy.Piecewise(*l)


def sqrt(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.sqrt(X.thisptr))

def exp(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.exp(X.thisptr))

def _max(*args):
    cdef symengine.vec_basic v
    cdef Basic e_
    for e in args:
        e_ = sympify(e)
        v.push_back(e_.thisptr)
    return c2py(symengine.max(v))

def _min(*args):
    cdef symengine.vec_basic v
    cdef Basic e_
    for e in args:
        e_ = sympify(e)
        v.push_back(e_.thisptr)
    return c2py(symengine.min(v))

def digamma(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.digamma(X.thisptr))

def trigamma(x):
    cdef Basic X = sympify(x)
    return c2py(symengine.trigamma(X.thisptr))

def function_symbol(name, *args):
    cdef symengine.vec_basic v
    cdef Basic e_
    for e in args:
        e_ = sympify(e)
        if e_ is not None:
            v.push_back(e_.thisptr)
    return c2py(symengine.function_symbol(name.encode("utf-8"), v))
