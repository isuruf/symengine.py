from cython.operator cimport dereference as deref, preincrement as inc
cimport symengine
from symengine cimport (RCP, map_basic_basic, umap_int_basic,
    umap_int_basic_iterator, umap_basic_num, umap_basic_num_iterator,
    rcp_const_basic, std_pair_short_rcp_const_basic,
    rcp_const_seriescoeffinterface)
from libcpp cimport bool as cppbool
from libcpp.string cimport string
from libcpp.vector cimport vector
from cpython cimport PyObject, Py_XINCREF, Py_XDECREF
from libc.string cimport memcpy
import cython
import itertools
import numbers
import collections
import warnings
from symengine.compatibility import is_sequence
from .matrices cimport MatrixBase
from .matrices import MatrixBase

include "config.pxi"

class SympifyError(Exception):
    pass

cdef object c2py(rcp_const_basic o):
    cdef Basic r
    if (symengine.is_a_Add(deref(o))):
        r = Expr.__new__(Add)
    elif (symengine.is_a_Mul(deref(o))):
        r = Expr.__new__(Mul)
    elif (symengine.is_a_Pow(deref(o))):
        r = Expr.__new__(Pow)
    elif (symengine.is_a_Integer(deref(o))):
        if (deref(symengine.rcp_static_cast_Integer(o)).is_zero()):
            return S.Zero
        elif (deref(symengine.rcp_static_cast_Integer(o)).is_one()):
            return S.One
        elif (deref(symengine.rcp_static_cast_Integer(o)).is_minus_one()):
            return S.NegativeOne
        r = Number.__new__(Integer)
    elif (symengine.is_a_Rational(deref(o))):
        r = S.Half
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.Half
        r = Number.__new__(Rational)
    elif (symengine.is_a_Complex(deref(o))):
        r = S.ImaginaryUnit
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.ImaginaryUnit
        r = Complex.__new__(Complex)
    elif (symengine.is_a_Dummy(deref(o))):
        r = Symbol.__new__(Dummy)
    elif (symengine.is_a_Symbol(deref(o))):
        if (symengine.is_a_PySymbol(deref(o))):
            return <object>(deref(symengine.rcp_static_cast_PySymbol(o)).get_py_object())
        r = Expr.__new__(Symbol)
    elif (symengine.is_a_Constant(deref(o))):
        r = S.Pi
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.Pi
        r = S.Exp1
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.Exp1
        r = S.GoldenRatio
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.GoldenRatio
        r = S.Catalan
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.Catalan
        r = S.EulerGamma
        if (symengine.eq(deref(o), deref(r.thisptr))):
            return S.EulerGamma
        r = Constant.__new__(Constant)
    elif (symengine.is_a_Infty(deref(o))):
        if (deref(symengine.rcp_static_cast_Infty(o)).is_positive()):
            return S.Infinity
        elif (deref(symengine.rcp_static_cast_Infty(o)).is_negative()):
            return S.NegativeInfinity
        return S.ComplexInfinity
    elif (symengine.is_a_NaN(deref(o))):
        return S.NaN
    elif (symengine.is_a_PyFunction(deref(o))):
        r = PyFunction.__new__(PyFunction)
    elif (symengine.is_a_FunctionSymbol(deref(o))):
        r = FunctionSymbol.__new__(FunctionSymbol)
    elif (symengine.is_a_Abs(deref(o))):
        r = Function.__new__(Abs)
    elif (symengine.is_a_Max(deref(o))):
        r = Function.__new__(Max)
    elif (symengine.is_a_Min(deref(o))):
        r = Function.__new__(Min)
    elif (symengine.is_a_BooleanAtom(deref(o))):
        if (deref(symengine.rcp_static_cast_BooleanAtom(o)).get_val()):
            return S.true
        return S.false
    elif (symengine.is_a_Equality(deref(o))):
        r = Relational.__new__(Equality)
    elif (symengine.is_a_Unequality(deref(o))):
        r = Relational.__new__(Unequality)
    elif (symengine.is_a_LessThan(deref(o))):
        r = Relational.__new__(LessThan)
    elif (symengine.is_a_StrictLessThan(deref(o))):
        r = Relational.__new__(StrictLessThan)
    elif (symengine.is_a_Gamma(deref(o))):
        r = Function.__new__(Gamma)
    elif (symengine.is_a_Derivative(deref(o))):
        r = Expr.__new__(Derivative)
    elif (symengine.is_a_Subs(deref(o))):
        r = Expr.__new__(Subs)
    elif (symengine.is_a_RealDouble(deref(o))):
        r = Number.__new__(RealDouble)
    elif (symengine.is_a_ComplexDouble(deref(o))):
        r = ComplexDouble.__new__(ComplexDouble)
    elif (symengine.is_a_RealMPFR(deref(o))):
        r = Number.__new__(RealMPFR)
    elif (symengine.is_a_ComplexMPC(deref(o))):
        r = ComplexMPC.__new__(ComplexMPC)
    elif (symengine.is_a_Log(deref(o))):
        r = Function.__new__(Log)
    elif (symengine.is_a_Sin(deref(o))):
        r = Function.__new__(Sin)
    elif (symengine.is_a_Cos(deref(o))):
        r = Function.__new__(Cos)
    elif (symengine.is_a_Tan(deref(o))):
        r = Function.__new__(Tan)
    elif (symengine.is_a_Cot(deref(o))):
        r = Function.__new__(Cot)
    elif (symengine.is_a_Csc(deref(o))):
        r = Function.__new__(Csc)
    elif (symengine.is_a_Sec(deref(o))):
        r = Function.__new__(Sec)
    elif (symengine.is_a_ASin(deref(o))):
        r = Function.__new__(ASin)
    elif (symengine.is_a_ACos(deref(o))):
        r = Function.__new__(ACos)
    elif (symengine.is_a_ATan(deref(o))):
        r = Function.__new__(ATan)
    elif (symengine.is_a_ACot(deref(o))):
        r = Function.__new__(ACot)
    elif (symengine.is_a_ACsc(deref(o))):
        r = Function.__new__(ACsc)
    elif (symengine.is_a_ASec(deref(o))):
        r = Function.__new__(ASec)
    elif (symengine.is_a_Sinh(deref(o))):
        r = Function.__new__(Sinh)
    elif (symengine.is_a_Cosh(deref(o))):
        r = Function.__new__(Cosh)
    elif (symengine.is_a_Tanh(deref(o))):
        r = Function.__new__(Tanh)
    elif (symengine.is_a_Coth(deref(o))):
        r = Function.__new__(Coth)
    elif (symengine.is_a_Csch(deref(o))):
        r = Function.__new__(Csch)
    elif (symengine.is_a_Sech(deref(o))):
        r = Function.__new__(Sech)
    elif (symengine.is_a_ASinh(deref(o))):
        r = Function.__new__(ASinh)
    elif (symengine.is_a_ACosh(deref(o))):
        r = Function.__new__(ACosh)
    elif (symengine.is_a_ATanh(deref(o))):
        r = Function.__new__(ATanh)
    elif (symengine.is_a_ACoth(deref(o))):
        r = Function.__new__(ACoth)
    elif (symengine.is_a_ACsch(deref(o))):
        r = Function.__new__(ACsch)
    elif (symengine.is_a_ASech(deref(o))):
        r = Function.__new__(ASech)
    elif (symengine.is_a_ATan2(deref(o))):
        r = Function.__new__(ATan2)
    elif (symengine.is_a_LambertW(deref(o))):
        r = Function.__new__(LambertW)
    elif (symengine.is_a_Zeta(deref(o))):
        r = Function.__new__(zeta)
    elif (symengine.is_a_DirichletEta(deref(o))):
        r = Function.__new__(dirichlet_eta)
    elif (symengine.is_a_KroneckerDelta(deref(o))):
        r = Function.__new__(KroneckerDelta)
    elif (symengine.is_a_LeviCivita(deref(o))):
        r = Function.__new__(LeviCivita)
    elif (symengine.is_a_Erf(deref(o))):
        r = Function.__new__(erf)
    elif (symengine.is_a_Erfc(deref(o))):
        r = Function.__new__(erfc)
    elif (symengine.is_a_LowerGamma(deref(o))):
        r = Function.__new__(lowergamma)
    elif (symengine.is_a_UpperGamma(deref(o))):
        r = Function.__new__(uppergamma)
    elif (symengine.is_a_LogGamma(deref(o))):
        r = Function.__new__(loggamma)
    elif (symengine.is_a_Beta(deref(o))):
        r = Function.__new__(beta)
    elif (symengine.is_a_PolyGamma(deref(o))):
        r = Function.__new__(polygamma)
    elif (symengine.is_a_Sign(deref(o))):
        r = Function.__new__(sign)
    elif (symengine.is_a_Floor(deref(o))):
        r = Function.__new__(floor)
    elif (symengine.is_a_Ceiling(deref(o))):
        r = Function.__new__(ceiling)
    elif (symengine.is_a_Conjugate(deref(o))):
        r = Function.__new__(conjugate)
    elif (symengine.is_a_PyNumber(deref(o))):
        r = PyNumber.__new__(PyNumber)
    elif (symengine.is_a_Piecewise(deref(o))):
        r = Function.__new__(Piecewise)
    elif (symengine.is_a_Contains(deref(o))):
        r = Boolean.__new__(Contains)
    elif (symengine.is_a_Interval(deref(o))):
        r = Set.__new__(Interval)
    elif (symengine.is_a_EmptySet(deref(o))):
        r = Set.__new__(EmptySet)
    elif (symengine.is_a_UniversalSet(deref(o))):
        r = Set.__new__(UniversalSet)
    elif (symengine.is_a_FiniteSet(deref(o))):
        r = Set.__new__(FiniteSet)
    elif (symengine.is_a_Union(deref(o))):
        r = Set.__new__(Union)
    elif (symengine.is_a_Complement(deref(o))):
        r = Set.__new__(Complement)
    elif (symengine.is_a_ConditionSet(deref(o))):
        r = Set.__new__(ConditionSet)
    elif (symengine.is_a_ImageSet(deref(o))):
        r = Set.__new__(ImageSet)
    elif (symengine.is_a_And(deref(o))):
        r = Boolean.__new__(And)
    elif (symengine.is_a_Not(deref(o))):
        r = Boolean.__new__(Not)
    elif (symengine.is_a_Or(deref(o))):
        r = Boolean.__new__(Or)
    elif (symengine.is_a_Xor(deref(o))):
        r = Boolean.__new__(Xor)
    else:
        raise Exception("Unsupported SymEngine class.")
    r.thisptr = o
    return r

def sympy2symengine(a, raise_error=False):
    """
    Converts 'a' from SymPy to SymEngine.

    If the expression cannot be converted, it either returns None (if
    raise_error==False) or raises a SympifyError exception (if
    raise_error==True).
    """
    import sympy
    from sympy.core.function import AppliedUndef as sympy_AppliedUndef
    if isinstance(a, sympy.Symbol):
        return Symbol(a.name)
    elif isinstance(a, sympy.Dummy):
        return Dummy(a.name)
    elif isinstance(a, sympy.Mul):
        return mul(*[sympy2symengine(x, raise_error) for x in a.args])
    elif isinstance(a, sympy.Add):
        return add(*[sympy2symengine(x, raise_error) for x in a.args])
    elif isinstance(a, (sympy.Pow, sympy.exp)):
        x, y = a.as_base_exp()
        return sympy2symengine(x, raise_error) ** sympy2symengine(y, raise_error)
    elif isinstance(a, sympy.Integer):
        return Integer(a.p)
    elif isinstance(a, sympy.Rational):
        return Integer(a.p) / Integer(a.q)
    elif isinstance(a, sympy.Float):
        IF HAVE_SYMENGINE_MPFR:
            if a._prec > 53:
                return RealMPFR(str(a), a._prec)
            else:
                return RealDouble(float(str(a)))
        ELSE:
            return RealDouble(float(str(a)))
    elif a is sympy.I:
        return I
    elif a is sympy.E:
        return E
    elif a is sympy.pi:
        return pi
    elif a is sympy.GoldenRatio:
        return golden_ratio
    elif a is sympy.Catalan:
        return catalan
    elif a is sympy.EulerGamma:
        return eulergamma
    elif a is sympy.S.NegativeInfinity:
        return minus_oo
    elif a is sympy.S.Infinity:
        return oo
    elif a is sympy.S.ComplexInfinity:
        return zoo
    elif a is sympy.nan:
        return nan
    elif a is sympy.S.true:
        return true
    elif a is sympy.S.false:
        return false
    elif isinstance(a, sympy.functions.elementary.trigonometric.TrigonometricFunction):
        if isinstance(a, sympy.sin):
            return sin(a.args[0])
        elif isinstance(a, sympy.cos):
            return cos(a.args[0])
        elif isinstance(a, sympy.tan):
            return tan(a.args[0])
        elif isinstance(a, sympy.cot):
            return cot(a.args[0])
        elif isinstance(a, sympy.csc):
            return csc(a.args[0])
        elif isinstance(a, sympy.sec):
            return sec(a.args[0])
    elif isinstance(a, sympy.functions.elementary.trigonometric.InverseTrigonometricFunction):
        if isinstance(a, sympy.asin):
            return asin(a.args[0])
        elif isinstance(a, sympy.acos):
            return acos(a.args[0])
        elif isinstance(a, sympy.atan):
            return atan(a.args[0])
        elif isinstance(a, sympy.acot):
            return acot(a.args[0])
        elif isinstance(a, sympy.acsc):
            return acsc(a.args[0])
        elif isinstance(a, sympy.asec):
            return asec(a.args[0])
        elif isinstance(a, sympy.atan2):
            return atan2(*a.args)
    elif isinstance(a, sympy.functions.elementary.hyperbolic.HyperbolicFunction):
        if isinstance(a, sympy.sinh):
            return sinh(a.args[0])
        elif isinstance(a, sympy.cosh):
            return cosh(a.args[0])
        elif isinstance(a, sympy.tanh):
            return tanh(a.args[0])
        elif isinstance(a, sympy.coth):
            return coth(a.args[0])
        elif isinstance(a, sympy.csch):
            return csch(a.args[0])
        elif isinstance(a, sympy.sech):
            return sech(a.args[0])
    elif isinstance(a, sympy.asinh):
        return asinh(a.args[0])
    elif isinstance(a, sympy.acosh):
        return acosh(a.args[0])
    elif isinstance(a, sympy.atanh):
        return atanh(a.args[0])
    elif isinstance(a, sympy.acoth):
        return acoth(a.args[0])
    elif isinstance(a, sympy.log):
        return log(a.args[0])
    elif isinstance(a, sympy.Abs):
        return abs(sympy2symengine(a.args[0], raise_error))
    elif isinstance(a, sympy.Max):
        return _max(*a.args)
    elif isinstance(a, sympy.Min):
        return _min(*a.args)
    elif isinstance(a, sympy.Equality):
        return eq(*a.args)
    elif isinstance(a, sympy.Unequality):
        return ne(*a.args)
    elif isinstance(a, sympy.GreaterThan):
        return ge(*a.args)
    elif isinstance(a, sympy.StrictGreaterThan):
        return gt(*a.args)
    elif isinstance(a, sympy.LessThan):
        return le(*a.args)
    elif isinstance(a, sympy.StrictLessThan):
        return lt(*a.args)
    elif isinstance(a, sympy.LambertW):
        return LambertW(a.args[0])
    elif isinstance(a, sympy.zeta):
        return zeta(*a.args)
    elif isinstance(a, sympy.dirichlet_eta):
        return dirichlet_eta(a.args[0])
    elif isinstance(a, sympy.KroneckerDelta):
        return KroneckerDelta(*a.args)
    elif isinstance(a, sympy.LeviCivita):
        return LeviCivita(*a.args)
    elif isinstance(a, sympy.erf):
        return erf(a.args[0])
    elif isinstance(a, sympy.erfc):
        return erfc(a.args[0])
    elif isinstance(a, sympy.lowergamma):
        return lowergamma(*a.args)
    elif isinstance(a, sympy.uppergamma):
        return uppergamma(*a.args)
    elif isinstance(a, sympy.loggamma):
        return loggamma(a.args[0])
    elif isinstance(a, sympy.beta):
        return beta(*a.args)
    elif isinstance(a, sympy.polygamma):
        return polygamma(*a.args)
    elif isinstance(a, sympy.sign):
        return sign(a.args[0])
    elif isinstance(a, sympy.floor):
        return floor(a.args[0])
    elif isinstance(a, sympy.ceiling):
        return ceiling(a.args[0])
    elif isinstance(a, sympy.conjugate):
        return conjugate(a.args[0])
    elif isinstance(a, sympy.And):
        return logical_and(*a.args)
    elif isinstance(a, sympy.Or):
        return logical_or(*a.args)
    elif isinstance(a, sympy.Not):
        return logical_not(a.args[0])
    elif isinstance(a, sympy.Nor):
        return Nor(*a.args)
    elif isinstance(a, sympy.Nand):
        return Nand(*a.args)
    elif isinstance(a, sympy.Xor):
        return logical_xor(*a.args)
    elif isinstance(a, sympy.gamma):
        return gamma(a.args[0])
    elif isinstance(a, sympy.Derivative):
        return Derivative(a.expr, *a.variables)
    elif isinstance(a, sympy.Subs):
        return Subs(a.expr, a.variables, a.point)
    elif isinstance(a, sympy_AppliedUndef):
        name = str(a.func)
        return function_symbol(name, *(a.args))
    elif isinstance(a, (sympy.Piecewise)):
        return Piecewise(*(a.args))
    elif isinstance(a, sympy.Interval):
        return interval(*(a.args))
    elif isinstance(a, sympy.EmptySet):
        return emptyset()
    elif isinstance(a, sympy.FiniteSet):
        return finiteset(*(a.args))
    elif isinstance(a, sympy.Contains):
        return contains(*(a.args))
    elif isinstance(a, sympy.Union):
        return set_union(*(a.args))
    elif isinstance(a, sympy.Intersection):
        return set_intersection(*(a.args))
    elif isinstance(a, sympy.Complement):
        return set_complement(*(a.args))
    elif isinstance(a, sympy.ImageSet):
        return imageset(*(a.args))
    elif isinstance(a, sympy.Function):
        return PyFunction(a, a.args, a.func, sympy_module)
    elif isinstance(a, sympy.MatrixBase):
        row, col = a.shape
        v = []
        for r in a.tolist():
            for e in r:
                v.append(e)
        if isinstance(a, sympy.MutableDenseMatrix):
            from .matrices import MutableDenseMatrix
            return MutableDenseMatrix(row, col, v)
        elif isinstance(a, sympy.ImmutableDenseMatrix):
            from .matrices import ImmutableDenseMatrix
            return ImmutableDenseMatrix(row, col, v)
        else:
            raise NotImplementedError
    elif isinstance(a, sympy.polys.domains.modularinteger.ModularInteger):
        return PyNumber(a, sympy_module)
    elif sympy.__version__ > '1.0':
        if isinstance(a, sympy.acsch):
            return acsch(a.args[0])
        elif isinstance(a, sympy.asech):
            return asech(a.args[0])
        elif isinstance(a, sympy.ConditionSet):
            return conditionset(*(a.args))

    if raise_error:
        raise SympifyError(("sympy2symengine: Cannot convert '%r' (of type %s)"
                            " to a symengine type.") % (a, type(a)))


def sympify(a):
    """
    Converts an expression 'a' into a SymEngine type.

    Arguments
    =========

    a ............. An expression to convert.

    Examples
    ========

    >>> from symengine import sympify
    >>> sympify(1)
    1
    >>> sympify("a+b")
    a + b
    """
    if isinstance(a, str):
        return c2py(symengine.parse(a.encode("utf-8")))
    elif isinstance(a, tuple):
        v = []
        for e in a:
            v.append(sympify(e))
        return tuple(v)
    elif isinstance(a, list):
        v = []
        for e in a:
            v.append(sympify(e))
        return v
    return _sympify(a, True)


def _sympify(a, raise_error=True):
    """
    Converts an expression 'a' into a SymEngine type.

    Arguments
    =========

    a ............. An expression to convert.
    raise_error ... Will raise an error on a failure (default True), otherwise
                    it returns None if 'a' cannot be converted.

    Examples
    ========

    >>> from symengine.li.symengine_wrapper import _sympify
    >>> _sympify(1)
    1
    >>> _sympify("abc", False)
    >>>

    """
    if isinstance(a, (Basic, MatrixBase)):
        return a
    elif isinstance(a, bool):
        return (true if a else false)
    elif isinstance(a, numbers.Integral):
        return Integer(a)
    elif isinstance(a, float):
        return RealDouble(a)
    elif isinstance(a, complex):
        return ComplexDouble(a)
    elif hasattr(a, '_symengine_'):
        return _sympify(a._symengine_(), raise_error)
    elif hasattr(a, '_sympy_'):
        return _sympify(a._sympy_(), raise_error)
    elif hasattr(a, 'pyobject'):
        return _sympify(a.pyobject(), raise_error)

    try:
        import sympy
        return sympy2symengine(a, raise_error)
    except ImportError:
        pass

    if raise_error:
        raise SympifyError(
            "sympify: Cannot convert '%r' (of type %s) to a symengine type." % (
                a, type(a)))


class Singleton(object):

    __call__ = staticmethod(sympify)

    @property
    def Zero(self):
        return zero

    @property
    def One(self):
        return one

    @property
    def NegativeOne(self):
        return minus_one

    @property
    def Half(self):
        return half

    @property
    def Pi(self):
        return pi

    @property
    def NaN(self):
        return nan

    @property
    def Infinity(self):
        return oo

    @property
    def NegativeInfinity(self):
        return minus_oo

    @property
    def ComplexInfinity(self):
        return zoo

    @property
    def Exp1(self):
        return E

    @property
    def GoldenRatio(self):
        return golden_ratio

    @property
    def Catalan(self):
        return catalan

    @property
    def EulerGamma(self):
        return eulergamma

    @property
    def ImaginaryUnit(self):
        return I

    @property
    def true(self):
        return true

    @property
    def false(self):
        return false

S = Singleton()


cdef class DictBasicIter(object):

    cdef init(self, map_basic_basic.iterator begin, map_basic_basic.iterator end):
        self.begin = begin
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.begin != self.end:
            obj = c2py(deref(self.begin).first)
        else:
            raise StopIteration
        inc(self.begin)
        return obj


cdef class _DictBasic(object):

    def __init__(self, tocopy = None):
        if tocopy != None:
            self.add_dict(tocopy)

    def as_dict(self):
        ret = {}
        it = self.c.begin()
        while it != self.c.end():
            ret[c2py(deref(it).first)] = c2py(deref(it).second)
            inc(it)
        return ret

    def add_dict(self, d):
        cdef _DictBasic D
        if isinstance(d, DictBasic):
            D = d
            self.c.insert(D.c.begin(), D.c.end())
        else:
            for key, value in d.iteritems():
                self.add(key, value)

    def add(self, key, value):
        cdef Basic K = sympify(key)
        cdef Basic V = sympify(value)
        cdef symengine.std_pair_rcp_const_basic_rcp_const_basic pair
        pair.first = K.thisptr
        pair.second = V.thisptr
        return self.c.insert(pair).second

    def copy(self):
        return DictBasic(self)

    __copy__ = copy

    def __len__(self):
        return self.c.size()

    def __getitem__(self, key):
        cdef Basic K = sympify(key)
        it = self.c.find(K.thisptr)
        if it == self.c.end():
            raise KeyError(key)
        else:
            return c2py(deref(it).second)

    def __setitem__(self, key, value):
        cdef Basic K = sympify(key)
        cdef Basic V = sympify(value)
        self.c[K.thisptr] = V.thisptr

    def clear(self):
        self.clear()

    def __delitem__(self, key):
        cdef Basic K = sympify(key)
        self.c.erase(K.thisptr)

    def __contains__(self, key):
        cdef Basic K = sympify(key)
        it = self.c.find(K.thisptr)
        return it != self.c.end()

    def __iter__(self):
        cdef DictBasicIter d = DictBasicIter()
        d.init(self.c.begin(), self.c.end())
        return d


class DictBasic(_DictBasic, collections.MutableMapping):

    def __str__(self):
        return "{" + ", ".join(["%s: %s" % (str(key), str(value)) for key, value in self.items()]) + "}"

    def __repr__(self):
        return self.__str__()

def get_dict(*args):
    """
    Returns a DictBasic instance from args. Inputs can be,
        1. a DictBasic
        2. a Python dictionary
        3. two args old, new
    """
    cdef _DictBasic D = DictBasic()
    if len(args) == 2:
        if is_sequence(args[0]):
            for k, v in zip(args[0], args[1]):
                D.add(k, v)
        else:
            D.add(args[0], args[1])
        return D
    elif len(args) == 1:
        arg = args[0]
    else:
        raise TypeError("subs/msubs takes one or two arguments (%d given)" % \
                len(args))
    if isinstance(arg, DictBasic):
        return arg
    for k, v in arg.items():
        D.add(k, v)
    return D


cdef tuple vec_basic_to_tuple(symengine.vec_basic& vec):
    return tuple(vec_basic_to_list(vec))


cdef list vec_basic_to_list(symengine.vec_basic& vec):
    result = []
    for i in range(vec.size()):
        result.append(c2py(<rcp_const_basic>(vec[i])))
    return result


cdef list vec_pair_to_list(symengine.vec_pair& vec):
    result = []
    cdef rcp_const_basic a, b
    for i in range(vec.size()):
        a = <rcp_const_basic>vec[i].first
        b = <rcp_const_basic>vec[i].second
        result.append((c2py(a), c2py(b)))
    return result


cdef class Basic(object):

    def __str__(self):
        return deref(self.thisptr).__str__().decode("utf-8")

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return deref(self.thisptr).hash()

    def __dealloc__(self):
        self.thisptr.reset()

    def _unsafe_reset(self):
        self.thisptr.reset()

    def __add__(a, b):
        cdef Basic A = _sympify(a, False)
        B_ = _sympify(b, False)
        if A is None or B_ is None or isinstance(B_, MatrixBase): return NotImplemented
        cdef Basic B = B_
        return c2py(symengine.add(A.thisptr, B.thisptr))

    def __sub__(a, b):
        cdef Basic A = _sympify(a, False)
        B_ = _sympify(b, False)
        if A is None or B_ is None or isinstance(B_, MatrixBase): return NotImplemented
        cdef Basic B = B_
        return c2py(symengine.sub(A.thisptr, B.thisptr))

    def __mul__(a, b):
        cdef Basic A = _sympify(a, False)
        B_ = _sympify(b, False)
        if A is None or B_ is None or isinstance(B_, MatrixBase): return NotImplemented
        cdef Basic B = B_
        return c2py(symengine.mul(A.thisptr, B.thisptr))

    def __truediv__(a, b):
        cdef Basic A = _sympify(a, False)
        cdef Basic B = _sympify(b, False)
        if A is None or B is None: return NotImplemented
        return c2py(symengine.div(A.thisptr, B.thisptr))

    # This is for Python 2.7 compatibility only:
    def __div__(a, b):
        cdef Basic A = _sympify(a, False)
        cdef Basic B = _sympify(b, False)
        if A is None or B is None: return NotImplemented
        return c2py(symengine.div(A.thisptr, B.thisptr))

    def __pow__(a, b, c):
        if c is not None:
            return powermod(a, b, c)
        cdef Basic A = _sympify(a, False)
        cdef Basic B = _sympify(b, False)
        if A is None or B is None: return NotImplemented
        return c2py(symengine.pow(A.thisptr, B.thisptr))

    def __neg__(Basic self not None):
        return c2py(symengine.neg(self.thisptr))

    def __abs__(Basic self not None):
        return c2py(symengine.abs(self.thisptr))

    def __richcmp__(a, b, int op):
        A = _sympify(a, False)
        B = _sympify(b, False)
        if not (isinstance(A, Basic) and isinstance(B, Basic)):
            if (op == 2):
                return False
            elif (op == 3):
                return True
            else:
                return NotImplemented
        return Basic._richcmp_(A, B, op)

    def _richcmp_(Basic A, Basic B, int op):
        if (op == 2):
            return symengine.eq(deref(A.thisptr), deref(B.thisptr))
        elif (op == 3):
            return symengine.neq(deref(A.thisptr), deref(B.thisptr))
        if (op == 0):
            return c2py(<rcp_const_basic>(symengine.Lt(A.thisptr, B.thisptr)))
        elif (op == 1):
            return c2py(<rcp_const_basic>(symengine.Le(A.thisptr, B.thisptr)))
        elif (op == 4):
            return c2py(<rcp_const_basic>(symengine.Gt(A.thisptr, B.thisptr)))
        elif (op == 5):
            return c2py(<rcp_const_basic>(symengine.Ge(A.thisptr, B.thisptr)))

    def expand(Basic self not None, cppbool deep=True):
        return c2py(symengine.expand(self.thisptr, deep))

    def diff(Basic self not None, x = None):
        if x is None:
            f = self.free_symbols
            if (len(f) != 1):
                raise RuntimeError("Variable w.r.t should be given")
            return self.diff(f.pop())
        cdef Basic s = sympify(x)
        return c2py(symengine.diff(self.thisptr, s.thisptr))

    def subs_dict(Basic self not None, *args):
        warnings.warn("subs_dict() is deprecated. Use subs() instead", DeprecationWarning)
        return self.subs(*args)

    def subs_oldnew(Basic self not None, old, new):
        warnings.warn("subs_oldnew() is deprecated. Use subs() instead", DeprecationWarning)
        return self.subs({old: new})

    def subs(Basic self not None, *args):
        cdef _DictBasic D = get_dict(*args)
        return c2py(symengine.ssubs(self.thisptr, D.c))

    replace = xreplace = subs

    def msubs(Basic self not None, *args):
        cdef _DictBasic D = get_dict(*args)
        return c2py(symengine.msubs(self.thisptr, D.c))

    def as_numer_denom(Basic self not None):
        cdef rcp_const_basic _num, _den
        symengine.as_numer_denom(self.thisptr, symengine.outArg(_num), symengine.outArg(_den))
        return c2py(<rcp_const_basic>_num), c2py(<rcp_const_basic>_den)

    def as_real_imag(Basic self not None):
        cdef rcp_const_basic _real, _imag
        symengine.as_real_imag(self.thisptr, symengine.outArg(_real), symengine.outArg(_imag))
        return c2py(<rcp_const_basic>_real), c2py(<rcp_const_basic>_imag)

    def n(self, prec = 53, real = False):
        if real:
            return eval_real(self, prec)
        else:
            return eval(self, prec)

    evalf = n

    @property
    def args(self):
        cdef symengine.vec_basic args = deref(self.thisptr).get_args()
        return vec_basic_to_tuple(args)

    @property
    def free_symbols(self):
        cdef symengine.set_basic _set = symengine.free_symbols(deref(self.thisptr))
        return {c2py(<rcp_const_basic>(elem)) for elem in _set}

    @property
    def is_Atom(self):
        return False

    @property
    def is_Symbol(self):
        return False

    @property
    def is_symbol(self):
        return False

    @property
    def is_Dummy(self):
        return False

    @property
    def is_Function(self):
        return False

    @property
    def is_Add(self):
        return False

    @property
    def is_Mul(self):
        return False

    @property
    def is_Pow(self):
        return False

    @property
    def is_Number(self):
        return False

    @property
    def is_number(self):
        return None

    @property
    def is_Float(self):
        return False

    @property
    def is_Rational(self):
        return False

    @property
    def is_Integer(self):
        return False

    @property
    def is_integer(self):
        return False

    @property
    def is_finite(self):
        return None

    @property
    def is_Derivative(self):
        return False

    @property
    def is_AlgebraicNumber(self):
        return False

    @property
    def is_Relational(self):
        return False

    @property
    def is_Equality(self):
        return False

    @property
    def is_Boolean(self):
        return False

    @property
    def is_Not(self):
        return False

    @property
    def is_Matrix(self):
        return False

    def copy(self):
        return self

    def _symbolic_(self, ring):
        return ring(self._sage_())

    def atoms(self, *types):
        if types:
            s = set()
            if (isinstance(self, types)):
                s.add(self)
            for arg in self.args:
                s.update(arg.atoms(*types))
            return s
        else:
            return self.free_symbols

    def simplify(self, *args, **kwargs):
        return sympify(self._sympy_().simplify(*args, **kwargs))

    def as_coefficients_dict(self):
        d = collections.defaultdict(int)
        d[self] = 1
        return d

    def coeff(self, x, n=1):
        cdef Basic _x = sympify(x)
        require(_x, Symbol)
        cdef Basic _n = sympify(n)
        return c2py(symengine.coeff(deref(self.thisptr), deref(_x.thisptr), deref(_n.thisptr)))

    def has(self, *symbols):
        return any([has_symbol(self, symbol) for symbol in symbols])

    def args_as_sage(Basic self):
        cdef symengine.vec_basic Y = deref(self.thisptr).get_args()
        s = []
        for i in range(Y.size()):
            s.append(c2py(<rcp_const_basic>(Y[i]))._sage_())
        return s

    def args_as_sympy(Basic self):
        cdef symengine.vec_basic Y = deref(self.thisptr).get_args()
        s = []
        for i in range(Y.size()):
            s.append(c2py(<rcp_const_basic>(Y[i]))._sympy_())
        return s

def series(ex, x=None, x0=0, n=6, as_deg_coef_pair=False):
    # TODO: check for x0 an infinity, see sympy/core/expr.py
    # TODO: nonzero x0
    # underscored local vars are of symengine.py type
    cdef Basic _ex = sympify(ex)
    syms = _ex.free_symbols
    if not syms:
        return _ex

    cdef Basic _x
    if x is None:
        _x = list(syms)[0]
    else:
        _x = sympify(x)
    require(_x, Symbol)
    if not _x in syms:
        return _ex

    if x0 != 0:
        _ex = _ex.subs({_x: _x + x0})

    cdef RCP[const symengine.Symbol] X = symengine.rcp_static_cast_Symbol(_x.thisptr)
    cdef umap_int_basic umap
    cdef umap_int_basic_iterator iter, iterend

    if not as_deg_coef_pair:
        b = c2py(<symengine.rcp_const_basic>deref(symengine.series(_ex.thisptr, X, n)).as_basic())
        if x0 != 0:
            b = b.subs({_x: _x - x0})
        return b

    umap = deref(symengine.series(_ex.thisptr, X, n)).as_dict()

    iter = umap.begin()
    iterend = umap.end()
    poly = 0
    l = []
    while iter != iterend:
        l.append([deref(iter).first, c2py(<symengine.rcp_const_basic>(deref(iter).second))])
        inc(iter)
    if as_deg_coef_pair:
        return l
    return add(*l)


class Symbol(Expr):

    """
    Symbol is a class to store a symbolic variable with a given name.
    """

    def __init__(Basic self, name, *args, **kwargs):
        if type(self) == Symbol:
            self.thisptr = symengine.make_rcp_Symbol(name.encode("utf-8"))
        else:
            self.thisptr = symengine.make_rcp_PySymbol(name.encode("utf-8"), <PyObject*>self)

    def _sympy_(self):
        import sympy
        return sympy.Symbol(str(self))

    def _sage_(self):
        import sage.all as sage
        return sage.SR.symbol(str(self))

    @property
    def name(self):
        return self.__str__()

    @property
    def is_Atom(self):
        return True

    @property
    def is_Symbol(self):
        return True

    @property
    def is_symbol(self):
        return True

    @property
    def is_commutative(self):
        return True

    @property
    def func(self):
        return self.__class__


class Dummy(Symbol):

    def __init__(Basic self, name=None, *args, **kwargs):
        if name is None:
            self.thisptr = symengine.make_rcp_Dummy()
        else:
            self.thisptr = symengine.make_rcp_Dummy(name.encode("utf-8"))

    def _sympy_(self):
        import sympy
        return sympy.Dummy(str(self))

    @property
    def is_Dummy(self):
        return True

    @property
    def func(self):
        return self.__class__


def symarray(prefix, shape, **kwargs):
    """ Creates an nd-array of symbols

    Parameters
    ----------
    prefix: str
    shape: tuple
    \*\*kwargs:
        Passed on to :class:`Symbol`.

    Notes
    -----
    This function requires NumPy.

    """
    import numpy as np
    arr = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        arr[index] = Symbol('%s_%s' % (prefix, '_'.join(map(str, index))), **kwargs)
    return arr


cdef class Constant(Expr):

    def __cinit__(self, name = None):
        if name is None:
            return
        self.thisptr = symengine.make_rcp_Constant(name.encode("utf-8"))

    def _sympy_(self):
        raise Exception("Unknown Constant")

    def _sage_(self):
        raise Exception("Unknown Constant")


cdef class ImaginaryUnit(Complex):

    def __cinit__(Basic self):
        self.thisptr = symengine.I

I = ImaginaryUnit()


cdef class Pi(Constant):

    def __cinit__(Basic self):
        self.thisptr = symengine.pi

    def _sympy_(self):
        import sympy
        return sympy.pi

    def _sage_(self):
        import sage.all as sage
        return sage.pi

pi = Pi()


cdef class Exp1(Constant):

    def __cinit__(Basic self):
        self.thisptr = symengine.E

    def _sympy_(self):
        import sympy
        return sympy.E

    def _sage_(self):
        import sage.all as sage
        return sage.e

E = Exp1()


cdef class GoldenRatio(Constant):

    def __cinit__(Basic self):
        self.thisptr = symengine.GoldenRatio

    def _sympy_(self):
        import sympy
        return sympy.GoldenRatio

    def _sage_(self):
        import sage.all as sage
        return sage.golden_ratio

golden_ratio = GoldenRatio()


cdef class Catalan(Constant):

    def __cinit__(Basic self):
        self.thisptr = symengine.Catalan

    def _sympy_(self):
        import sympy
        return sympy.Catalan

    def _sage_(self):
        import sage.all as sage
        return sage.catalan

catalan = Catalan()


cdef class EulerGamma(Constant):

    def __cinit__(Basic self):
        self.thisptr = symengine.EulerGamma

    def _sympy_(self):
        import sympy
        return sympy.EulerGamma

    def _sage_(self):
        import sage.all as sage
        return sage.euler_gamma

eulergamma = EulerGamma()


cdef class Boolean(Expr):

    def logical_not(self):
        return c2py(<rcp_const_basic>(deref(symengine.rcp_static_cast_Boolean(self.thisptr)).logical_not()))


cdef class BooleanAtom(Boolean):

    @property
    def is_Boolean(self):
        return True

    @property
    def is_Atom(self):
        return True


cdef class BooleanTrue(BooleanAtom):

    def __cinit__(Basic self):
        self.thisptr = symengine.boolTrue

    def _sympy_(self):
        import sympy
        return sympy.S.true

    def _sage_(self):
        return True

    def __nonzero__(self):
        return True

    def __bool__(self):
        return True


true = BooleanTrue()


cdef class BooleanFalse(BooleanAtom):

    def __cinit__(Basic self):
        self.thisptr = symengine.boolFalse

    def _sympy_(self):
        import sympy
        return sympy.S.false

    def _sage_(self):
        return False

    def __nonzero__(self):
        return False

    def __bool__(self):
        return False

false = BooleanFalse()


class And(Boolean):

    def __new__(cls, *args):
        return logical_and(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.And(*s)


class Or(Boolean):

    def __new__(cls, *args):
        return logical_or(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Or(*s)


class Not(Boolean):

    def __new__(cls, x):
        return logical_not(x)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()[0]
        return sympy.Not(s)


class Xor(Boolean):

    def __new__(cls, *args):
        return logical_xor(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Xor(*s)


class Relational(Boolean):

    @property
    def is_Relational(self):
        return True

Rel = Relational


class Equality(Relational):

    def __new__(cls, *args):
        return eq(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Equality(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.eq(*s)

    @property
    def is_Equality(self):
        return True

    func = __class__


Eq = Equality


class Unequality(Relational):

    def __new__(cls, *args):
        return ne(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.Unequality(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.ne(*s)

    func = __class__


Ne = Unequality


class LessThan(Relational):

    def __new__(cls, *args):
        return le(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.LessThan(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.le(*s)


Le = LessThan


class StrictLessThan(Relational):

    def __new__(cls, *args):
        return lt(*args)

    def _sympy_(self):
        import sympy
        s = self.args_as_sympy()
        return sympy.StrictLessThan(*s)

    def _sage_(self):
        import sage.all as sage
        s = self.args_as_sage()
        return sage.lt(*s)


Lt = StrictLessThan


cdef class Number(Expr):
    @property
    def is_Atom(self):
        return True

    @property
    def is_Number(self):
        return True

    @property
    def is_number(self):
        return True

    @property
    def is_commutative(self):
        return True

    @property
    def is_positive(Basic self):
        return deref(symengine.rcp_static_cast_Number(self.thisptr)).is_positive()

    @property
    def is_negative(Basic self):
        return deref(symengine.rcp_static_cast_Number(self.thisptr)).is_negative()

    @property
    def is_zero(Basic self):
        return deref(symengine.rcp_static_cast_Number(self.thisptr)).is_zero()

    @property
    def is_nonzero(self):
        return not (self.is_complex or self.is_zero)

    @property
    def is_nonnegative(self):
        return not (self.is_complex or self.is_negative)

    @property
    def is_nonpositive(self):
        return not (self.is_complex or self.is_positive)

    @property
    def is_complex(Basic self):
        return deref(symengine.rcp_static_cast_Number(self.thisptr)).is_complex()


class Rational(Number):

    def __new__(cls, p, q):
        return Integer(p)/q

    @property
    def is_Rational(self):
        return True

    @property
    def is_rational(self):
        return True

    @property
    def is_real(self):
        return True

    @property
    def is_finite(self):
        return True

    @property
    def is_integer(self):
        return False

    @property
    def p(self):
        return self.get_num_den()[0]

    @property
    def q(self):
        return self.get_num_den()[1]

    def get_num_den(Basic self):
        cdef RCP[const symengine.Integer] _num, _den
        symengine.get_num_den(deref(symengine.rcp_static_cast_Rational(self.thisptr)),
                           symengine.outArg_Integer(_num), symengine.outArg_Integer(_den))
        return [c2py(<rcp_const_basic>_num), c2py(<rcp_const_basic>_den)]

    def _sympy_(self):
        rat = self.get_num_den()
        return rat[0]._sympy_() / rat[1]._sympy_()

    def _sage_(self):
        try:
            from sage.symbolic.symengine_conversions import convert_to_rational
            return convert_to_rational(self)
        except ImportError:
            rat = self.get_num_den()
            return rat[0]._sage_() / rat[1]._sage_()

    @property
    def func(self):
        return self.__class__

class Integer(Rational):

    def __new__(cls, i):
        i = int(i)
        cdef int i_
        cdef symengine.integer_class i__
        cdef string tmp
        try:
            # Try to convert "i" to int
            i_ = i
            int_ok = True
        except OverflowError:
            # Too big, need to use mpz
            int_ok = False
            tmp = str(i).encode("utf-8")
            i__ = symengine.integer_class(tmp)
        # Note: all other exceptions are left intact
        if int_ok:
            return c2py(<rcp_const_basic>symengine.integer(i_))
        else:
            return c2py(<rcp_const_basic>symengine.integer(i__))

    @property
    def is_Integer(self):
        return True

    @property
    def is_integer(self):
        return True

    def doit(self, **hints):
        return self

    def __hash__(Basic self):
        return deref(self.thisptr).hash()

    def __richcmp__(a, b, int op):
        A = _sympify(a, False)
        B = _sympify(b, False)
        if not (isinstance(A, Integer) and isinstance(B, Integer)):
            if (op == 2):
                return False
            elif (op == 3):
                return True
            return NotImplemented
        return Integer._richcmp_(A, B, op)

    def _richcmp_(Basic A, Basic B, int op):
        cdef int i = deref(symengine.rcp_static_cast_Integer(A.thisptr)).compare(deref(symengine.rcp_static_cast_Integer(B.thisptr)))
        if (op == 0):
            return i < 0
        elif (op == 1):
            return i <= 0
        elif (op == 2):
            return i == 0
        elif (op == 3):
            return i != 0
        elif (op == 4):
            return i > 0
        elif (op == 5):
            return i >= 0
        else:
            return NotImplemented

    def __floordiv__(x, y):
        return quotient(x, y)

    def __mod__(x, y):
        return mod(x, y)

    def __divmod__(x, y):
        return quotient_mod(x, y)

    def _sympy_(Basic self):
        import sympy
        return sympy.Integer(deref(self.thisptr).__str__().decode("utf-8"))

    def _sage_(Basic self):
        try:
            from sage.symbolic.symengine_conversions import convert_to_integer
            return convert_to_integer(self)
        except ImportError:
            import sage.all as sage
            return sage.Integer(str(self))

    def __int__(self):
        return int(str(self))

    def __long__(self):
        return long(str(self))

    def __float__(self):
        return float(str(self))

    @property
    def p(self):
        return int(self)

    @property
    def q(self):
        return 1

    def get_num_den(Basic self):
        return self, 1

    @property
    def func(self):
        return self.__class__


def dps_to_prec(n):
    """Return the number of bits required to represent n decimals accurately."""
    return max(1, int(round((int(n)+1)*3.3219280948873626)))


class BasicMeta(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, self._classes)

class Float(Number):

    @property
    def is_rational(self):
        return None

    @property
    def is_irrational(self):
        return None

    @property
    def is_real(self):
        return True

    @property
    def is_Float(self):
        return True

    def __new__(cls, num, dps=None, precision=None):
        if cls is not Float:
            return super(Float, cls).__new__(cls)

        if dps is not None and precision is not None:
            raise ValueError('Both decimal and binary precision supplied. '
                             'Supply only one. ')
        if dps is None and precision is None:
            dps = 15
        if precision is None:
            precision = dps_to_prec(dps)

        IF HAVE_SYMENGINE_MPFR:
            if precision > 53:
                if isinstance(num, RealMPFR) and precision == num.get_prec():
                    return num
                return RealMPFR(str(num), precision)
        if precision > 53:
            raise ValueError('RealMPFR unavailable for high precision numerical values.')
        elif isinstance(num, RealDouble):
            return num
        else:
            return RealDouble(float(num))


RealNumber = Float


class RealDouble(Float):

    def __new__(cls, i):
        cdef double i_ = i
        return c2py(symengine.make_rcp_RealDouble(i_))

    def _sympy_(Basic self):
        import sympy
        return sympy.Float(deref(self.thisptr).__str__().decode("utf-8"))

    def _sage_(Basic self):
        import sage.all as sage
        cdef double i = deref(symengine.rcp_static_cast_RealDouble(self.thisptr)).as_double()
        return sage.RealDoubleField()(i)

    def __float__(self):
        return float(str(self))


cdef class ComplexDouble(Number):

    def __cinit__(self, i = None):
        if i is None:
            return
        cdef double complex i_ = i
        self.thisptr = symengine.make_rcp_ComplexDouble(i_)

    def real_part(Basic self):
        return c2py(<rcp_const_basic>deref(symengine.rcp_static_cast_ComplexDouble(self.thisptr)).real_part())

    def imaginary_part(Basic self):
        return c2py(<rcp_const_basic>deref(symengine.rcp_static_cast_ComplexDouble(self.thisptr)).imaginary_part())

    def _sympy_(self):
        import sympy
        return self.real_part()._sympy_() + sympy.I * self.imaginary_part()._sympy_()

    def _sage_(self):
        import sage.all as sage
        return self.real_part()._sage_() + sage.I * self.imaginary_part()._sage_()


class RealMPFR(Float):

    IF HAVE_SYMENGINE_MPFR:
        def __new__(cls, i = None, long prec = 53, unsigned base = 10):
            if i is None:
                return
            cdef string i_ = str(i).encode("utf-8")
            cdef symengine.mpfr_class m
            m = symengine.mpfr_class(i_, prec, base)
            return c2py(<rcp_const_basic>symengine.real_mpfr(symengine.std_move_mpfr(m)))

        def get_prec(Basic self):
            return Integer(deref(symengine.rcp_static_cast_RealMPFR(self.thisptr)).get_prec())

        def _sympy_(self):
            import sympy
            cdef long prec_ = self.get_prec()
            prec = max(1, int(round(prec_/3.3219280948873626)-1))
            return sympy.Float(str(self), prec)

        def _sage_(self):
            try:
                from sage.symbolic.symengine_conversions import convert_to_real_number
                return convert_to_real_number(self)
            except ImportError:
                import sage.all as sage
                return sage.RealField(int(self.get_prec()))(str(self))

        def __float__(self):
            return float(str(self))
    ELSE:
        pass


cdef class ComplexMPC(Number):
    IF HAVE_SYMENGINE_MPC:
        def __cinit__(self, i = None, j = 0, long prec = 53, unsigned base = 10):
            if i is None:
                return
            cdef string i_ = ("(" + str(i) + " " + str(j) + ")").encode("utf-8")
            cdef symengine.mpc_class m = symengine.mpc_class(i_, prec, base)
            self.thisptr = <rcp_const_basic>symengine.complex_mpc(symengine.std_move_mpc(m))

        def real_part(self):
            return c2py(<rcp_const_basic>deref(symengine.rcp_static_cast_ComplexMPC(self.thisptr)).real_part())

        def imaginary_part(self):
            return c2py(<rcp_const_basic>deref(symengine.rcp_static_cast_ComplexMPC(self.thisptr)).imaginary_part())

        def _sympy_(self):
            import sympy
            return self.real_part()._sympy_() + sympy.I * self.imaginary_part()._sympy_()

        def _sage_(self):
            try:
                from sage.symbolic.symengine_conversions import convert_to_mpcomplex_number
                return convert_to_mpcomplex_number(self)
            except ImportError:
                import sage.all as sage
                return sage.MPComplexField(int(self.get_prec()))(str(self.real_part()), str(self.imaginary_part()))
    ELSE:
        pass


cdef class Complex(Number):

    def real_part(self):
        return c2py(<rcp_const_basic>deref(symengine.rcp_static_cast_Complex(self.thisptr)).real_part())

    def imaginary_part(self):
        return c2py(<rcp_const_basic>deref(symengine.rcp_static_cast_Complex(self.thisptr)).imaginary_part())

    def _sympy_(self):
        import sympy
        return self.real_part()._sympy_() + sympy.I * self.imaginary_part()._sympy_()

    def _sage_(self):
        import sage.all as sage
        return self.real_part()._sage_() + sage.I * self.imaginary_part()._sage_()


cdef class Infinity(Number):

    @property
    def is_infinite(self):
        return True

    def __cinit__(Basic self):
        self.thisptr = symengine.Inf

    def _sympy_(self):
        import sympy
        return sympy.oo

    def _sage_(self):
        import sage.all as sage
        return sage.oo

oo = Infinity()


cdef class NegativeInfinity(Number):

    @property
    def is_infinite(self):
        return True

    def __cinit__(Basic self):
        self.thisptr = symengine.neg(symengine.Inf)

    def _sympy_(self):
        import sympy
        return -sympy.oo

    def _sage_(self):
        import sage.all as sage
        return -sage.oo

minus_oo = NegativeInfinity()


cdef class ComplexInfinity(Number):

    @property
    def is_infinite(self):
        return True

    def __cinit__(Basic self):
        self.thisptr = symengine.ComplexInf

    def _sympy_(self):
        import sympy
        return sympy.zoo

    def _sage_(self):
        import sage.all as sage
        return sage.unsigned_infinity

zoo = ComplexInfinity()


cdef class NaN(Number):

    @property
    def is_rational(self):
        return None

    @property
    def is_integer(self):
        return None

    @property
    def is_real(self):
        return None

    @property
    def is_finite(self):
        return None

    def __cinit__(Basic self):
        self.thisptr = symengine.Nan

    def _sympy_(self):
        import sympy
        return sympy.nan

    def _sage_(self):
        import sage.all as sage
        return sage.NaN

nan = NaN()


class Zero(Integer):
    def __new__(cls):
        cdef Basic r = Number.__new__(Zero)
        r.thisptr = <rcp_const_basic>symengine.integer(0)
        return r

zero = Zero()


class One(Integer):
    def __new__(cls):
        cdef Basic r = Number.__new__(One)
        r.thisptr = <rcp_const_basic>symengine.integer(1)
        return r

one = One()


class NegativeOne(Integer):
    def __new__(cls):
        cdef Basic r = Number.__new__(NegativeOne)
        r.thisptr = <rcp_const_basic>symengine.integer(-1)
        return r

minus_one = NegativeOne()


class Half(Rational):
    def __new__(cls):
        cdef Basic q = Number.__new__(Half)
        q.thisptr = <rcp_const_basic>symengine.rational(1, 2)
        return q

half = Half()


class AssocOp(Expr):

    @classmethod
    def make_args(cls, expr):
        if isinstance(expr, cls):
            return expr.args
        else:
            return (sympify(expr),)


class Add(AssocOp):

    identity = 0

    def __new__(cls, *args, **kwargs):
        cdef symengine.vec_basic v_
        cdef Basic e
        for e_ in args:
            e = _sympify(e_)
            v_.push_back(e.thisptr)
        return c2py(symengine.add(v_))

    @classmethod
    def _from_args(self, args):
        if len(args) == 0:
            return self.identity
        elif len(args) == 1:
            return args[0]

        return Add(*args)

    @property
    def is_Add(self):
        return True

    def _sympy_(self):
        from sympy import Add
        return Add(*self.args)

    def _sage_(Basic self):
        cdef RCP[const symengine.Add] X = symengine.rcp_static_cast_Add(self.thisptr)
        cdef rcp_const_basic a, b
        deref(X).as_two_terms(symengine.outArg(a), symengine.outArg(b))
        return c2py(a)._sage_() + c2py(b)._sage_()

    @property
    def func(self):
        return self.__class__

    def as_coefficients_dict(Basic self):
        cdef RCP[const symengine.Add] X = symengine.rcp_static_cast_Add(self.thisptr)
        cdef umap_basic_num umap
        cdef umap_basic_num_iterator iter, iterend
        d = collections.defaultdict(int)
        d[Integer(1)] = c2py(<rcp_const_basic>(deref(X).get_coef()))
        umap = deref(X).get_dict()
        iter = umap.begin()
        iterend = umap.end()
        while iter != iterend:
            d[c2py(<rcp_const_basic>(deref(iter).first))] =\
                    c2py(<rcp_const_basic>(deref(iter).second))
            inc(iter)
        return d


class Mul(AssocOp):

    identity = 1

    def __new__(cls, *args, **kwargs):
        cdef symengine.vec_basic v_
        cdef Basic e
        for e_ in args:
            e = _sympify(e_)
            v_.push_back(e.thisptr)
        return c2py(symengine.mul(v_))

    @classmethod
    def _from_args(self, args):
        if len(args) == 0:
            return self.identity
        elif len(args) == 1:
            return args[0]

        return Mul(*args)

    @property
    def is_Mul(self):
        return True

    def _sympy_(self):
        from sympy import Mul
        return Mul(*self.args)

    def _sage_(Basic self):
        cdef RCP[const symengine.Mul] X = symengine.rcp_static_cast_Mul(self.thisptr)
        cdef rcp_const_basic a, b
        deref(X).as_two_terms(symengine.outArg(a), symengine.outArg(b))
        return c2py(a)._sage_() * c2py(b)._sage_()

    @property
    def func(self):
        return self.__class__

    def as_coefficients_dict(Basic self):
        cdef RCP[const symengine.Mul] X = symengine.rcp_static_cast_Mul(self.thisptr)
        cdef RCP[const symengine.Integer] one = symengine.integer(1)
        cdef map_basic_basic dict = deref(X).get_dict()
        d = collections.defaultdict(int)
        d[c2py(<rcp_const_basic>symengine.mul_from_dict(\
                <RCP[const symengine.Number]>(one),
                symengine.std_move_map_basic_basic(dict)))] =\
                c2py(<rcp_const_basic>deref(X).get_coef())
        return d


class Pow(Expr):

    def __new__(cls, a, b):
        return _sympify(a) ** b

    @property
    def base(Basic self):
        cdef RCP[const symengine.Pow] X = symengine.rcp_static_cast_Pow(self.thisptr)
        return c2py(deref(X).get_base())

    @property
    def exp(Basic self):
        cdef RCP[const symengine.Pow] X = symengine.rcp_static_cast_Pow(self.thisptr)
        return c2py(deref(X).get_exp())

    def as_base_exp(self):
        return self.base, self.exp

    @property
    def is_Pow(self):
        return True

    @property
    def is_commutative(self):
        return (self.base.is_commutative and self.exp.is_commutative)

    def _sympy_(Basic self):
        cdef RCP[const symengine.Pow] X = symengine.rcp_static_cast_Pow(self.thisptr)
        base = c2py(deref(X).get_base())
        exp = c2py(deref(X).get_exp())
        return base._sympy_() ** exp._sympy_()

    def _sage_(Basic self):
        cdef RCP[const symengine.Pow] X = symengine.rcp_static_cast_Pow(self.thisptr)
        base = c2py(deref(X).get_base())
        exp = c2py(deref(X).get_exp())
        return base._sage_() ** exp._sage_()

    @property
    def func(self):
        return self.__class__

# For backwards compatibility
add = Add
mul = Mul

cdef rcp_const_basic pynumber_to_symengine(PyObject* o1):
    cdef Basic X = sympify(<object>o1)
    return X.thisptr

cdef PyObject* symengine_to_sage(rcp_const_basic o1):
    import sage.all as sage
    t = sage.SR(c2py(o1)._sage_())
    Py_XINCREF(<PyObject*>t)
    return <PyObject*>(t)

cdef PyObject* symengine_to_sympy(rcp_const_basic o1):
    t = c2py(o1)._sympy_()
    Py_XINCREF(<PyObject*>t)
    return <PyObject*>(t)

cdef RCP[const symengine.Number] sympy_eval(PyObject* o1, long bits):
    prec = max(1, int(round(bits/3.3219280948873626)-1))
    cdef Number X = sympify((<object>o1).n(prec))
    return symengine.rcp_static_cast_Number(X.thisptr)

cdef RCP[const symengine.Number] sage_eval(PyObject* o1, long bits):
    cdef Number X = sympify((<object>o1).n(bits))
    return symengine.rcp_static_cast_Number(X.thisptr)

cdef rcp_const_basic sage_diff(PyObject* o1, rcp_const_basic symbol):
    cdef Basic X = sympify((<object>o1).diff(c2py(symbol)._sage_()))
    return X.thisptr

cdef rcp_const_basic sympy_diff(PyObject* o1, rcp_const_basic symbol):
    cdef Basic X = sympify((<object>o1).diff(c2py(symbol)._sympy_()))
    return X.thisptr

def create_sympy_module():
    cdef PyModule s = PyModule.__new__(PyModule)
    s.thisptr = symengine.make_rcp_PyModule(&symengine_to_sympy, &pynumber_to_symengine, &sympy_eval,
                                    &sympy_diff)
    return s

def create_sage_module():
    cdef PyModule s = PyModule.__new__(PyModule)
    s.thisptr = symengine.make_rcp_PyModule(&symengine_to_sage, &pynumber_to_symengine, &sage_eval,
                                    &sage_diff)
    return s

sympy_module = create_sympy_module()
sage_module = create_sage_module()

from .functions import *

cdef class PyNumber(Number):
    def __cinit__(self, obj = None, PyModule module = None):
        if obj is None:
            return
        Py_XINCREF(<PyObject*>(obj))
        self.thisptr = symengine.make_rcp_PyNumber(<PyObject*>(obj), module.thisptr)

    def _sympy_(self):
        import sympy
        return sympy.sympify(self.pyobject())

    def _sage_(self):
        import sage.all as sage
        return sage.SR(self.pyobject())

    def pyobject(self):
        return <object>deref(symengine.rcp_static_cast_PyNumber(self.thisptr)).get_py_object()


class PyFunction(FunctionSymbol):

    def __init__(Basic self, pyfunction = None, args = None, pyfunction_class=None, module=None):
        if pyfunction is None:
            return
        cdef symengine.vec_basic v
        cdef Basic arg_
        for arg in args:
            arg_ = sympify(arg)
            v.push_back(arg_.thisptr)
        cdef PyFunctionClass _pyfunction_class = get_function_class(pyfunction_class, module)
        cdef PyObject* _pyfunction = <PyObject*>pyfunction
        Py_XINCREF(_pyfunction)
        self.thisptr = symengine.make_rcp_PyFunction(v, _pyfunction_class.thisptr, _pyfunction)

    def _sympy_(self):
        import sympy
        return sympy.sympify(self.pyobject())

    def _sage_(self):
        import sage.all as sage
        return sage.SR(self.pyobject())

    def pyobject(Basic self):
        return <object>deref(symengine.rcp_static_cast_PyFunction(self.thisptr)).get_py_object()


cdef class PyFunctionClass(object):

    def __cinit__(self, function, PyModule module not None):
        self.thisptr = symengine.make_rcp_PyFunctionClass(<PyObject*>(function), str(function).encode("utf-8"),
                                module.thisptr)

funcs = {}

def get_function_class(function, module):
    if not function in funcs:
        funcs[function] = PyFunctionClass(function, module)
    return funcs[function]

# TODO: remove this once SymEngine conversions are available in Sage.
def wrap_sage_function(func):
    return PyFunction(func, func.operands(), func.operator(), sage_module)

class Derivative(Expr):

    def __new__(self, expr, *variables):
        if len(variables) == 1 and is_sequence(variables[0]):
            return diff(expr, *variables[0])
        return diff(expr, *variables)

    @property
    def is_Derivative(self):
        return True

    @property
    def expr(Basic self):
        cdef RCP[const symengine.Derivative] X = symengine.rcp_static_cast_Derivative(self.thisptr)
        return c2py(deref(X).get_arg())

    @property
    def variables(self):
        return self.args[1:]

    def _sympy_(Basic self):
        cdef RCP[const symengine.Derivative] X = \
            symengine.rcp_static_cast_Derivative(self.thisptr)
        arg = c2py(deref(X).get_arg())._sympy_()
        cdef symengine.multiset_basic Y = deref(X).get_symbols()
        s = []
        for i in Y:
            s.append(c2py(<rcp_const_basic>(i))._sympy_())
        import sympy
        return sympy.Derivative(arg, *s)

    def _sage_(Basic self):
        cdef RCP[const symengine.Derivative] X = \
            symengine.rcp_static_cast_Derivative(self.thisptr)
        arg = c2py(deref(X).get_arg())._sage_()
        cdef symengine.multiset_basic Y = deref(X).get_symbols()
        s = []
        for i in Y:
            s.append(c2py(<rcp_const_basic>(i))._sage_())
        return arg.diff(*s)

    @property
    def func(self):
        return self.__class__


class Subs(Expr):

    def __new__(self, expr, variables, point):
        return sympify(expr).subs(variables, point)

    @property
    def expr(Basic self):
        cdef RCP[const symengine.Subs] me = symengine.rcp_static_cast_Subs(self.thisptr)
        return c2py(deref(me).get_arg())

    @property
    def variables(Basic self):
        cdef RCP[const symengine.Subs] me = symengine.rcp_static_cast_Subs(self.thisptr)
        cdef symengine.vec_basic variables = deref(me).get_variables()
        return vec_basic_to_tuple(variables)

    @property
    def point(Basic self):
        cdef RCP[const symengine.Subs] me = symengine.rcp_static_cast_Subs(self.thisptr)
        cdef symengine.vec_basic point = deref(me).get_point()
        return vec_basic_to_tuple(point)

    def _sympy_(Basic self):
        cdef RCP[const symengine.Subs] X = symengine.rcp_static_cast_Subs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sympy_()
        cdef symengine.vec_basic V = deref(X).get_variables()
        cdef symengine.vec_basic P = deref(X).get_point()
        v = []
        p = []
        for i in range(V.size()):
            v.append(c2py(<rcp_const_basic>(V[i]))._sympy_())
            p.append(c2py(<rcp_const_basic>(P[i]))._sympy_())
        import sympy
        return sympy.Subs(arg, v, p)

    def _sage_(Basic self):
        cdef RCP[const symengine.Subs] X = symengine.rcp_static_cast_Subs(self.thisptr)
        arg = c2py(deref(X).get_arg())._sage_()
        cdef symengine.vec_basic V = deref(X).get_variables()
        cdef symengine.vec_basic P = deref(X).get_point()
        v = {}
        for i in range(V.size()):
            v[c2py(<rcp_const_basic>(V[i]))._sage_()] = \
                c2py(<rcp_const_basic>(P[i]))._sage_()
        return arg.subs(v)

    @property
    def func(self):
        return self.__class__


cdef class Set(Expr):

    def intersection(self, a):
        cdef Set other = sympify(a)
        cdef RCP[const symengine.Set] other_ = symengine.rcp_static_cast_Set(other.thisptr)
        return c2py(<rcp_const_basic>(deref(symengine.rcp_static_cast_Set(self.thisptr))
                    .set_intersection(other_)))

    def union(self, a):
        cdef Set other = sympify(a)
        cdef RCP[const symengine.Set] other_ = symengine.rcp_static_cast_Set(other.thisptr)
        return c2py(<rcp_const_basic>(deref(symengine.rcp_static_cast_Set(self.thisptr))
                    .set_union(other_)))

    def complement(self, a):
        cdef Set other = sympify(a)
        cdef RCP[const symengine.Set] other_ = symengine.rcp_static_cast_Set(other.thisptr)
        return c2py(<rcp_const_basic>(deref(symengine.rcp_static_cast_Set(self.thisptr))
                    .set_complement(other_)))

    def contains(self, a):
        cdef Basic a_ = sympify(a)
        return c2py(<rcp_const_basic>(deref(symengine.rcp_static_cast_Set(self.thisptr))
                    .contains(a_.thisptr)))


class Interval(Set):

    def __new__(self, *args):
        return interval(*args)

    def _sympy_(self):
        import sympy
        return sympy.Interval(*[arg._sympy_() for arg in self.args])


class EmptySet(Set):

    def __new__(self):
        return emptyset()

    def _sympy_(self):
        import sympy
        return sympy.EmptySet()

    @property
    def func(self):
        return self.__class__


class UniversalSet(Set):

    def __new__(self):
        return universalset()

    def _sympy_(self):
        import sympy
        return sympy.S.UniversalSet

    @property
    def func(self):
        return self.__class__


class FiniteSet(Set):

    def __new__(self, *args):
        return finiteset(*args)

    def _sympy_(self):
        import sympy
        return sympy.FiniteSet(*[arg._sympy_() for arg in self.args])


class Contains(Boolean):

    def __new__(self, expr, sset):
        return contains(expr, sset)

    def _sympy_(self):
        import sympy
        return sympy.Contains(*[arg._sympy_() for arg in self.args])


class Union(Set):

    def __new__(self, *args):
        return set_union(*args)

    def _sympy_(self):
        import sympy
        return sympy.Union(*[arg._sympy_() for arg in self.args])


class Complement(Set):

    def __new__(self, universe, container):
        return set_complement(universe, container)

    def _sympy_(self):
        import sympy
        return sympy.Complement(*[arg._sympy_() for arg in self.args])


class ConditionSet(Set):

    def __new__(self, sym, condition):
        return conditionset(sym, condition)


class ImageSet(Set):

    def __new__(self, sym, expr, base):
        return imageset(sym, expr, base)


cdef class Sieve:
    @staticmethod
    def generate_primes(n):
        cdef symengine.vector[unsigned] primes
        symengine.sieve_generate_primes(primes, n)
        s = []
        for i in range(primes.size()):
            s.append(primes[i])
        return s

cdef class Sieve_iterator:
    cdef symengine.sieve_iterator *thisptr
    cdef unsigned limit
    def __cinit__(self):
        self.thisptr = new symengine.sieve_iterator()
        self.limit = 0

    def __cinit__(self, n):
        self.thisptr = new symengine.sieve_iterator(n)
        self.limit = n

    def __iter__(self):
        return self

    def __next__(self):
        n = deref(self.thisptr).next_prime()
        if self.limit > 0 and n > self.limit:
            raise StopIteration
        else:
            return n


def module_cleanup():
    global I, E, pi, oo, minus_oo, zoo, nan, true, false, golden_ratio, catalan, eulergamma, sympy_module, sage_module
    del I, E, pi, oo, minus_oo, zoo, nan, true, false, golden_ratio, catalan, eulergamma, sympy_module, sage_module

import atexit
atexit.register(module_cleanup)

def diff(ex, *x):
    ex = sympify(ex)
    for i in x:
        ex = ex.diff(i)
    return ex

def expand(x, deep=True):
    return sympify(x).expand(deep)

expand_mul = expand

def perfect_power(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return symengine.perfect_power(deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def is_square(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return symengine.perfect_square(deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def integer_nthroot(a, n):
    cdef Basic _a = sympify(a)
    require(_a, Integer)
    cdef RCP[const symengine.Integer] _r
    cdef int ret_val = symengine.i_nth_root(symengine.outArg_Integer(_r), deref(symengine.rcp_static_cast_Integer(_a.thisptr)), n)
    return (c2py(<rcp_const_basic>_r), ret_val == 1)

def eq(lhs, rhs = None):
    cdef Basic X = sympify(lhs)
    if rhs is None:
        return c2py(<rcp_const_basic>(symengine.Eq(X.thisptr)))
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Eq(X.thisptr, Y.thisptr)))

def ne(lhs, rhs):
    cdef Basic X = sympify(lhs)
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Ne(X.thisptr, Y.thisptr)))

def ge(lhs, rhs):
    cdef Basic X = sympify(lhs)
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Ge(X.thisptr, Y.thisptr)))

Ge = GreaterThan = ge

def gt(lhs, rhs):
    cdef Basic X = sympify(lhs)
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Gt(X.thisptr, Y.thisptr)))

Gt = StrictGreaterThan = gt

def le(lhs, rhs):
    cdef Basic X = sympify(lhs)
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Le(X.thisptr, Y.thisptr)))

def lt(lhs, rhs):
    cdef Basic X = sympify(lhs)
    cdef Basic Y = sympify(rhs)
    return c2py(<rcp_const_basic>(symengine.Lt(X.thisptr, Y.thisptr)))

def logical_and(*args):
    cdef symengine.set_boolean s
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_and(s)))

def logical_or(*args):
    cdef symengine.set_boolean s
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_or(s)))

def Nor(*args):
    cdef symengine.set_boolean s
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_nor(s)))

def Nand(*args):
    cdef symengine.set_boolean s
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_nand(s)))

def logical_not(x):
    cdef Basic x_ = sympify(x)
    require(x_, Boolean)
    cdef RCP[const symengine.Boolean] _x = symengine.rcp_static_cast_Boolean(x_.thisptr)
    return c2py(<rcp_const_basic>(symengine.logical_not(_x)))

def logical_xor(*args):
    cdef symengine.vec_boolean v
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        v.push_back(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_xor(v)))

def Xnor(*args):
    cdef symengine.vec_boolean v
    cdef Boolean e_
    for e in args:
        e_ = sympify(e)
        v.push_back(symengine.rcp_static_cast_Boolean(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.logical_xnor(v)))

def eval_double(x):
    cdef Basic X = sympify(x)
    return c2py(<rcp_const_basic>(symengine.real_double(symengine.eval_double(deref(X.thisptr)))))

def eval_complex_double(x):
    cdef Basic X = sympify(x)
    return c2py(<rcp_const_basic>(symengine.complex_double(symengine.eval_complex_double(deref(X.thisptr)))))

have_mpfr = False
have_mpc = False
have_piranha = False
have_flint = False
have_llvm = False

IF HAVE_SYMENGINE_MPFR:
    have_mpfr = True
    def eval_mpfr(x, long prec):
        cdef Basic X = sympify(x)
        cdef symengine.mpfr_class a = symengine.mpfr_class(prec)
        symengine.eval_mpfr(a.get_mpfr_t(), deref(X.thisptr), symengine.MPFR_RNDN)
        return c2py(<rcp_const_basic>(symengine.real_mpfr(symengine.std_move_mpfr(a))))

IF HAVE_SYMENGINE_MPC:
    have_mpc = True
    def eval_mpc(x, long prec):
        cdef Basic X = sympify(x)
        cdef symengine.mpc_class a = symengine.mpc_class(prec)
        symengine.eval_mpc(a.get_mpc_t(), deref(X.thisptr), symengine.MPFR_RNDN)
        return c2py(<rcp_const_basic>(symengine.complex_mpc(symengine.std_move_mpc(a))))

IF HAVE_SYMENGINE_PIRANHA:
    have_piranha = True

IF HAVE_SYMENGINE_FLINT:
    have_flint = True

IF HAVE_SYMENGINE_LLVM:
    have_llvm = True

def require(obj, t):
    if not isinstance(obj, t):
        raise TypeError("{} required. {} is of type {}".format(t, obj, type(obj)))

def eval(x, long prec):
    if prec <= 53:
        return eval_complex_double(x)
    else:
        IF HAVE_SYMENGINE_MPC:
            return eval_mpc(x, prec)
        ELSE:
            raise ValueError("Precision %s is only supported with MPC" % prec)

def eval_real(x, long prec):
    if prec <= 53:
        return eval_double(x)
    else:
        IF HAVE_SYMENGINE_MPFR:
            return eval_mpfr(x, prec)
        ELSE:
            raise ValueError("Precision %s is only supported with MPFR" % prec)

def probab_prime_p(n, reps = 25):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return symengine.probab_prime_p(deref(symengine.rcp_static_cast_Integer(_n.thisptr)), reps) >= 1

isprime = probab_prime_p

def nextprime(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return c2py(<rcp_const_basic>(symengine.nextprime(deref(symengine.rcp_static_cast_Integer(_n.thisptr)))))

def gcd(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<rcp_const_basic>(symengine.gcd(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def lcm(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<rcp_const_basic>(symengine.lcm(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def gcd_ext(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    cdef RCP[const symengine.Integer] g, s, t
    symengine.gcd_ext(symengine.outArg_Integer(g), symengine.outArg_Integer(s), symengine.outArg_Integer(t),
        deref(symengine.rcp_static_cast_Integer(_a.thisptr)), deref(symengine.rcp_static_cast_Integer(_b.thisptr)))
    return (c2py(<rcp_const_basic>s), c2py(<rcp_const_basic>t), c2py(<rcp_const_basic>g))

igcdex = gcd_ext

def mod(a, b):
    if b == 0:
        raise ZeroDivisionError
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<rcp_const_basic>(symengine.mod(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def quotient(a, b):
    if b == 0:
        raise ZeroDivisionError
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return c2py(<rcp_const_basic>(symengine.quotient(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))))

def quotient_mod(a, b):
    if b == 0:
        raise ZeroDivisionError
    cdef RCP[const symengine.Integer] q, r
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    symengine.quotient_mod(symengine.outArg_Integer(q), symengine.outArg_Integer(r),
        deref(symengine.rcp_static_cast_Integer(_a.thisptr)), deref(symengine.rcp_static_cast_Integer(_b.thisptr)))
    return (c2py(<rcp_const_basic>q), c2py(<rcp_const_basic>r))

def mod_inverse(a, b):
    cdef RCP[const symengine.Integer] inv
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    cdef int ret_val = symengine.mod_inverse(symengine.outArg_Integer(inv),
        deref(symengine.rcp_static_cast_Integer(_a.thisptr)), deref(symengine.rcp_static_cast_Integer(_b.thisptr)))
    if ret_val == 0:
        return None
    return c2py(<rcp_const_basic>inv)

def crt(rem, mod):
    cdef symengine.vec_integer _rem, _mod
    cdef Basic _a
    cdef cppbool ret_val
    for i in range(len(rem)):
        _a = sympify(rem[i])
        require(_a, Integer)
        _rem.push_back(symengine.rcp_static_cast_Integer(_a.thisptr))
        _a = sympify(mod[i])
        require(_a, Integer)
        _mod.push_back(symengine.rcp_static_cast_Integer(_a.thisptr))

    cdef RCP[const symengine.Integer] c
    ret_val = symengine.crt(symengine.outArg_Integer(c), _rem, _mod)
    if not ret_val:
        return None
    return c2py(<rcp_const_basic>c)

def fibonacci(n):
    if n < 0 :
        raise NotImplementedError
    return c2py(<rcp_const_basic>(symengine.fibonacci(n)))

def fibonacci2(n):
    if n < 0 :
        raise NotImplementedError
    cdef RCP[const symengine.Integer] f1, f2
    symengine.fibonacci2(symengine.outArg_Integer(f1), symengine.outArg_Integer(f2), n)
    return [c2py(<rcp_const_basic>f1), c2py(<rcp_const_basic>f2)]

def lucas(n):
    if n < 0 :
        raise NotImplementedError
    return c2py(<rcp_const_basic>(symengine.lucas(n)))

def lucas2(n):
    if n < 0 :
        raise NotImplementedError
    cdef RCP[const symengine.Integer] f1, f2
    symengine.lucas2(symengine.outArg_Integer(f1), symengine.outArg_Integer(f2), n)
    return [c2py(<rcp_const_basic>f1), c2py(<rcp_const_basic>f2)]

def binomial(n, k):
    if k < 0:
        raise ArithmeticError
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    return c2py(<rcp_const_basic>symengine.binomial(deref(symengine.rcp_static_cast_Integer(_n.thisptr)), k))

def factorial(n):
    if n < 0:
        raise ArithmeticError
    return c2py(<rcp_const_basic>(symengine.factorial(n)))

def divides(a, b):
    cdef Basic _a = sympify(a)
    cdef Basic _b = sympify(b)
    require(_a, Integer)
    require(_b, Integer)
    return symengine.divides(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_b.thisptr)))

def factor(n, B1 = 1.0):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)), B1)
    if (ret_val == 1):
        return c2py(<rcp_const_basic>f)
    else:
        return None

def factor_lehman_method(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor_lehman_method(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    if (ret_val == 1):
        return c2py(<rcp_const_basic>f)
    else:
        return None

def factor_pollard_pm1_method(n, B = 10, retries = 5):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor_pollard_pm1_method(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)), B, retries)
    if (ret_val == 1):
        return c2py(<rcp_const_basic>f)
    else:
        return None

def factor_pollard_rho_method(n, retries = 5):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] f
    cdef int ret_val = symengine.factor_pollard_rho_method(symengine.outArg_Integer(f),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)), retries)
    if (ret_val == 1):
        return c2py(<rcp_const_basic>f)
    else:
        return None

def prime_factors(n):
    cdef symengine.vec_integer factors
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    symengine.prime_factors(factors, deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    s = []
    for i in range(factors.size()):
        s.append(c2py(<rcp_const_basic>(factors[i])))
    return s

def prime_factor_multiplicities(n):
    cdef symengine.vec_integer factors
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    symengine.prime_factors(factors, deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    cdef Basic r
    dict = {}
    for i in range(factors.size()):
        r = c2py(<rcp_const_basic>(factors[i]))
        if (r not in dict):
            dict[r] = 1
        else:
            dict[r] += 1
    return dict

def bernoulli(n):
    if n < 0:
        raise ArithmeticError
    return c2py(<rcp_const_basic>(symengine.bernoulli(n)))

def primitive_root(n):
    cdef RCP[const symengine.Integer] g
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef cppbool ret_val = symengine.primitive_root(symengine.outArg_Integer(g),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    if ret_val == 0:
        return None
    return c2py(<rcp_const_basic>g)

def primitive_root_list(n):
    cdef symengine.vec_integer root_list
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    symengine.primitive_root_list(root_list,
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))
    s = []
    for i in range(root_list.size()):
        s.append(c2py(<rcp_const_basic>(root_list[i])))
    return s

def totient(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] m = symengine.rcp_static_cast_Integer(_n.thisptr)
    return c2py(<rcp_const_basic>symengine.totient(m))

def carmichael(n):
    cdef Basic _n = sympify(n)
    require(_n, Integer)
    cdef RCP[const symengine.Integer] m = symengine.rcp_static_cast_Integer(_n.thisptr)
    return c2py(<rcp_const_basic>symengine.carmichael(m))

def multiplicative_order(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    cdef RCP[const symengine.Integer] n1 = symengine.rcp_static_cast_Integer(_n.thisptr)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] o
    cdef cppbool c = symengine.multiplicative_order(symengine.outArg_Integer(o),
        a1, n1)
    if not c:
        return None
    return c2py(<rcp_const_basic>o)

def legendre(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    return symengine.legendre(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def jacobi(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    return symengine.jacobi(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def kronecker(a, n):
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    require(_n, Integer)
    require(_a, Integer)
    return symengine.kronecker(deref(symengine.rcp_static_cast_Integer(_a.thisptr)),
        deref(symengine.rcp_static_cast_Integer(_n.thisptr)))

def nthroot_mod(a, n, m):
    cdef RCP[const symengine.Integer] root
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    require(_n, Integer)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] n1 = symengine.rcp_static_cast_Integer(_n.thisptr)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    cdef cppbool ret_val = symengine.nthroot_mod(symengine.outArg_Integer(root), a1, n1, m1)
    if not ret_val:
        return None
    return c2py(<rcp_const_basic>root)

def nthroot_mod_list(a, n, m):
    cdef symengine.vec_integer root_list
    cdef Basic _n = sympify(n)
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    require(_n, Integer)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] n1 = symengine.rcp_static_cast_Integer(_n.thisptr)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    symengine.nthroot_mod_list(root_list, a1, n1, m1)
    s = []
    for i in range(root_list.size()):
        s.append(c2py(<rcp_const_basic>(root_list[i])))
    return s

def sqrt_mod(a, p, all_roots=False):
    if all_roots:
        return nthroot_mod_list(a, 2, p)
    return nthroot_mod(a, 2, p)

def powermod(a, b, m):
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    cdef Number _b = sympify(b)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    cdef RCP[const symengine.Number] b1 = symengine.rcp_static_cast_Number(_b.thisptr)
    cdef RCP[const symengine.Integer] root
    cdef cppbool ret_val = symengine.powermod(symengine.outArg_Integer(root), a1, b1, m1)
    if ret_val == 0:
        return None
    return c2py(<rcp_const_basic>root)

def powermod_list(a, b, m):
    cdef Basic _a = sympify(a)
    cdef Basic _m = sympify(m)
    cdef Number _b = sympify(b)
    require(_a, Integer)
    require(_m, Integer)
    cdef RCP[const symengine.Integer] a1 = symengine.rcp_static_cast_Integer(_a.thisptr)
    cdef RCP[const symengine.Integer] m1 = symengine.rcp_static_cast_Integer(_m.thisptr)
    cdef RCP[const symengine.Number] b1 = symengine.rcp_static_cast_Number(_b.thisptr)
    cdef symengine.vec_integer v

    symengine.powermod_list(v, a1, b1, m1)
    s = []
    for i in range(v.size()):
        s.append(c2py(<rcp_const_basic>(v[i])))
    return s

def has_symbol(obj, symbol=None):
    cdef Basic b = _sympify(obj)
    cdef Basic s = _sympify(symbol)
    require(s, Symbol)
    if (not symbol):
        return not b.free_symbols.empty()
    else:
        return symengine.has_symbol(deref(b.thisptr),
                deref(symengine.rcp_static_cast_Symbol(s.thisptr)))

def ccode(expr):
    cdef Basic expr_ = sympify(expr)
    return symengine.ccode(deref(expr_.thisptr)).decode("utf-8")


def interval(start, end, left_open=False, right_open=False):
    if isinstance(start, NegativeInfinity):
        left_open = True
    if isinstance(end, Infinity):
        right_open = True
    cdef Number start_ = sympify(start)
    cdef Number end_ = sympify(end)
    cdef cppbool left_open_ = left_open
    cdef cppbool right_open_ = right_open
    cdef RCP[const symengine.Number] n1 = symengine.rcp_static_cast_Number(start_.thisptr)
    cdef RCP[const symengine.Number] n2 = symengine.rcp_static_cast_Number(end_.thisptr)
    return c2py(symengine.interval(n1, n2, left_open_, right_open_))


def emptyset():
    return c2py(<rcp_const_basic>(symengine.emptyset()))


def universalset():
    return c2py(<rcp_const_basic>(symengine.universalset()))


def finiteset(*args):
    cdef symengine.set_basic s
    cdef Basic e_
    for e in args:
        e_ = sympify(e)
        s.insert(<rcp_const_basic>(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.finiteset(s)))


def contains(expr, sset):
    cdef Basic expr_ = sympify(expr)
    cdef Set sset_ = sympify(sset)
    cdef RCP[const symengine.Set] s = symengine.rcp_static_cast_Set(sset_.thisptr)
    return c2py(<rcp_const_basic>(symengine.contains(expr_.thisptr, s)))


def set_union(*args):
    cdef symengine.set_set s
    cdef Set e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Set(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.set_union(s)))


def set_intersection(*args):
    cdef symengine.set_set s
    cdef Set e_
    for e in args:
        e_ = sympify(e)
        s.insert(symengine.rcp_static_cast_Set(e_.thisptr))
    return c2py(<rcp_const_basic>(symengine.set_intersection(s)))


def set_complement(universe, container):
    cdef Set universe_ = sympify(universe)
    cdef Set container_ = sympify(container)
    cdef RCP[const symengine.Set] u = symengine.rcp_static_cast_Set(universe_.thisptr)
    cdef RCP[const symengine.Set] c = symengine.rcp_static_cast_Set(container_.thisptr)
    return c2py(<rcp_const_basic>(symengine.set_complement(u, c)))


def set_complement_helper(container, universe):
    cdef Set container_ = sympify(container)
    cdef Set universe_ = sympify(universe)
    cdef RCP[const symengine.Set] c = symengine.rcp_static_cast_Set(container_.thisptr)
    cdef RCP[const symengine.Set] u = symengine.rcp_static_cast_Set(universe_.thisptr)
    return c2py(<rcp_const_basic>(symengine.set_complement_helper(c, u)))


def conditionset(sym, condition):
    cdef Basic sym_ = sympify(sym)
    cdef Boolean condition_ = sympify(condition)
    cdef RCP[const symengine.Boolean] c = symengine.rcp_static_cast_Boolean(condition_.thisptr)
    return c2py(<rcp_const_basic>(symengine.conditionset(sym_.thisptr, c)))


def imageset(sym, expr, base):
    cdef Basic sym_ = sympify(sym)
    cdef Basic expr_ = sympify(expr)
    cdef Set base_ = sympify(base)
    cdef RCP[const symengine.Set] b = symengine.rcp_static_cast_Set(base_.thisptr)
    return c2py(<rcp_const_basic>(symengine.imageset(sym_.thisptr, expr_.thisptr, b)))


def solve(f, sym, domain=None):
    cdef Basic f_ = sympify(f)
    cdef Basic sym_ = sympify(sym)
    require(sym_, Symbol)
    cdef RCP[const symengine.Symbol] x = symengine.rcp_static_cast_Symbol(sym_.thisptr)
    if domain is None:
        return c2py(<rcp_const_basic>(symengine.solve(f_.thisptr, x)))
    cdef Set domain_ = sympify(domain)
    cdef RCP[const symengine.Set] d = symengine.rcp_static_cast_Set(domain_.thisptr)
    return c2py(<rcp_const_basic>(symengine.solve(f_.thisptr, x, d)))


def cse(exprs):
    cdef symengine.vec_basic vec
    cdef symengine.vec_pair replacements
    cdef symengine.vec_basic reduced_exprs
    cdef Basic b
    for expr in exprs:
        b = sympify(expr)
        vec.push_back(b.thisptr)
    symengine.cse(replacements, reduced_exprs, vec)
    return (vec_pair_to_list(replacements), vec_basic_to_list(reduced_exprs))


# Turn on nice stacktraces:
symengine.print_stack_on_segfault()
