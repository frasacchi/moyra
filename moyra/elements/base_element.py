from functools import cache
import warnings
import sympy as sym
from sympy.abc import t

from moyra.symbolic_model import SymbolicModel

class BaseElement:
    def __init__(self,q,name="default") -> None:
        self._q = q
        self._elementname = name

    @property
    def q(self):
        return self._q

    @property
    @cache
    def qd(self):
        return self.q.diff(t)
    
    @property
    @cache
    def qdd(self):
        return self.q.diff(t,2)

    @property
    def ke(self):
        warnings.warn(f'ke not implemented in the ele {self}')
        return 0

    @property
    def pe(self):
        warnings.warn(f'pe not implemented in the ele {self}')
        return 0

    @property
    def rdf(self):
        warnings.warn(f'rdf not implemented in the ele {self}')
        return 0

    @property
    def M(self):
        warnings.warn(f'M not implemented in the ele {self}')
        return sym.zeros(len(self.q))
    
    def to_symbolic_model(self):
        Lag = sym.Matrix([self.ke-self.pe])
        D = sym.Matrix([self.rdf])
        term_1 = Lag.jacobian(self.qd).diff(t).T.expand()
        term_2 = Lag.jacobian(self.q).T
        term_3 = D.jacobian(self.qd).T
        f = sym.expand(term_1.subs({j:0 for j in self.qdd})) - term_2 + term_3
        return SymbolicModel(self.q,self.M,f,self.ke,self.pe)      

    @property
    def elementname(self):
        return self._elementname
    @elementname.setter
    def elementname(self,value):
        self._elementname = value

    def __str__(self):
        return f'{self.elementname}:{self.__class__.__name__}'