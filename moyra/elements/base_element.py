from functools import cache
import warnings
import sympy as sym
from sympy.abc import t
import sympy.physics.mechanics as me
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

    def to_symbolic_model(self, legacy=False):
        Lag = sym.Matrix([self.ke-self.pe])
        D = sym.Matrix([self.rdf])
        # legacy method is a lot slower but can produce more compact results
        Q_v = (self.M.diff(t))*self.qd if not legacy else Lag.jacobian(self.qd).diff(t).T
        M = self.M if not legacy else Q_v.jacobian(self.qdd).T
        Q_v = Q_v if not legacy else me.msubs(Q_v,{i:0 for i in self.qdd})
        term_2 = Lag.jacobian(self.q).T
        term_3 = D.jacobian(self.qd).T
        f = Q_v - term_2 + term_3
        return SymbolicModel(self.q,M,f,self.ke,self.pe)      

    @property
    def elementname(self):
        return self._elementname
    @elementname.setter
    def elementname(self,value):
        self._elementname = value

    def __str__(self):
        return f'{self.elementname}:{self.__class__.__name__}'