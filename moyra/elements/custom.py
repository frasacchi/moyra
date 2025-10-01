import sympy as sym
from functools import cache
from .base_element import BaseElement
import sympy.physics.mechanics as me
from moyra.symbolic_model import SymbolicModel
from sympy.abc import t

class Custom(BaseElement):
    def __init__(self,q,ke,pe,name="default"):
        self._ke = ke
        self._pe = pe
        super(Custom, self).__init__(q,name)
        
    ke = property(lambda self:self._ke)
    pe = property(lambda self:self._pe)
    rdf = property(lambda self:0)
    M = property(lambda self: sym.zeros(len(self.q)))
    
    def to_symbolic_model(self, legacy=False):
        Lag = sym.Matrix([self.ke-self.pe])
        D = sym.Matrix([self.rdf])
        # legacy method is a lot slower but can produce more compact results
        Q_v = Lag.jacobian(self.qd).diff(t).T
        M = Q_v.jacobian(self.qdd).T
        Q_v = me.msubs(Q_v,{i:0 for i in self.qdd})
        term_2 = Lag.jacobian(self.q).T
        term_3 = D.jacobian(self.qd).T
        f = Q_v - term_2 + term_3
        return SymbolicModel(self.q,M,f,self.ke,self.pe)  
        
