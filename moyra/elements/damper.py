from functools import cache
import sympy as sym
from .base_element import BaseElement

class Damper(BaseElement):
    def __init__(self,q,velocity,damping_constant,name = "default"):
        self.__c = damping_constant
        self.__z_dot = velocity
        super(Damper, self).__init__(q,name)
    
    ke = property(lambda self: 0)
    pe = property(lambda self: 0)
    M = property(lambda self: sym.zeros(len(self.q)))

    @property
    @cache
    def rdf(self):
        return sym.Rational(1,2)*self.__c*self.__z_dot**2