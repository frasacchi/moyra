import sympy as sym
from .base_element import BaseElement

class Spring(BaseElement):
    def __init__(self,q,deflection,spring_constant,name="default"):
        self.__k = spring_constant
        self.__z = deflection
        super(Spring, self).__init__(q,name)
        
    ke = property(lambda self:0)
    rdf = property(lambda self:0)
    M = property(lambda self: sym.zeros(len(self.q)))

    def pe(self):
        return sym.Rational(1,2)*self.__k*self.__z**2
        
